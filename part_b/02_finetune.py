import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import soundfile as sf
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


PUNCT_RE = re.compile(r"[^\w\s'’ʼ-]", flags=re.UNICODE)
WS_RE = re.compile(r"\s+")


def normalize(text):
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = PUNCT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/cv_az_finetune")
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--output-dir", default="checkpoints/whisper-small-az-lora")
    p.add_argument("--language", default="azerbaijani")
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-history-out", default="results/training_log.json")
    return p.parse_args()


def load_split(split_dir):
    meta = Path(split_dir) / "metadata.jsonl"
    rows = []
    with open(meta, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_hf_dataset(rows, processor):

    inputs, labels = [], []
    for r in rows:
        audio, sr = sf.read(r["audio_path"], dtype="float32")
        assert sr == 16_000
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        feats = processor.feature_extractor(audio, sampling_rate=16_000).input_features[0]
        ids = processor.tokenizer(r["sentence"]).input_ids
        inputs.append(feats)
        labels.append(ids)
    return Dataset.from_dict({"input_features": inputs, "labels": labels})


@dataclass
class WhisperDataCollator:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def build_compute_metrics(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        pred_norm = [normalize(s) for s in pred_str]
        label_norm = [normalize(s) for s in label_str]
        pairs = [(r, h) for r, h in zip(label_norm, pred_norm) if r]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}
        refs, hyps = zip(*pairs)
        return {"wer": jiwer.wer(list(refs), list(hyps)), "cer": jiwer.cer(list(refs), list(hyps))}
    return compute_metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    train_rows = load_split(Path(args.data) / "train")
    val_rows = load_split(Path(args.data) / "validation")
    print(f"train: {len(train_rows)} | validation: {len(val_rows)}")

    print(f"Loading {args.model}")
    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Tokenizing train set")
    train_ds = build_hf_dataset(train_rows, processor)
    print("Tokenizing validation set")
    val_ds = build_hf_dataset(val_rows, processor)

    collator = WhisperDataCollator(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        gradient_checkpointing=True,
        fp16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        report_to=[],
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=build_compute_metrics(processor),
        tokenizer=processor.feature_extractor,
    )

    print("Training")
    trainer.train()

    final_dir = Path(args.output_dir) / "best"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"Saved best model to {final_dir}")

    log_out = Path(args.log_history_out)
    log_out.parent.mkdir(parents=True, exist_ok=True)
    with open(log_out, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)
    print(f"Wrote training log to {log_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
