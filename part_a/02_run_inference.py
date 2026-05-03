import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/cv_az_test")
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--output", default="results/part_a_predictions.jsonl")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--language", default="azerbaijani")
    p.add_argument("--device", default=None)
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


def pick_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_metadata(data_dir):
    meta_path = Path(data_dir) / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Run 01_load_dataset.py first.")
    rows = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    args = parse_args()
    device = pick_device(args.device)
    use_fp16 = args.fp16 and device == "cuda"
    print(f"Device: {device}{' (fp16)' if use_fp16 else ''}")

    rows = load_metadata(args.data)
    print(f"Loaded {len(rows)} samples from {args.data}")

    print(f"Loading model: {args.model}")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.to(device)
    if use_fp16:
        model.half()
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
    gen_kwargs = {"forced_decoder_ids": forced_decoder_ids, "num_beams": args.num_beams, "max_new_tokens": 225}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_audio_s = 0.0
    t0 = time.time()
    with output_path.open("w", encoding="utf-8") as f, torch.inference_mode():
        for start in tqdm(range(0, len(rows), args.batch_size), desc="Inference"):
            batch = rows[start : start + args.batch_size]
            arrays = []
            for r in batch:
                audio, sr = sf.read(r["audio_path"], dtype="float32")
                assert sr == 16_000, f"Expected 16kHz, got {sr} for {r['audio_path']}"
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                arrays.append(audio)

            inputs = processor(arrays, sampling_rate=16_000, return_tensors="pt", padding=True)
            features = inputs.input_features.to(device)
            if use_fp16:
                features = features.half()

            pred_ids = model.generate(features, **gen_kwargs)
            hyps = processor.batch_decode(pred_ids, skip_special_tokens=True)

            for i, hyp in enumerate(hyps):
                row = {
                    "audio_id": batch[i]["path"],
                    "client_id": batch[i].get("client_id", ""),
                    "reference": batch[i]["sentence"],
                    "hypothesis": hyp.strip(),
                    "audio_seconds": len(arrays[i]) / 16_000,
                }
                total_audio_s += row["audio_seconds"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    rtf = elapsed / total_audio_s if total_audio_s else 0
    print(f"Done in {elapsed:.1f}s | {total_audio_s:.1f}s of audio | RTF={rtf:.2f}")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
