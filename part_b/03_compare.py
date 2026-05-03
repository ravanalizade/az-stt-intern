import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import jiwer
import soundfile as sf
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


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
    p.add_argument("--data", default="data/cv_az_test")
    p.add_argument("--base-model", default="openai/whisper-small")
    p.add_argument("--adapter", default="checkpoints/whisper-small-az-lora/best")
    p.add_argument("--base-predictions", default="results/part_a_predictions.jsonl")
    p.add_argument("--ft-predictions-out", default="results/part_b_ft_predictions.jsonl")
    p.add_argument("--comparison-out", default="results/part_b_comparison.md")
    p.add_argument("--metrics-out", default="results/part_b_metrics.json")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--language", default="azerbaijani")
    p.add_argument("--device", default=None)
    return p.parse_args()


def pick_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_metadata(data_dir):
    rows = []
    with open(Path(data_dir) / "metadata.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_audio(rows):
    arrays = []
    for r in rows:
        a, sr = sf.read(r["audio_path"], dtype="float32")
        assert sr == 16_000
        if a.ndim > 1:
            a = a.mean(axis=1)
        arrays.append(a)
    return arrays


def run_inference(model, processor, rows, arrays, device, language, batch_size):
    forced = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    gen_kwargs = {"forced_decoder_ids": forced, "num_beams": 1, "max_new_tokens": 225}
    out = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), batch_size)):
            batch_rows = rows[start : start + batch_size]
            batch_audio = arrays[start : start + batch_size]
            inputs = processor(batch_audio, sampling_rate=16_000, return_tensors="pt", padding=True)
            features = inputs.input_features.to(device)
            pred_ids = model.generate(features, **gen_kwargs)
            hyps = processor.batch_decode(pred_ids, skip_special_tokens=True)
            for r, hyp, a in zip(batch_rows, hyps, batch_audio):
                out.append({
                    "audio_id": r["path"],
                    "reference": r["sentence"],
                    "hypothesis": hyp.strip(),
                    "audio_seconds": len(a) / 16_000,
                })
    return out


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def corpus_metrics(rows):
    refs = [normalize(r["reference"]) for r in rows]
    hyps = [normalize(r["hypothesis"]) for r in rows]
    return jiwer.wer(refs, hyps), jiwer.cer(refs, hyps)


def main():
    args = parse_args()
    device = pick_device(args.device)

    rows = load_metadata(args.data)
    print(f"{len(rows)} test samples")
    print("Loading audio")
    arrays = load_audio(rows)

    base_path = Path(args.base_predictions)
    if base_path.exists():
        base_rows = load_jsonl(base_path)
        if len(base_rows) != len(rows):
            print(f"WARN: {len(base_rows)} base predictions vs {len(rows)} samples — re-running")
            base_rows = None
    else:
        base_rows = None

    if base_rows is None:
        print("Running base model inference")
        processor = WhisperProcessor.from_pretrained(args.base_model)
        base_model = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device).eval()
        base_rows = run_inference(base_model, processor, rows, arrays, device, args.language, args.batch_size)
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Running fine-tuned model (adapter: {args.adapter})")
    processor = WhisperProcessor.from_pretrained(args.adapter)
    base_for_ft = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    ft_model = PeftModel.from_pretrained(base_for_ft, args.adapter).to(device).eval()
    t0 = time.time()
    ft_rows = run_inference(ft_model, processor, rows, arrays, device, args.language, args.batch_size)
    print(f"  done in {time.time() - t0:.1f}s")

    write_jsonl(ft_rows, args.ft_predictions_out)
    print(f"Wrote {args.ft_predictions_out}")

    base_wer, base_cer = corpus_metrics(base_rows)
    ft_wer, ft_cer = corpus_metrics(ft_rows)

    metrics = {
        "n_samples": len(rows),
        "base":      {"wer": round(base_wer, 4), "cer": round(base_cer, 4)},
        "finetuned": {"wer": round(ft_wer, 4),   "cer": round(ft_cer, 4)},
        "delta":     {"wer": round(ft_wer - base_wer, 4),
                      "cer": round(ft_cer - base_cer, 4)},
        "relative_wer_improvement": round((base_wer - ft_wer) / base_wer * 100, 2) if base_wer else 0,
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n  Base:      WER={base_wer*100:.2f}%  CER={base_cer*100:.2f}%")
    print(f"  Finetuned: WER={ft_wer*100:.2f}%  CER={ft_cer*100:.2f}%")
    print(f"  Δ WER: {(ft_wer - base_wer)*100:+.2f} pp   Δ CER: {(ft_cer - base_cer)*100:+.2f} pp")

    base_by_id = {r["audio_id"]: r for r in base_rows}
    diffs = []
    for ft in ft_rows:
        b = base_by_id.get(ft["audio_id"])
        if b is None:
            continue
        ref = ft["reference"]
        b_wer = jiwer.wer(normalize(ref), normalize(b["hypothesis"])) if normalize(ref) else 0
        f_wer = jiwer.wer(normalize(ref), normalize(ft["hypothesis"])) if normalize(ref) else 0
        diffs.append({
            "audio_id": ft["audio_id"], "reference": ref,
            "base_hyp": b["hypothesis"], "ft_hyp": ft["hypothesis"],
            "base_wer": b_wer, "ft_wer": f_wer, "delta": f_wer - b_wer,
        })
    diffs.sort(key=lambda d: d["delta"])
    improved = [d for d in diffs if d["delta"] < 0][:10]
    regressed = [d for d in diffs if d["delta"] > 0][-10:][::-1]

    md = []
    md.append("# Part B — Base vs Fine-tuned Comparison\n")
    md.append(f"Test set: {len(rows)} samples (same as Part A)\n")
    md.append("## Aggregate metrics\n")
    md.append("| Model | WER | CER |")
    md.append("|---|---|---|")
    md.append(f"| Base (whisper-small) | {base_wer*100:.2f}% | {base_cer*100:.2f}% |")
    md.append(f"| Fine-tuned (LoRA)    | {ft_wer*100:.2f}% | {ft_cer*100:.2f}% |")
    md.append(f"| **Δ**                | **{(ft_wer - base_wer)*100:+.2f} pp** | **{(ft_cer - base_cer)*100:+.2f} pp** |")
    md.append(f"\nRelative WER improvement: **{metrics['relative_wer_improvement']:.2f}%**\n")

    def fmt(title, items):
        out = [f"## {title}\n"]
        for i, d in enumerate(items, 1):
            out.append(f"**{i}.** base WER={d['base_wer']*100:.0f}% → ft WER={d['ft_wer']*100:.0f}% (Δ {d['delta']*100:+.0f} pp)")
            out.append(f"- REF:  {d['reference']}")
            out.append(f"- BASE: {d['base_hyp']}")
            out.append(f"- FT:   {d['ft_hyp']}")
            out.append("")
        return "\n".join(out)

    md.append(fmt("Top 10 largest improvements", improved))
    md.append(fmt("Top 10 regressions (where fine-tuning hurt)", regressed))

    with open(args.comparison_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Wrote {args.comparison_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
