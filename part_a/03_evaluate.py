import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import jiwer


# Azerbaijani uses: a b c ç d e ə f g ğ h x ı i j k q l m n o ö p r s ş t u ü v y z
# Plus the dotted/dotless i pair (i / ı, İ / I) which Unicode lowercase handles
# inconsistently in Python — we keep the default casefold; results are consistent
# as long as we apply the same pipeline to reference and hypothesis.

PUNCT_RE = re.compile(r"[^\w\s'’ʼ-]", flags=re.UNICODE)
WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = PUNCT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def compute_per_sample_wer(ref: str, hyp: str) -> float:
    ref, hyp = normalize(ref), normalize(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.wer(ref, hyp)


def compute_per_sample_cer(ref: str, hyp: str) -> float:
    ref, hyp = normalize(ref), normalize(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.cer(ref, hyp)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="results/part_a_predictions.jsonl")
    p.add_argument("--metrics-out", default="results/part_a_metrics.json")
    p.add_argument("--examples-out", default="results/part_a_best_worst.md")
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    rows = []
    with open(args.predictions, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("No predictions found", file=sys.stderr)
        return 1
    print(f"Loaded {len(rows)} predictions")

    refs_norm = [normalize(r["reference"]) for r in rows]
    hyps_norm = [normalize(r["hypothesis"]) for r in rows]

    # Corpus-level metrics: jiwer aggregates over all sentences, not the mean of
    # per-sentence WERs. This is the standard reporting metric.
    corpus_wer = jiwer.wer(refs_norm, hyps_norm)
    corpus_cer = jiwer.cer(refs_norm, hyps_norm)

    for r in rows:
        r["wer"] = compute_per_sample_wer(r["reference"], r["hypothesis"])
        r["cer"] = compute_per_sample_cer(r["reference"], r["hypothesis"])

    # Mean of per-sample WERs differs from corpus WER (longer sentences carry
    # more weight in corpus WER). Report both.
    mean_wer = sum(r["wer"] for r in rows) / len(rows)
    mean_cer = sum(r["cer"] for r in rows) / len(rows)

    metrics = {
        "n_samples": len(rows),
        "corpus_wer": round(corpus_wer, 4),
        "corpus_cer": round(corpus_cer, 4),
        "mean_per_sample_wer": round(mean_wer, 4),
        "mean_per_sample_cer": round(mean_cer, 4),
        "model_predictions_file": args.predictions,
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Corpus WER: {corpus_wer*100:.2f}%   Corpus CER: {corpus_cer*100:.2f}%")
    print(f"Mean WER:   {mean_wer*100:.2f}%   Mean CER:   {mean_cer*100:.2f}%")

    # Sort by WER. Best = lowest, worst = highest. Tie-break on CER so identical
    # WER=0 samples surface the cleanest reference text.
    rows_sorted = sorted(rows, key=lambda r: (r["wer"], r["cer"]))
    best = rows_sorted[: args.top_k]
    worst = list(reversed(rows_sorted[-args.top_k :]))

    def fmt_block(title, samples):
        lines = [f"### {title}\n"]
        for i, r in enumerate(samples, 1):
            lines.append(f"**{i}.** WER={r['wer']*100:.1f}% | CER={r['cer']*100:.1f}% | {r['audio_seconds']:.1f}s")
            lines.append(f"- REF: {r['reference']}")
            lines.append(f"- HYP: {r['hypothesis']}")
            lines.append("")
        return "\n".join(lines)

    md = [
        "# Part A — Best & Worst Predictions",
        "",
        f"Model predictions: `{args.predictions}`  ",
        f"Samples: {len(rows)}  ",
        f"Corpus WER: **{corpus_wer*100:.2f}%** | Corpus CER: **{corpus_cer*100:.2f}%**",
        "",
        fmt_block(f"Top {args.top_k} best (lowest WER)", best),
        fmt_block(f"Top {args.top_k} worst (highest WER)", worst),
    ]
    with open(args.examples_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Wrote {args.metrics_out}")
    print(f"Wrote {args.examples_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
