import argparse
import io
import json
import os
import random
import sys
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
import soundfile as sf
import librosa


BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/az_az"
TARGET_SR = 16_000


def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"  GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--max-samples", type=int, default=0, help="0 = all available")
    p.add_argument("--output", default="data/fleurs_az_test")
    p.add_argument("--cache-dir", default="data/_fleurs_cache")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    tsv_path = download(f"{BASE}/{args.split}.tsv", cache / f"az_{args.split}.tsv")
    # FLEURS TSV: no header, columns are
    # id, file_name, raw_transcription, transcription, _ignored, num_samples, gender
    df = pd.read_csv(
        tsv_path, sep="\t", header=None, quoting=3,
        names=["id", "file_name", "raw_transcription", "transcription", "_x", "num_samples", "gender"],
    )
    print(f"Transcripts: {len(df)} rows")

    file_to_meta = {row["file_name"]: row for _, row in df.iterrows()}

    targets = list(file_to_meta.keys())
    random.Random(args.seed).shuffle(targets)
    if args.max_samples > 0:
        wanted = set(targets[: args.max_samples])
    else:
        wanted = set(targets)
    print(f"Want {len(wanted)} samples")

    tar_path = download(f"{BASE}/audio/{args.split}.tar.gz", cache / f"az_{args.split}.tar.gz")
    out_audio_dir = Path(args.output) / "audio"
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    collected = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not wanted:
                continue
            fname = os.path.basename(member.name)
            if fname not in wanted:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            raw = f.read()
            try:
                audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
            except Exception:
                audio, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

            out_wav = out_audio_dir / fname.replace(".wav", "_16k.wav")
            sf.write(out_wav, audio, TARGET_SR, subtype="PCM_16")

            meta = file_to_meta[fname]
            collected.append({
                "audio_path": str(out_wav.resolve()),
                "sentence": meta["transcription"],
                "raw_sentence": meta["raw_transcription"],
                "gender": meta["gender"],
                "path": fname,
            })
            wanted.discard(fname)

    if not collected:
        print("No audio collected.", file=sys.stderr)
        return 1

    out_meta = Path(args.output) / "metadata.jsonl"
    with open(out_meta, "w", encoding="utf-8") as f:
        for r in collected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(collected)} samples to {args.output}")
    print(f"Sample: {collected[0]['sentence']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
