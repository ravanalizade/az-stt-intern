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


BASE = "https://huggingface.co/datasets/fsicoli/common_voice_17_0/resolve/main"
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
    p.add_argument("--lang", default="az")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--output", default="data/cv_az_test")
    p.add_argument("--cache-dir", default="data/_cv_cache")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    n_shards_path = download(f"{BASE}/n_shards.json", cache / "n_shards.json")
    with open(n_shards_path) as f:
        n_shards = json.load(f)
    if args.lang not in n_shards:
        print(f"Language '{args.lang}' not found.", file=sys.stderr)
        return 1
    n = n_shards[args.lang][args.split]
    print(f"Language={args.lang} split={args.split} shards={n}")

    tsv_path = download(
        f"{BASE}/transcript/{args.lang}/{args.split}.tsv",
        cache / f"{args.lang}_{args.split}.tsv",
    )
    df = pd.read_csv(tsv_path, sep="\t", quoting=3)
    print(f"Transcripts: {len(df)} rows, columns: {list(df.columns)}")

    df["path"] = df["path"].astype(str).apply(lambda p: p if p.endswith(".mp3") else p + ".mp3")
    path_to_meta = {row["path"]: row for _, row in df.iterrows()}

    # Decide which filenames we want before touching any audio. Shuffle
    # deterministically, then download shards until we've collected enough.
    target_paths = list(path_to_meta.keys())
    random.Random(args.seed).shuffle(target_paths)
    wanted = set(target_paths[: args.max_samples] if args.max_samples > 0 else target_paths)
    print(f"Want {len(wanted)} samples")

    out_audio_dir = Path(args.output) / "audio"
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    collected = []
    for shard_idx in range(n):
        if not wanted:
            break
        tar_url = f"{BASE}/audio/{args.lang}/{args.split}/{args.lang}_{args.split}_{shard_idx}.tar"
        tar_path = download(tar_url, cache / f"{args.lang}_{args.split}_{shard_idx}.tar")
        with tarfile.open(tar_path, "r") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                fname = os.path.basename(member.name)
                if fname not in wanted:
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                raw = f.read()
                # Decode mp3 -> 16kHz mono float32 immediately. soundfile handles
                # mp3 since 0.13; fall back to librosa for older versions.
                try:
                    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
                except Exception:
                    audio, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != TARGET_SR:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

                out_wav = out_audio_dir / fname.replace(".mp3", ".wav")
                sf.write(out_wav, audio, TARGET_SR, subtype="PCM_16")

                meta = path_to_meta[fname]
                collected.append({
                    "audio_path": str(out_wav.resolve()),
                    "sentence": meta["sentence"],
                    "client_id": meta.get("client_id", ""),
                    "path": fname,
                })
                wanted.discard(fname)
                if not wanted:
                    break
        print(f"  shard {shard_idx}: collected {len(collected)}, remaining {len(wanted)}")

    if not collected:
        print("No audio collected — something is wrong.", file=sys.stderr)
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
