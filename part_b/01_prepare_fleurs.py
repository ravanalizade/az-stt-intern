import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--output", default="data/fleurs_az_finetune")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_loader(split, max_samples, output, seed):
    cmd = [
        sys.executable, "part_a/01_load_fleurs.py",
        "--split", split,
        "--max-samples", str(max_samples),
        "--output", output,
        "--seed", str(seed),
    ]
    print(f"$ {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)


def main():
    args = parse_args()
    out = Path(args.output)
    run_loader("train", args.n_train, str(out / "train"), args.seed)
    run_loader("dev",   args.n_val,   str(out / "validation"), args.seed)
    print(f"\nReady: {out}/train and {out}/validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
