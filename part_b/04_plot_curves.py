import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="results/training_log.json")
    p.add_argument("--output", default="results/training_curves.png")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.log, encoding="utf-8") as f:
        log = json.load(f)

    # HF Trainer interleaves three kinds of entries in log_history:
    #   - training step logs:  {"loss": ..., "epoch": ..., "step": ...}
    #   - eval logs:           {"eval_loss": ..., "eval_wer": ..., "epoch": ...}
    #   - final summary:       {"train_runtime": ..., ...}
    train_pts = [(e["epoch"], e["loss"]) for e in log if "loss" in e and "eval_loss" not in e]
    eval_pts = [(e["epoch"], e.get("eval_wer"), e.get("eval_loss"))
                for e in log if "eval_wer" in e]

    if not train_pts or not eval_pts:
        print(f"Not enough data in log: {len(train_pts)} train pts, {len(eval_pts)} eval pts",
              file=sys.stderr)
        return 1

    train_x, train_y = zip(*train_pts)
    eval_x = [e[0] for e in eval_pts]
    eval_wer = [e[1] for e in eval_pts]
    eval_loss = [e[2] for e in eval_pts]

    best_idx = min(range(len(eval_wer)), key=lambda i: eval_wer[i])
    best_epoch = eval_x[best_idx]
    best_wer = eval_wer[best_idx]

    fig, ax_loss = plt.subplots(figsize=(9, 5))
    ax_wer = ax_loss.twinx()

    ax_loss.plot(train_x, train_y, color="#1f77b4", alpha=0.5, linewidth=1, label="Training loss (step)")
    ax_loss.plot(eval_x, eval_loss, color="#1f77b4", marker="o", linewidth=2, label="Validation loss (epoch)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss", color="#1f77b4")
    ax_loss.tick_params(axis="y", labelcolor="#1f77b4")

    ax_wer.plot(eval_x, [w * 100 for w in eval_wer], color="#d62728", marker="s",
                linewidth=2, label="Validation WER")
    ax_wer.set_ylabel("Validation WER (%)", color="#d62728")
    ax_wer.tick_params(axis="y", labelcolor="#d62728")

    ax_loss.axvline(best_epoch, color="green", linestyle="--", alpha=0.6,
                    label=f"Best ckpt @ epoch {best_epoch:.1f} (WER={best_wer*100:.2f}%)")

    lines1, labels1 = ax_loss.get_legend_handles_labels()
    lines2, labels2 = ax_wer.get_legend_handles_labels()
    ax_loss.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.title("Whisper-small (LoRA) fine-tuning on Common Voice Azerbaijani")
    fig.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.output}")
    print(f"Best epoch: {best_epoch:.1f} | WER: {best_wer*100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
