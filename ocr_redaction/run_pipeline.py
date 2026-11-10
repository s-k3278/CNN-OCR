
"""Convenience entry point for training and evaluation."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    model = Path("models/ocr_crnn.pt")
    train = [
        sys.executable, "-m", "training.train",
        "--data-root", args.data_root,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--threads", str(args.threads),
        "--output", str(model),
    ]
    subprocess.run(train, check=True)

    evaluate = [
        sys.executable, "-m", "training.evaluate",
        "--data-root", args.data_root,
        "--model", str(model),
        "--output", "outputs/evaluation.json",
        "--batch-size", str(args.batch_size),
        "--threads", str(args.threads),
    ]
    subprocess.run(evaluate, check=True)


if __name__ == "__main__":
    main()
