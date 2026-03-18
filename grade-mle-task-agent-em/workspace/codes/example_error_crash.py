"""
Error example 1 — script crashes mid-execution.

The script starts normally, loads data, then raises an intentional exception.
The agent catches the non-zero exit code and reports the error in the grid JSON
with success=False and score=None.

Standalone usage:
    python example_error_crash.py \
        --train-dataset-path ../../ventilator-pressure-prediction/train.csv \
        --test-dataset-path  ../../ventilator-pressure-prediction/test.csv \
        --output-submission-path ./submission_example_error_crash.csv \
        --epochs 5
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dataset-path", required=True, type=Path)
    p.add_argument("--test-dataset-path", required=True, type=Path)
    p.add_argument("--output-submission-path", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[example_error_crash] loading data...")

    train = pd.read_csv(args.train_dataset_path)
    print(f"  train loaded: {len(train)} rows")

    # Intentional crash — simulates a bug in the user's training code
    raise RuntimeError(
        "Intentional crash: this simulates a bug in the training script "
        "(e.g. NaN in loss, CUDA OOM, wrong tensor shape, etc.)"
    )


if __name__ == "__main__":
    main()
