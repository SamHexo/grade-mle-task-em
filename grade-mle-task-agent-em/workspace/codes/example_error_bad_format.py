"""
Error example 2 — submission written with wrong format.

The script runs without crashing and writes a CSV to the correct path,
but the columns don't match the sample submission (wrong column name +
only half the rows). The agent detects the mismatch during validation,
marks format_valid=False, and reports success=False with score=None.

Standalone usage:
    python example_error_bad_format.py \
        --train-dataset-path ../../ventilator-pressure-prediction/train.csv \
        --test-dataset-path  ../../ventilator-pressure-prediction/test.csv \
        --output-submission-path ./submission_example_error_bad_format.csv \
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
    print(f"[example_error_bad_format] running...")

    test = pd.read_csv(args.test_dataset_path)

    # Intentionally wrong: wrong column name, and only half the rows
    bad_submission = pd.DataFrame({
        "id": test["id"].iloc[: len(test) // 2],
        "predicted_pressure": 0.0,   # should be "pressure"
    })

    args.output_submission_path.parent.mkdir(parents=True, exist_ok=True)
    bad_submission.to_csv(args.output_submission_path, index=False)
    print(f"  Written {len(bad_submission)} rows to {args.output_submission_path}")
    print("  (intentionally wrong: bad column name + missing rows)")


if __name__ == "__main__":
    main()
