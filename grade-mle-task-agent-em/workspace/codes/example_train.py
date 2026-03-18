"""
Example training script — sklearn Ridge regression (no GPU required).

Mandatory arguments (passed automatically by the agent):
    --train-dataset-path       Path to train.csv
    --test-dataset-path        Path to test.csv  (no pressure column)
    --output-submission-path   Where to write submission_<name>.csv

Optional:
    --epochs INT               Number of SGD passes (default: 5).
                               Ridge itself is closed-form; epochs controls
                               how many times the scaler is re-fit (kept for
                               API parity with torch scripts).

Standalone usage:
    python example_train.py \
        --train-dataset-path  ../../ventilator-pressure-prediction/train.csv \
        --test-dataset-path   ../../ventilator-pressure-prediction/test.csv \
        --output-submission-path ./my_submission.csv \
        --epochs 5
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["R", "C", "time_step", "u_in", "u_out"]
TARGET_COL = "pressure"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dataset-path", required=True, type=Path)
    p.add_argument("--test-dataset-path", required=True, type=Path)
    p.add_argument("--output-submission-path", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[example_train] epochs={args.epochs} (unused by Ridge, accepted for API parity)")
    print(f"  train : {args.train_dataset_path}")
    print(f"  test  : {args.test_dataset_path}")
    print(f"  output: {args.output_submission_path}")

    # Load data
    train = pd.read_csv(args.train_dataset_path)
    test = pd.read_csv(args.test_dataset_path)

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_COL].values
    X_test = test[FEATURE_COLS].values

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Write submission
    args.output_submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"id": test["id"], "pressure": preds})
    submission.to_csv(args.output_submission_path, index=False)
    print(f"  Submission written: {len(submission)} rows → {args.output_submission_path}")


if __name__ == "__main__":
    main()
