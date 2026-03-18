"""
Error example 3 — submission written to a hardcoded path (ignores --output-submission-path).

The script runs without crashing and writes a perfectly valid submission,
but to a hardcoded filename ("marla_submission.csv" in the working directory)
instead of the path the agent passed via --output-submission-path.
The agent looks for the file at the expected path, finds nothing, and reports
success=False with score=None.

Standalone usage:
    python example_error_wrong_name.py \
        --train-dataset-path ../../ventilator-pressure-prediction/train.csv \
        --test-dataset-path  ../../ventilator-pressure-prediction/test.csv \
        --output-submission-path ./submission_example_error_wrong_name.csv \
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
    print(f"[example_error_wrong_name] running...")
    print(f"  (intentionally ignoring --output-submission-path)")

    train = pd.read_csv(args.train_dataset_path)
    test = pd.read_csv(args.test_dataset_path)

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_COL].values
    X_test = test[FEATURE_COLS].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Hardcoded output name — agent will not find it at the expected path
    hardcoded_path = Path("marla_submission.csv")
    submission = pd.DataFrame({"id": test["id"], "pressure": preds})
    submission.to_csv(hardcoded_path, index=False)
    print(f"  Written {len(submission)} rows to {hardcoded_path}  ← wrong path!")


if __name__ == "__main__":
    main()
