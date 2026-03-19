"""
Data preparation script for Google Brain - Ventilator Pressure Prediction.

Two modes:

  1. From mle-bench cache (recommended — no raw CSV needed):
       python prepare_data.py --from-mlebench-cache

  2. From raw Kaggle train.csv (produces the identical split):
       python prepare_data.py --raw-train-csv /path/to/kaggle/train.csv

The mle-bench cache is at:
  ~/Library/Caches/mle-bench/data/ventilator-pressure-prediction/prepared/
  └── public/   train.csv, test.csv, sample_submission.csv
  └── private/  test.csv   ← ground truth labels

Both modes write to the same output directory (default: this folder):
  train.csv             training data
  test.csv              public test (no labels)
  sample_submission.csv expected submission format
  private_test.csv      private test with labels (used by grade.py)
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

MLEBENCH_CACHE = Path.home() / "Library/Caches/mle-bench/data/ventilator-pressure-prediction/prepared"


def from_mlebench_cache(output_dir: Path):
    public = MLEBENCH_CACHE / "public"
    private = MLEBENCH_CACHE / "private"

    for src in [public / "train.csv", public / "test.csv", public / "sample_submission.csv", private / "test.csv"]:
        if not src.exists():
            print(f"Error: {src} not found. Run mle-bench prepare first.")
            raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(public / "train.csv",             output_dir / "train.csv")
    shutil.copy(public / "test.csv",              output_dir / "test.csv")
    shutil.copy(public / "sample_submission.csv", output_dir / "sample_submission.csv")
    shutil.copy(private / "test.csv",             output_dir / "private_test.csv")

    print(f"  train.csv             ← {public / 'train.csv'}")
    print(f"  test.csv              ← {public / 'test.csv'}")
    print(f"  sample_submission.csv ← {public / 'sample_submission.csv'}")
    print(f"  private_test.csv      ← {private / 'test.csv'}")
    print(f"\nDone. Files written to: {output_dir}")


def from_raw_csv(raw_train_csv: Path, output_dir: Path):
    dtypes = {
        "id": "int32",
        "breath_id": "int32",
        "R": "int8",
        "C": "int8",
        "time_step": "float64",
        "u_in": "float64",
        "u_out": "int8",
        "pressure": "float64",
    }

    print(f"Reading {raw_train_csv}...")
    old_train = pd.read_csv(raw_train_csv, dtype=dtypes)

    # Split by breath_id to avoid leakage (same as mle-bench prepare.py)
    groups = [df.index.tolist() for _, df in old_train.groupby("breath_id")]
    train_idx, test_idx = train_test_split(groups, test_size=0.1, random_state=0)

    train_idx = [idx for sublist in train_idx for idx in sublist]
    test_idx = [idx for sublist in test_idx for idx in sublist]

    new_train = old_train.loc[train_idx].copy()
    new_test = old_train.loc[test_idx].copy()

    new_train["id"] = range(1, len(new_train) + 1)
    new_test["id"] = range(1, len(new_test) + 1)

    assert set(new_train["breath_id"]).isdisjoint(set(new_test["breath_id"])), \
        "Test set contains breath_ids that are in the train set"

    public_test = new_test.drop(columns=["pressure"])
    sample_submission = public_test[["id"]].copy()
    sample_submission["pressure"] = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    new_train.to_csv(output_dir / "train.csv", index=False, float_format="%.10g")
    public_test.to_csv(output_dir / "test.csv", index=False, float_format="%.10g")
    sample_submission.to_csv(output_dir / "sample_submission.csv", index=False, float_format="%.10g")
    new_test.to_csv(output_dir / "private_test.csv", index=False, float_format="%.10g")

    print(f"  train.csv:             {len(new_train):,} rows")
    print(f"  test.csv:              {len(public_test):,} rows (no labels)")
    print(f"  private_test.csv:      {len(new_test):,} rows (with labels)")
    print(f"  sample_submission.csv: {len(sample_submission):,} rows")
    print(f"\nDone. Files written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ventilator-pressure-prediction data")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--from-mlebench-cache",
        action="store_true",
        help=f"Copy from mle-bench cache at {MLEBENCH_CACHE}",
    )
    group.add_argument(
        "--raw-train-csv",
        type=Path,
        help="Path to the raw Kaggle train.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to write output files (default: same folder as this script)",
    )
    args = parser.parse_args()

    if args.from_mlebench_cache:
        from_mlebench_cache(args.output_dir)
    else:
        if not args.raw_train_csv.exists():
            print(f"Error: {args.raw_train_csv} does not exist")
            raise SystemExit(1)
        from_raw_csv(args.raw_train_csv, args.output_dir)
