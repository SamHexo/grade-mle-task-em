"""
Example training script — PyTorch MLP (GPU if available, CPU fallback).

Mandatory arguments (passed automatically by the agent):
    --train-dataset-path       Path to train.csv
    --test-dataset-path        Path to test.csv  (no pressure column)
    --output-submission-path   Where to write submission_<name>.csv

Optional:
    --epochs INT               Number of training epochs (default: 5).

Standalone usage:
    python example_train_torch.py \
        --train-dataset-path  ../../ventilator-pressure-prediction/train.csv \
        --test-dataset-path   ../../ventilator-pressure-prediction/test.csv \
        --output-submission-path ./my_submission_torch.csv \
        --epochs 5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


FEATURE_COLS = ["R", "C", "time_step", "u_in", "u_out"]
TARGET_COL = "pressure"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dataset-path", required=True, type=Path)
    p.add_argument("--test-dataset-path", required=True, type=Path)
    p.add_argument("--output-submission-path", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=5)
    return p.parse_args()


class MLP(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[example_train_torch] epochs={args.epochs}, device={device}")
    print(f"  train : {args.train_dataset_path}")
    print(f"  test  : {args.test_dataset_path}")
    print(f"  output: {args.output_submission_path}")

    # Load data
    train = pd.read_csv(args.train_dataset_path)
    test = pd.read_csv(args.test_dataset_path)

    X_train = train[FEATURE_COLS].values.astype(np.float32)
    y_train = train[TARGET_COL].values.astype(np.float32)
    X_test = test[FEATURE_COLS].values.astype(np.float32)

    # Normalize with train stats
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Tensors
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    # Model
    model = MLP(in_features=len(FEATURE_COLS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Train
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        print(f"  epoch {epoch}/{args.epochs}  loss={total_loss / len(train_ds):.4f}")

    # Predict
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).to(device)
        preds = model(X_t).cpu().numpy()

    # Write submission
    args.output_submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"id": test["id"], "pressure": preds})
    submission.to_csv(args.output_submission_path, index=False)
    print(f"  Submission written: {len(submission)} rows → {args.output_submission_path}")


if __name__ == "__main__":
    main()
