# ml_utils.py
"""
Shared ML / PyTorch helpers for the Tennis ML Intelligence Project.

Usage examples
--------------
from ml_utils import (
    load_period_tensors, TennisMatchDataset, split_by_date,
    SimpleLogit, train_epoch, evaluate
)

# Load tensors
payload = load_period_tensors("D:\\Tennis\\data\\ml\\tensors_ERA_1991_PLUS.pt")
ds = TennisMatchDataset(payload["X"], payload["y"])

# Simple model
model = SimpleLogit(in_dim=payload["n_features"])

"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def load_period_tensors(path_pt: str | Path):
    return torch.load(Path(path_pt), map_location="cpu")

class TennisMatchDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert X.ndim == 2 and y.ndim == 2, "Expecting 2D tensors"
        self.X = X.float()
        self.y = y.float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def split_by_date(meta_parquet_path: str | Path, train_end_year: int, val_end_year: int):
    """
    Time-based split using years in the meta parquet.
    Returns boolean masks (train_mask, val_mask, test_mask).
    """
    import pandas as pd
    df = pd.read_parquet(meta_parquet_path)
    years = df["date"].dt.year
    train_mask = years <= train_end_year
    val_mask = (years > train_end_year) & (years <= val_end_year)
    test_mask = years > val_end_year
    return df, train_mask.values, val_mask.values, test_mask.values

class SimpleLogit(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.lin(x))

def train_epoch(model, dl, opt, loss_fn=nn.BCELoss()):
    model.train()
    total = 0.0
    n = 0
    for xb, yb in dl:
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(n,1)

@torch.no_grad()
def evaluate(model, dl, loss_fn=nn.BCELoss()):
    model.eval()
    total = 0.0
    n = 0
    for xb, yb in dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(n,1)
