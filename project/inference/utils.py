import pickle
import argparse
import random
import math
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

# ================================================================
# 0. Data Loading (adapted from train_gam.py)
# ================================================================
def load_pickle_dataset(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(path, "rb") as f:
        raw: List[Dict] = pickle.load(f)

    assert isinstance(raw, list) and len(raw) > 0, "Empty or invalid pickle!"
    n_judges = len(raw[0]["scores"])
    for idx, ex in enumerate(raw):
        assert len(ex["scores"]) == n_judges, f"row {idx} has inconsistent length"
        assert all(isinstance(x, (int,float)) for x in ex["scores"]), "Non-numeric score!"
        assert isinstance(ex["target"], (int,float)), "Non-numeric target!"

    X = torch.tensor([ex["scores"] for ex in raw], dtype=torch.float32)
    y = torch.tensor([[ex["target"]] for ex in raw], dtype=torch.float32)
    return X, y

class JudgeDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X, self.y = X, y
    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ================================================================
# 1. Model Definitions (adapted from train_gam.py for loading)
# ================================================================
class OneDimNet(nn.Module):
    def __init__(self, hidden: int = 16, monotone: bool = True):
        super().__init__()
        self.monotone = monotone and hidden > 0 # Store for forward, though loaded weights are key
        if hidden == 0:
            self.net = nn.Linear(1, 1, bias=False)
        else:
            self.net = nn.Sequential(
                nn.Linear(1, hidden, bias=False),
                nn.ReLU(),
                nn.Linear(hidden, 1, bias=False)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional monotonicity clamp, primarily for training/fine-tuning
        # For pure evaluation, loaded weights determine behavior.
        if self.monotone:
            for p in self.parameters():
                if p.requires_grad: # Only clamp parameters that are part of the optimization
                    p.data.clamp_(min=0)
        return self.net(x)

class GAMAggregator(nn.Module):
    def __init__(self, n_judges: int, hidden: int = 16, monotone: bool = True, initial_bias: Optional[float] = None):
        super().__init__()
        self.f = nn.ModuleList(
            [OneDimNet(hidden, monotone) for _ in range(n_judges)]
        )
        # Bias will be loaded from state_dict. Initialize for architecture.
        self.bias = nn.Parameter(torch.tensor(initial_bias if initial_bias is not None else 0.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [f_i(x[:, i:i+1]) for i, f_i in enumerate(self.f)]
        return torch.stack(parts, dim=-1).sum(-1) + self.bias

class SingleLayerMLP(nn.Module):
    def __init__(self, n_judges: int, hidden_dim: int = 64, initial_output_bias: Optional[float] = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_judges, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Biases and weights will be loaded from state_dict.
        # The initialization below is not strictly necessary when loading a saved model.
        # if initial_output_bias is not None and self.net[-1].bias is not None:
        #      nn.init.constant_(self.net[-1].bias, initial_output_bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
