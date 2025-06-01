# ================================================================
# 0. Imports
# ================================================================
import pickle, json, math, pathlib, random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ================================================================
# 1. Data-loading helpers
# ================================================================
"""
Expected Pickle structure
-------------------------
The file should contain *one Python list*, each element a dict:

{
    "scores": [s1, s2, ..., sn],   # list[float]   length = n_judges
    "target": t                    # float         human or "gold" score
}

Example creation:

examples = [
    {"scores":[1.2, 0.8, 2.0, ...], "target":1.7},
    ...
]
with open("train_data.pkl","wb") as f: pickle.dump(examples,f)
"""

def load_pickle_dataset(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(path, "rb") as f:
        raw: List[Dict] = pickle.load(f)

    # Sanity-checks -------------------------------------------------
    assert isinstance(raw, list) and len(raw) > 0, "Empty or invalid pickle!"
    n_judges = len(raw[0]["scores"])
    for idx, ex in enumerate(raw):
        assert len(ex["scores"]) == n_judges, f"row {idx} has inconsistent length"
        assert all(isinstance(x, (int,float)) for x in ex["scores"]), "Non-numeric score!"
        assert isinstance(ex["target"], (int,float)), "Non-numeric target!"

    # Convert to tensors -------------------------------------------
    X = torch.tensor([ex["scores"] for ex in raw], dtype=torch.float32)
    y = torch.tensor([[ex["target"]] for ex in raw], dtype=torch.float32)  # (N,1)
    return X, y


class JudgeDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X, self.y = X, y
    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ================================================================
# 2. Model definition:  Neural Additive Model  (GAM)
# ================================================================
class OneDimNet(nn.Module):
    """
    Learns a 1-D calibration curve f_i(s_i).
    hidden=0  → pure linear   (almost same as logistic regression)
    """
    def __init__(self, hidden: int = 16, monotone: bool = True):
        super().__init__()
        self.monotone = monotone and hidden > 0
        if hidden == 0:
            self.net = nn.Linear(1, 1, bias=False)
        else:
            self.net = nn.Sequential(
                nn.Linear(1, hidden, bias=False),
                nn.ReLU(),
                nn.Linear(hidden, 1, bias=False)
            )
        # init weights small to keep early outputs near 0
        for p in self.net.parameters(): nn.init.normal_(p, mean=0.0, std=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional monotonicity clamp (∂f/∂x ≥ 0) ------------------
        if self.monotone:
            for p in self.parameters(): p.data.clamp_(min=0)
        return self.net(x)   # shape same as x


class GAMAggregator(nn.Module):
    """
    Final prediction:  ŷ = Σ_i f_i(s_i) + bias
    """
    def __init__(self, n_judges: int, hidden: int = 16, monotone: bool = True):
        super().__init__()
        self.f = nn.ModuleList(
            [OneDimNet(hidden, monotone) for _ in range(n_judges)]
        )
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_judges)
        parts = [f_i(x[:, i:i+1]) for i, f_i in enumerate(self.f)]
        return torch.stack(parts, dim=-1).sum(-1) + self.bias   # -> (batch,1)

    # ---------- interpretability helpers ----------
    @torch.no_grad()
    def plot_curves(self, score_range=(0., 2.), points=100):
        s = torch.linspace(*score_range, points).unsqueeze(1)
        plt.figure(figsize=(6,4))
        for idx, f_i in enumerate(self.f):
            plt.plot(s, f_i(s), label=f"judge {idx}")
        plt.xlabel("raw judge score"); plt.ylabel("contribution")
        plt.legend(); plt.title("Learned calibration curves"); plt.show()

    @torch.no_grad()
    def explain_sample(self, scores: List[float]):
        """
        Returns dict {judge_i: contribution_i, bias: b, total: ŷ}
        """
        x = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
        contribs = [f(x[:, i:i+1]).item() for i, f in enumerate(self.f)]
        total = sum(contribs) + self.bias.item()
        return {"contribs": contribs, "bias": self.bias.item(), "prediction": total}


# ================================================================
# 3. Training pipeline
# ================================================================
def train_model(
    data_pkl: str,
    hidden: int = 16,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    monotone: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
    save_path: str = "judge_aggregator.pt"
):
    # ---------- Reproducibility ----------
    torch.manual_seed(seed); random.seed(seed)

    # ---------- Load data ----------
    X, y = load_pickle_dataset(data_pkl)
    n_judges = X.shape[1]
    # train/val split ------------------------------------------------
    idxs = list(range(len(X))); random.shuffle(idxs)
    split = int(len(X)*(1-val_split))
    idx_train, idx_val = idxs[:split], idxs[split:]

    ds_train = JudgeDataset(X[idx_train], y[idx_train])
    ds_val   = JudgeDataset(X[idx_val],   y[idx_val])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size)

    # ---------- Model, optim, loss ----------
    model = GAMAggregator(n_judges, hidden, monotone)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = math.inf
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for xb, yb in dl_train:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()*len(xb)

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            val_loss = sum(loss_fn(model(xb), yb).item()*len(xb) for xb,yb in dl_val) / len(ds_val)

        print(f"Epoch {epoch:3d}: train MSE={total_loss/len(ds_train):.4f} | val MSE={val_loss:.4f}")

        # ---- simple early-stopping ----
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)

    print(f"Best val MSE={best_val:.4f}  (model saved to {save_path})")
    return model


# ================================================================
# 4. Run training  (uncomment to execute)
# ================================================================
# model = train_model(
#     data_pkl="train_data.pkl",
#     hidden=16,
#     epochs=30,
#     batch_size=256,
#     monotone=True
# )

# ================================================================
# 5. Post-hoc interpretability demo
# ================================================================
# if model:                         # after training
#     model.plot_curves(score_range=(0,2))
#     sample_expl = model.explain_sample([1.0]*10)
#     print(json.dumps(sample_expl, indent=2))

# ... existing code ...

# ================================================================
# 6. Command-line entry point
# ================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train (and optionally visualise) the GAM aggregator "
                    "that maps judge scores → human ground-truth.")
    parser.add_argument("--data", required=True,
                        help="Path to the pickle produced by data preprocessing.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=16,
                        help="Hidden units in each 1-D sub-network. "
                             "Set 0 for a pure linear model.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-monotone", action="store_true",
                        help="Disable the ≥0 weight clamp (monotonicity).")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data reserved for validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="agg_model.pt",
                        help="File to save the best-val model weights.")
    parser.add_argument("--plot", action="store_true",
                        help="If set, plot the learned per-judge calibration curves "
                             "after training finishes.")
    args = parser.parse_args()

    # -------- run training ----------
    model = train_model(
        data_pkl=args.data,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        monotone=not args.no_monotone,
        val_split=args.val_split,
        seed=args.seed,
        save_path=args.out
    )

    # -------- optional quick visualisation ----------
    if args.plot:
        # (re-)load best checkpoint to ensure plotted model = best-val
        model.load_state_dict(torch.load(args.out))
        model.plot_curves()

    print("Finished. Best model weights saved to", args.out)
