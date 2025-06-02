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

# ================================================================
# 2. Evaluation Function for NN Model
# ================================================================
def evaluate_nn_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    description: str = "NN Model Evaluation"
) -> Dict[str, float]:
    model.eval()
    all_preds_nn = []
    all_targets_nn = []
    loss_fn = nn.MSELoss() 
    total_loss_nn = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_loss_nn += loss.item() * len(xb)
            all_preds_nn.append(pred.cpu())
            all_targets_nn.append(yb.cpu())

    if not all_preds_nn: # Handle empty dataloader case
        print(f"--- {description} Results ---")
        print("  No data to evaluate.")
        print("-------------------------")
        return {"mse": float('nan'), "mae": float('nan'), "r2": float('nan')}

    avg_loss_nn = total_loss_nn / len(dataloader.dataset) # type: ignore
    all_preds_tensor_nn = torch.cat(all_preds_nn)
    all_targets_tensor_nn = torch.cat(all_targets_nn)

    mae_nn = float(mean_absolute_error(all_targets_tensor_nn.numpy(), all_preds_tensor_nn.numpy()))
    r2_nn = float(r2_score(all_targets_tensor_nn.numpy(), all_preds_tensor_nn.numpy()))
    
    print(f"--- {description} Results ---")
    print(f"  MSE (from loss_fn): {avg_loss_nn:.4f}")
    print(f"  MAE: {mae_nn:.4f}")
    print(f"  R^2: {r2_nn:.4f}")
    print("-------------------------")
    
    return {"mse": avg_loss_nn, "mae": mae_nn, "r2": r2_nn}

# ================================================================
# 3. Naive Mean Baseline Evaluation
# ================================================================
def evaluate_naive_mean_baseline(
    X_original: torch.Tensor,
    y_true: torch.Tensor,
    judge_score_max: float = 4.0,
    target_score_max: float = 10.0
) -> Dict[str, float]:
    if X_original.numel() == 0 or y_true.numel() == 0: # Check for empty tensors
        print("--- Naive Mean Baseline Results ---")
        print("  No data to evaluate.")
        print("----------------------------------")
        return {"mse": float('nan'), "mae": float('nan'), "r2": float('nan')}

    if X_original.ndim == 1: X_original = X_original.unsqueeze(0)
    if y_true.ndim == 1: y_true = y_true.unsqueeze(1)

    mean_judge_scores = X_original.mean(dim=1, keepdim=True)
    
    scaling_factor = target_score_max / judge_score_max if judge_score_max != 0 else 1.0
    y_pred_naive = mean_judge_scores * scaling_factor
    
    y_true_np = y_true.cpu().numpy()
    y_pred_naive_np = y_pred_naive.cpu().numpy()
    
    mse_naive = mean_squared_error(y_true_np, y_pred_naive_np)
    mae_naive = mean_absolute_error(y_true_np, y_pred_naive_np)
    r2_naive = r2_score(y_true_np, y_pred_naive_np)
    
    print("--- Naive Mean Baseline Results ---")
    print(f"  Scaling factor used: {scaling_factor:.2f} (target_max={target_score_max} / judge_max={judge_score_max})")
    print(f"  MSE: {mse_naive:.4f}")
    print(f"  MAE: {mae_naive:.4f}")
    print(f"  R^2: {r2_naive:.4f}")
    print("----------------------------------")
    
    return {"mse": float(mse_naive), "mae": float(mae_naive), "r2": float(r2_naive)}

# ================================================================
# 4. Main Execution
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained NN model and compare with baselines.")
    parser.add_argument("--data", required=True, help="Path to the pickle dataset.")
    parser.add_argument("--model-path", required=True, help="Path to the saved trained model (.pt file).")
    parser.add_argument("--model-type", type=str, required=True, choices=["gam", "mlp"], help="Type of model that was trained.")
    parser.add_argument("--hidden", type=int, required=True, help="Hidden units for the model architecture. For GAM: in each 1-D sub-network. For MLP: size of the hidden layer.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of data used for validation during training (to reconstruct the same val set).")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for train/val split.")
    parser.add_argument("--force-cpu", action="store_true", help="Force evaluation on CPU.")
    parser.add_argument("--no-normalize-input", action="store_true", help="Set if the model was trained WITHOUT input normalization.")
    parser.add_argument("--no-monotone", action="store_true", help="Set if GAM model was trained WITHOUT monotonicity constraint (relevant for GAM instantiation).")
    parser.add_argument("--judge-max-score", type=float, default=4.0, help="Maximum score a judge can give (for naive baseline scaling).")
    parser.add_argument("--target-max-score", type=float, default=10.0, help="Maximum target score (human preference, for naive baseline scaling).")
    parser.add_argument("--eval-batch-size", type=int, default=256, help="Batch size for NN model evaluation.")


    args = parser.parse_args()

    # --- Device setup ---
    torch.manual_seed(args.seed); random.seed(args.seed)
    if args.force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Using device: {device}")

    # --- Load full dataset ---
    X_full, y_full = load_pickle_dataset(args.data)
    n_features = X_full.shape[1]

    # --- Recreate train/validation split ---
    num_samples = len(X_full)
    indices = list(range(num_samples))
    random.shuffle(indices) # Shuffles based on seed from args
    
    split_idx = int(num_samples * (1 - args.val_split))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    
    X_train_orig, y_train_orig = X_full[train_indices], y_full[train_indices] # y_train_orig not directly used here but good for completeness
    X_val_orig, y_val_orig     = X_full[val_indices], y_full[val_indices]
    
    print(f"Loaded data: {num_samples} total samples. {len(train_indices)} train, {len(val_indices)} val (using seed {args.seed}, split {args.val_split}).")

    # --- 1. Evaluate Naive Mean Baseline ---
    naive_metrics = {}
    if len(val_indices) > 0:
        print("\nEvaluating Naive Mean Baseline on Validation Set...")
        naive_metrics = evaluate_naive_mean_baseline(
            X_val_orig, 
            y_val_orig,
            judge_score_max=args.judge_max_score,
            target_score_max=args.target_max_score
        )
    else:
        print("\nSkipping Naive Mean Baseline on Validation Set (empty).")

    # --- 2. Evaluate Trained NN Model ---
    nn_metrics = {}
    if len(val_indices) > 0:
        print("\nEvaluating Trained NN Model on Validation Set...")
        
        X_val_processed = X_val_orig.clone()
        if not args.no_normalize_input:
            if len(X_train_orig) > 0:
                norm_mean = X_train_orig.mean(dim=0, keepdim=True)
                norm_std = X_train_orig.std(dim=0, keepdim=True)
                norm_std[norm_std == 0] = 1.0 
                X_val_processed = (X_val_orig - norm_mean) / norm_std
                print("Applied input normalization to validation data based on training set statistics.")
                print(f"  Calculated Mean for normalization: {norm_mean.squeeze().tolist()}")
                print(f"  Calculated Std for normalization:  {norm_std.squeeze().tolist()}")
            else: # Should not happen if val_split < 1.0 and num_samples > 0
                print("Warning: Training set is empty, cannot calculate normalization stats. Using unnormalized data for NN.")
        else:
            print("Input normalization is disabled for NN model (as per --no-normalize-input flag).")

        val_dataset = JudgeDataset(X_val_processed, y_val_orig)
        val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)

        monotone_param_for_gam = not args.no_monotone
        if args.model_type == "gam":
            nn_model = GAMAggregator(n_judges=n_features, hidden=args.hidden, monotone=monotone_param_for_gam) 
        elif args.model_type == "mlp":
            nn_model = SingleLayerMLP(n_judges=n_features, hidden_dim=args.hidden)
        else: # Should be caught by argparse choices
            raise ValueError(f"Internal error: Unknown model_type '{args.model_type}'")

        try:
            nn_model.load_state_dict(torch.load(args.model_path, map_location=device))
            nn_model.to(device)
            print(f"Loaded trained {args.model_type.upper()} model from {args.model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {args.model_path}")
            exit(1)
        except Exception as e:
            print(f"Error loading model state_dict from {args.model_path}: {e}")
            exit(1)
        
        if len(val_dataloader) > 0:
             nn_metrics = evaluate_nn_model(
                model=nn_model,
                dataloader=val_dataloader,
                device=device,
                description=f"{args.model_type.upper()} Model on Validation Set"
            )
        else: # Should be redundant due to outer len(val_indices) check, but defensive.
            print("NN Model: Validation dataloader is empty, skipping evaluation.")
            
    else:
        print("\nSkipping NN Model evaluation (validation set is empty).")

    # --- 3. Print Summary Comparison ---
    print("\n--- Overall Comparison (Validation Set) ---")
    if nn_metrics and not all(math.isnan(v) for v in nn_metrics.values()):
        print(f"NN Model ({args.model_type.upper()}):")
        print(f"  MSE: {nn_metrics.get('mse', float('nan')):.4f}")
        print(f"  MAE: {nn_metrics.get('mae', float('nan')):.4f}")
        print(f"  R^2: {nn_metrics.get('r2', float('nan')):.4f}")
    else:
        print(f"NN Model ({args.model_type.upper()}): Not evaluated or no data in validation set.")
        
    if naive_metrics and not all(math.isnan(v) for v in naive_metrics.values()):
        print("Naive Mean Baseline:")
        print(f"  MSE: {naive_metrics.get('mse', float('nan')):.4f}")
        print(f"  MAE: {naive_metrics.get('mae', float('nan')):.4f}")
        print(f"  R^2: {naive_metrics.get('r2', float('nan')):.4f}")
    else:
        print("Naive Mean Baseline: Not evaluated or no data in validation set.")
    print("----------------------------------------")

    print("\nEvaluation script finished.")
