# ================================================================
# 0. Imports
# ================================================================
import pickle, json, math, pathlib, random
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

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
    def __init__(self, n_judges: int, hidden: int = 16, monotone: bool = True, initial_bias: Optional[float] = None):
        super().__init__()
        self.f = nn.ModuleList(
            [OneDimNet(hidden, monotone) for _ in range(n_judges)]
        )
        if initial_bias is not None:
            self.bias = nn.Parameter(torch.tensor(initial_bias, dtype=torch.float32))
        else:
            self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_judges)
        parts = [f_i(x[:, i:i+1]) for i, f_i in enumerate(self.f)]
        return torch.stack(parts, dim=-1).sum(-1) + self.bias   # -> (batch,1)

    # ---------- interpretability helpers ----------
    @torch.no_grad()
    def explain_sample(self, scores: List[float]):
        """
        Returns dict {judge_i: contribution_i, bias: b, total: ŷ}
        """
        x = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
        contribs = [f(x[:, i:i+1]).item() for i, f in enumerate(self.f)]
        total = sum(contribs) + self.bias.item()
        return {"contribs": contribs, "bias": self.bias.item(), "prediction": total}


class SingleLayerMLP(nn.Module):
    """
    A simple single-hidden-layer MLP.
    Prediction: ŷ = W_2 * ReLU(W_1 * x + b_1) + b_2
    """
    def __init__(self, n_judges: int, hidden_dim: int = 64, initial_output_bias: Optional[float] = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_judges, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Init weights
        for layer_idx, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    if layer_idx == 2 and initial_output_bias is not None: # Last linear layer
                        nn.init.constant_(layer.bias, initial_output_bias)
                    else:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_judges)
        return self.net(x)  # -> (batch, 1)

    # ---------- interpretability helper (Partial Dependence Plot) ----------
    @torch.no_grad()
    def plot_partial_dependence(self, typical_input: torch.Tensor, judge_idx: int, score_range=(0., 2.), points=100):
        """
        Plots how the MLP's output changes as one judge's score varies,
        while other judges' scores are held constant at their 'typical_input' values.

        Args:
            typical_input (torch.Tensor): A 1D tensor of shape (n_judges,)
                                        representing typical scores for all judges.
            judge_idx (int): The index of the judge whose score to vary.
            score_range (tuple): (min_score, max_score) for the judge being varied.
            points (int): Number of points to plot in the score_range.
        """
        if not (0 <= judge_idx < typical_input.shape[0]):
            raise ValueError(f"judge_idx {judge_idx} is out of bounds for n_judges={typical_input.shape[0]}")

        s_values = torch.linspace(score_range[0], score_range[1], points)
        predictions = []

        # Create a batch of inputs where only the target judge's score varies
        # typical_input is (n_judges), batch_inputs will be (points, n_judges)
        batch_inputs = typical_input.unsqueeze(0).repeat(points, 1) # Shape: (points, n_judges)
        batch_inputs[:, judge_idx] = s_values

        preds = self.forward(batch_inputs) # Shape: (points, 1)
        predictions = preds.squeeze().cpu().numpy()

        plt.plot(s_values.cpu().numpy(), predictions)
        plt.xlabel(f"Judge {judge_idx} Score")
        plt.ylabel("Model Prediction")
        plt.title(f"Partial Dependence Plot for Judge {judge_idx}")
        # plt.show() # Removed to allow multiple plots if called in a loop


# ================================================================
# 3. Training pipeline
# ================================================================
def train_model(
    data_pkl: str,
    model_type: str = "gam",
    hidden: int = 16,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    monotone: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
    save_path: str = "judge_aggregator.pt",
    device: Optional[torch.device] = None,
    normalize_inputs: bool = True
):
    # ---------- Reproducibility & Device ----------
    torch.manual_seed(seed); random.seed(seed)
    if device is None:
        # Simplified device check, will revisit MPS if necessary
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # ---------- Construct actual save path ----------
    # save_path argument is the base name/prefix from args.out
    path_obj = pathlib.Path(save_path) 
    final_filename_stem = path_obj.stem
    final_output_dir = path_obj.parent
    actual_save_filename = f"{final_filename_stem}_{model_type.lower()}.pt"
    actual_save_path_str = str(final_output_dir / actual_save_filename)
    print(f"Model will be saved to: {actual_save_path_str}")

    # ---------- Load data ----------
    X, y = load_pickle_dataset(data_pkl)
    n_judges = X.shape[1]
    # train/val split ------------------------------------------------
    idxs = list(range(len(X))); random.shuffle(idxs)
    split = int(len(X)*(1-val_split))
    idx_train, idx_val = idxs[:split], idxs[split:]

    if len(idx_train) == 0:
        raise ValueError(
            f"Training set is empty ({len(idx_train)} samples from {len(X)} total) "
            f"with val_split={val_split}. Adjust val_split or increase dataset size."
        )
    
    X_train_orig, y_train = X[idx_train], y[idx_train] # Keep original for typical scores
    X_val_orig,   y_val   = X[idx_val],   y[idx_val]

    # ---------- Normalization (optional) ----------
    norm_mean, norm_std = None, None
    X_train_processed = X_train_orig.clone()
    X_val_processed = X_val_orig.clone() # Will be empty if no val set

    initial_bias_val: Optional[float] = None # For GAMAggregator
    initial_mlp_output_bias_val: Optional[float] = None # For SingleLayerMLP

    if normalize_inputs:
        norm_mean = X_train_orig.mean(dim=0, keepdim=True) # Shape (1, n_judges)
        norm_std = X_train_orig.std(dim=0, keepdim=True)   # Shape (1, n_judges)
        # Avoid division by zero for features with zero variance in training data
        norm_std[norm_std == 0] = 1.0 
        
        X_train_processed = (X_train_orig - norm_mean) / norm_std
        if len(X_val_orig) > 0: # Only normalize val if it exists and norm_mean/std are available
            X_val_processed = (X_val_orig - norm_mean) / norm_std
        print("Applied input normalization based on training set statistics.")
        print(f"  Mean: {norm_mean.squeeze().tolist()}")
        print(f"  Std:  {norm_std.squeeze().tolist()}")

        # It might be a good heuristic always, but let's target the problem.
        target_mean_for_bias = y_train.mean().item()
        initial_bias_val = target_mean_for_bias
        initial_mlp_output_bias_val = target_mean_for_bias
        print(f"Calculated initial bias / MLP output bias based on y_train mean: {target_mean_for_bias:.4f}")
    else:
        print("Input normalization is disabled.")

    ds_train = JudgeDataset(X_train_processed, y_train)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    ds_val, dl_val = None, None # Initialize dl_val here
    # Create validation dataset only if val_split > 0 and there are validation samples
    if val_split > 0.0 and len(idx_val) > 0:
        # Use processed validation data
        ds_val   = JudgeDataset(X_val_processed,   y_val)
        dl_val   = DataLoader(ds_val,   batch_size=batch_size)
    elif val_split > 0.0 and len(idx_val) == 0:
        print(f"Warning: val_split={val_split} but no validation samples found. Training without validation.")
        ds_val = None
        dl_val = None

    # ---------- Model, optim, loss ----------
    if model_type.lower() == "gam":
        model = GAMAggregator(n_judges, hidden, monotone, initial_bias=initial_bias_val)
        print("Training GAM model.")
    elif model_type.lower() == "mlp":
        model = SingleLayerMLP(n_judges, hidden_dim=hidden, initial_output_bias=initial_mlp_output_bias_val)
        print(f"Training SingleLayerMLP model with hidden_dim={hidden}.")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'gam' or 'mlp'.")
    
    model.to(device) # Move model to device

    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Add LR Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)
    loss_fn = nn.MSELoss()

    best_val_loss = math.inf 
    # Calculate typical scores from original, unnormalized training data
    typical_scores_original_scale = X_train_orig.mean(dim=0)
    model_saved_this_run = False

    print_first_batch_debug = True # Debug flag

    for epoch in range(1, epochs+1):
        model.train()
        total_train_loss = 0 
        for batch_idx, (xb, yb) in enumerate(dl_train): # Added batch_idx
            xb, yb = xb.to(device), yb.to(device) 
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)

            if epoch == 1 and batch_idx == 0 and print_first_batch_debug:
                print("\n--- First Batch Debug (Normalization Active) ---")
                print(f"Input xb (shape {xb.shape}, first 5 rows):\n{xb[:5]}")
                print(f"Target yb (shape {yb.shape}, first 5 rows):\n{yb[:5]}")
                print(f"Predictions pred (shape {pred.shape}, first 5 rows):\n{pred[:5]}")
                print(f"Loss for first batch: {loss.item()}")
                print("---------------------------------------------")
                # print_first_batch_debug = False # Optional: print only once ever

            loss.backward()
            opt.step()
            total_train_loss += loss.item()*len(xb)
        avg_train_loss = total_train_loss / len(ds_train)

        # ---- validation ----
        current_val_loss = float('nan') # Default if no validation
        if ds_val and dl_val: # Check if validation is possible
            model.eval()
            with torch.no_grad():
                val_loss_sum = 0
                for xb_val, yb_val in dl_val:
                    xb_val, yb_val = xb_val.to(device), yb_val.to(device) # Move batch to device
                    pred_val = model(xb_val)
                    val_loss_sum += loss_fn(pred_val, yb_val).item()*len(xb_val)
                if len(ds_val) > 0:
                    current_val_loss = val_loss_sum / len(ds_val)
                else: 
                    current_val_loss = float('nan')

            print(f"Epoch {epoch:3d}: train MSE={avg_train_loss:.4f} | val MSE={current_val_loss:.4f} | LR={opt.param_groups[0]['lr']:.1e}")
            
            if not math.isnan(current_val_loss):
                scheduler.step(current_val_loss) # Step the scheduler based on validation loss
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    torch.save(model.state_dict(), actual_save_path_str)
                    model_saved_this_run = True
        else: # No validation (val_split was 0 or resulted in empty ds_val)
            print(f"Epoch {epoch:3d}: train MSE={avg_train_loss:.4f} | (No validation) | LR={opt.param_groups[0]['lr']:.1e}")
            # Save model at the final epoch if no validation-based saving occurred
            if epoch == epochs:
                torch.save(model.state_dict(), actual_save_path_str)

    # ---- Post-training ----
    if val_split > 0.0 and ds_val:
        if model_saved_this_run:
            print(f"Best val MSE={best_val_loss:.4f} (model saved to {actual_save_path_str})")
        else:
            print(f"No model saved based on validation. Best val MSE remained {best_val_loss:.4f}.")
            if epochs > 0: 
                 print(f"Saving model from final epoch to {actual_save_path_str} as fallback.")

    if model_saved_this_run and pathlib.Path(actual_save_path_str).exists():
        print(f"Loading {'best validation' if ds_val and best_val_loss != math.inf else 'final epoch'} model weights from {actual_save_path_str} for return.")
        model.load_state_dict(torch.load(actual_save_path_str, map_location=device))
    elif epochs > 0 : 
        print(f"Warning: No model checkpoint found at {actual_save_path_str}. Returning model from its last training state.")
    
    return model, typical_scores_original_scale, norm_mean, norm_std, dl_val, actual_save_path_str # Added actual_save_path_str

# ================================================================
# 4. Evaluation Function
# ================================================================
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module, # e.g., nn.MSELoss()
    norm_mean: Optional[torch.Tensor] = None,
    norm_std: Optional[torch.Tensor] = None,
    description: str = "Evaluation"
) -> Dict[str, float]:
    """
    Evaluates the model on the given dataloader.
    Inputs are assumed to be ALREADY NORMALIZED if norm_mean/std were used during training
    and the dataloader provides such normalized inputs.
    This function primarily collects predictions and true values for metric calculation.
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            
            # Inputs (xb) are expected to be pre-normalized if normalization was used.
            # The dataloader (e.g., ds_val) should have been created with normalized X.
            pred = model(xb)
            
            loss = loss_fn(pred, yb)
            total_loss += loss.item() * len(xb)
            
            all_preds.append(pred.cpu()) # Move to CPU for sklearn
            all_targets.append(yb.cpu()) # Move to CPU for sklearn

    avg_loss = total_loss / len(dataloader.dataset) # type: ignore[arg-type]
    
    all_preds_tensor = torch.cat(all_preds)
    all_targets_tensor = torch.cat(all_targets)

    mae = float(mean_absolute_error(all_targets_tensor.numpy(), all_preds_tensor.numpy())) # Cast to float
    r2 = float(r2_score(all_targets_tensor.numpy(), all_preds_tensor.numpy())) # Cast to float

    print(f"--- {description} Results ---")
    print(f"  MSE: {avg_loss:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R^2: {r2:.4f}")
    print("-------------------------")
    
    return {"mse": avg_loss, "mae": mae, "r2": r2}

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
    parser.add_argument("--model-type", type=str, default="gam",
                        choices=["gam", "mlp"],
                        help="Type of model to train: 'gam' or 'mlp'. Default: 'gam'.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=16,
                        help="Hidden units. For GAM: in each 1-D sub-network (0 for linear). "
                             "For MLP: size of the hidden layer.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-monotone", action="store_true",
                        help="Disable the ≥0 weight clamp (monotonicity).")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data reserved for validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="agg_model",
                        help="Base filename for saving the model. '_<model_type>.pt' will be appended (e.g., if 'mymodel', then 'mymodel_gam.pt' or 'mymodel_mlp.pt'). Default: 'agg_model'.")
    parser.add_argument("--force-cpu", action="store_true", 
                        help="Force training on CPU even if CUDA is available.")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable input normalization.")
    args = parser.parse_args()

    # ---- Determine Device ----
    if args.force_cpu:
        current_device = torch.device("cpu")
    else:
        # Simplified device check, will revisit MPS if necessary
        if torch.cuda.is_available():
            current_device = torch.device("cuda")
        else:
            current_device = torch.device("cpu")

    # -------- run training ----------
    model, typical_scores_for_pdp, norm_mean_returned, norm_std_returned, returned_dl_val, final_model_save_path = train_model(
        data_pkl=args.data,
        model_type=args.model_type,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        monotone=not args.no_monotone,
        val_split=args.val_split,
        seed=args.seed,
        save_path=args.out,
        device=current_device,
        normalize_inputs=not args.no_normalize # Pass normalization flag
    )

    # -------- Evaluation Step ----------
    if model is not None and returned_dl_val is not None: # Use returned_dl_val
        print("\nStarting evaluation on the validation set...")
        
        eval_loss_fn = nn.MSELoss()
        
        validation_metrics = evaluate_model(
            model=model,
            dataloader=returned_dl_val, # Use returned_dl_val
            device=current_device,
            loss_fn=eval_loss_fn,
            description="Validation Set Evaluation"
        )
    else:
        if model is None:
            print("\nSkipping evaluation: Model training did not return a model.")
        else:
            print("\nSkipping evaluation: No validation data loader available (val_split might be 0).")

    if model is not None:
        print("Finished. Best model weights saved to", final_model_save_path)
    else:
        print("Finished. Training might not have completed successfully or saved a model.")
