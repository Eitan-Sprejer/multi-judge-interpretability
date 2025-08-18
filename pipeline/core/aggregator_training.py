"""
Aggregator Training Pipeline

Trains and evaluates aggregation models (GAM and MLP) that combine judge scores
to predict human preference scores.
"""

import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import GAM if available
try:
    from pygam import LinearGAM, s
    HAS_GAM = True
except ImportError:
    HAS_GAM = False
    logging.warning("PyGAM not installed. GAM training will not be available.")

from pipeline.utils.judge_rubrics import JUDGE_RUBRICS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Feature labels for interpretability - Updated for current 10 judges
FEATURE_LABELS = [
    "Truthfulness / Factual Accuracy",
    "Harmlessness / Safety",
    "Helpfulness / Utility",
    "Honesty / Transparency", 
    "Explanatory Depth / Detail",
    "Instruction Following / Compliance",
    "Clarity / Understandability",
    "Conciseness / Efficiency",
    "Logical Consistency / Reasoning",
    "Creativity / Originality"
]

# Current judge order for reference
CURRENT_JUDGES = [
    "truthfulness-judge",
    "harmlessness-judge", 
    "helpfulness-judge",
    "honesty-judge",
    "explanatory-depth-judge",
    "instruction-following-judge",
    "clarity-judge",
    "conciseness-judge",
    "logical-consistency-judge",
    "creativity-judge"
]


def load_training_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load training configuration from JSON file."""
    if config_path is None:
        # Default config path
        config_path = Path(__file__).parent.parent.parent / "config" / "training_config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded training config from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return get_default_config()
    except Exception as e:
        logger.error(f"Error loading config: {e}, using defaults")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default training configuration if config file is not available."""
    return {
        "mlp_training": {
            "medium_scale": {
                "hidden_dim": 64,
                "learning_rate": 0.005,
                "batch_size": 16,
                "n_epochs": 100,
                "early_stopping_patience": 15
            }
        }
    }


def determine_training_scale(n_samples: int) -> str:
    """Determine appropriate training scale based on number of samples."""
    if n_samples <= 100:
        return "small_scale"
    elif n_samples <= 1000:
        return "medium_scale"  
    elif n_samples <= 10000:
        return "large_scale"
    else:
        return "enterprise_scale"


class SingleLayerMLP(nn.Module):
    """Single hidden layer MLP for aggregating judge scores with dropout and regularization."""
    
    def __init__(self, n_judges: int = 10, hidden_dim: int = 64, dropout: float = 0.0):
        super(SingleLayerMLP, self).__init__()
        self.fc1 = nn.Linear(n_judges, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()


class GAMAggregator:
    """Generalized Additive Model aggregator for interpretable judge score combination."""
    
    def __init__(self, n_splines: int = 10, lam: float = 0.6):
        """
        Initialize GAM aggregator.
        
        Args:
            n_splines: Number of splines for each feature
            lam: Lambda regularization parameter
        """
        if not HAS_GAM:
            raise ImportError("PyGAM is required for GAM aggregator. Install with: pip install pygam")
        
        self.n_splines = n_splines
        self.lam = lam
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the GAM model.
        
        Args:
            X: Judge scores array (n_samples, n_judges)
            y: Human preference scores (n_samples,)
        """
        # Create GAM with splines for each feature
        terms = sum([s(i, n_splines=self.n_splines, lam=self.lam) for i in range(X.shape[1])])
        self.model = LinearGAM(terms)
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict human scores from judge scores."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        if self.model is None:
            raise ValueError("Model must be fitted before scoring")
        return self.model.score(X, y)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores for each judge."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        importance = {}
        for i, label in enumerate(FEATURE_LABELS):
            # Use p-value as inverse importance (lower p-value = more important)
            p_value = self.model.statistics_['p_values'][i] if i < len(self.model.statistics_['p_values']) else 1.0
            importance[label] = 1.0 - p_value
        
        return importance


class MLPTrainer:
    """Trainer for MLP aggregation model with early stopping and checkpointing."""
    
    def __init__(
        self,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        n_epochs: int = 100,
        dropout: float = 0.0,
        l2_reg: float = 0.0,
        early_stopping_patience: int = 15,
        min_delta: float = 1e-4,
        device: str = 'cpu'
    ):
        """
        Initialize MLP trainer.
        
        Args:
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            n_epochs: Maximum number of training epochs
            dropout: Dropout probability (0.0 = no dropout)
            l2_reg: L2 regularization strength (0.0 = no regularization)
            early_stopping_patience: Epochs to wait before stopping if no improvement
            min_delta: Minimum change to qualify as improvement
            device: Device to train on ('cpu' or 'cuda')
        """
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.best_model_state = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train the MLP model.
        
        Args:
            X_train: Training judge scores
            y_train: Training human scores
            X_val: Optional validation judge scores
            y_val: Optional validation human scores
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        n_features = X_train.shape[1]
        self.model = SingleLayerMLP(n_judges=n_features, hidden_dim=self.hidden_dim, dropout=self.dropout).to(self.device)
        
        # Loss and optimizer with L2 regularization
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        
        # Training loop with early stopping
        self.model.train()
        train_losses = []
        val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        best_epoch = 0
        
        logger.info(f"Training MLP with early stopping (patience={self.early_stopping_patience})")
        
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation and early stopping
            if X_val is not None and y_val is not None:
                val_loss = self._evaluate(X_val, y_val, criterion)
                val_losses.append(val_loss)
                
                # Check for improvement
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    self.patience_counter = 0
                    best_epoch = epoch + 1
                    logger.info(f"✓ Epoch {epoch+1}/{self.n_epochs}, Train: {avg_train_loss:.4f}, Val: {val_loss:.4f} (Best)")
                else:
                    self.patience_counter += 1
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"  Epoch {epoch+1}/{self.n_epochs}, Train: {avg_train_loss:.4f}, Val: {val_loss:.4f} (Patience: {self.patience_counter}/{self.early_stopping_patience})")
                
                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}. Best validation loss: {self.best_val_loss:.4f} at epoch {best_epoch}")
                    break
            elif (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Restore best model if we have validation data
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model from epoch {best_epoch} (val_loss: {self.best_val_loss:.4f})")
        
        return train_losses, val_losses
    
    def _evaluate(self, X: np.ndarray, y: np.ndarray, criterion) -> float:
        """Evaluate model on given data."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor).item()
        
        self.model.train()
        return loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict human scores from judge scores."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        return outputs.cpu().numpy()
    
    def save_model(self, path: Path):
        """Save model checkpoint."""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hidden_dim': self.hidden_dim,
            'n_judges': self.model.fc1.in_features
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model = SingleLayerMLP(
            n_judges=checkpoint['n_judges'],
            hidden_dim=checkpoint['hidden_dim']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                        save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        # Mark best validation loss
        best_val_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        plt.axvline(x=best_val_epoch, color='r', linestyle='--', alpha=0.7, 
                   label=f'Best Val (Epoch {best_val_epoch})')
        plt.plot(best_val_epoch, best_val_loss, 'ro', markersize=8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with final metrics
    final_train = train_losses[-1]
    if val_losses:
        final_val = val_losses[-1]
        textstr = f'Final Train Loss: {final_train:.4f}\nFinal Val Loss: {final_val:.4f}\nBest Val Loss: {best_val_loss:.4f}'
    else:
        textstr = f'Final Train Loss: {final_train:.4f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_partial_dependence(
    gam_model: GAMAggregator,
    features: List[int],
    title: str,
    n_cols: int = 2,
    save_path: Optional[Path] = None
):
    """
    Plot partial dependence plots for GAM model.
    
    Args:
        gam_model: Trained GAM model
        features: List of feature indices to plot
        title: Plot title
        n_cols: Number of columns in subplot grid
        save_path: Optional path to save figure
    """
    if not HAS_GAM:
        logger.warning("PyGAM not installed. Cannot create partial dependence plots.")
        return
    
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes_flat = axes.flatten()
    
    for idx, feature_idx in enumerate(features):
        ax = axes_flat[idx]
        
        # Generate grid for partial dependence
        XX = gam_model.model.generate_X_grid(term=feature_idx, meshgrid=False)
        x_values = XX[:, feature_idx]
        y_values = gam_model.model.partial_dependence(term=feature_idx, X=XX)
        
        # Plot partial dependence
        ax.plot(x_values, y_values, 'b-', linewidth=2, label='Partial Dependence')
        
        # Add trend line
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
        trend_line = slope * x_values + intercept
        ax.plot(x_values, trend_line, 'r--', linewidth=1.5, alpha=0.8, label='Trend')
        
        # Add statistics
        correlation_text = f'r = {r_value:.3f}'
        if p_value < 0.001:
            correlation_text += '***'
        elif p_value < 0.01:
            correlation_text += '**'
        elif p_value < 0.05:
            correlation_text += '*'
        
        ax.text(0.95, 0.95, correlation_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'{FEATURE_LABELS[feature_idx]}', fontsize=10)
        ax.set_xlabel('Judge Score')
        ax.set_ylabel('Effect on Prediction')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_features, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    plt.show()


def load_and_prepare_data(
    data_path: Path,
    human_score_col: str = 'human_score',
    judge_scores_col: str = 'judge_scores'
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load and prepare data for training.
    
    Args:
        data_path: Path to pickle file with data
        human_score_col: Column name for human scores
        judge_scores_col: Column name for judge scores
        
    Returns:
        Tuple of (dataframe, X features, y labels)
    """
    logger.info(f"Loading data from {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Convert to DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Handle different data formats
    if 'human_feedback' in data.columns and human_score_col not in data.columns:
        # Extract score from human_feedback dict
        data[human_score_col] = data['human_feedback'].apply(
            lambda x: x['score'] if isinstance(x, dict) and 'score' in x else None
        )
    
    # Filter valid data
    data = data[data[human_score_col].notna()]
    
    # Extract features and labels
    if judge_scores_col in data.columns:
        # Judge scores are in a single column as lists
        X = np.array(data[judge_scores_col].tolist())
    elif 'scores' in data.columns:
        # Alternative naming
        X = np.array(data['scores'].tolist())
    else:
        # Look for individual score columns
        score_cols = [col for col in data.columns if col.startswith('score_')]
        if score_cols:
            X = data[score_cols].values
        else:
            raise ValueError(f"Could not find judge scores in data. Available columns: {data.columns.tolist()}")
    
    y = data[human_score_col].values
    
    logger.info(f"Loaded {len(data)} samples with {X.shape[1]} features")
    
    return data, X, y


def main():
    """Main entry point for aggregator training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train aggregation models")
    parser.add_argument('--input', required=True, help='Path to input data file')
    parser.add_argument('--model-type', choices=['gam', 'mlp', 'both'], default='both',
                        help='Type of model to train')
    parser.add_argument('--output-dir', default='models/', help='Directory to save models')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (fraction)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # GAM parameters
    parser.add_argument('--gam-splines', type=int, default=10,
                        help='Number of splines for GAM')
    parser.add_argument('--gam-lambda', type=float, default=0.6,
                        help='Lambda regularization for GAM')
    
    # MLP parameters
    parser.add_argument('--mlp-hidden', type=int, default=64,
                        help='Hidden dimension for MLP')
    parser.add_argument('--mlp-epochs', type=int, default=100,
                        help='Number of training epochs for MLP')
    parser.add_argument('--mlp-lr', type=float, default=0.001,
                        help='Learning rate for MLP')
    parser.add_argument('--mlp-batch', type=int, default=32,
                        help='Batch size for MLP')
    
    # Visualization
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--plot-dir', default='plots/',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.plot:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data, X, y = load_and_prepare_data(Path(args.input))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train GAM
    if args.model_type in ['gam', 'both']:
        if HAS_GAM:
            logger.info("Training GAM model...")
            gam = GAMAggregator(n_splines=args.gam_splines, lam=args.gam_lambda)
            gam.fit(X_train, y_train)
            
            # Evaluate
            train_pred = gam.predict(X_train)
            test_pred = gam.predict(X_test)
            
            train_metrics = compute_metrics(y_train, train_pred)
            test_metrics = compute_metrics(y_test, test_pred)
            
            logger.info("GAM Results:")
            logger.info(f"  Train - MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
            logger.info(f"  Test  - MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")
            
            # Save model
            gam_path = output_dir / 'gam_model.pkl'
            with open(gam_path, 'wb') as f:
                pickle.dump(gam, f)
            logger.info(f"GAM model saved to {gam_path}")
            
            # Feature importance
            importance = gam.get_feature_importance()
            logger.info("\nFeature Importance (GAM):")
            for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {feature}: {score:.3f}")
            
            # Visualization
            if args.plot:
                plot_partial_dependence(
                    gam,
                    features=[3, 6, 8, 9],
                    title='Partial Dependence - Non-Safety Features',
                    n_cols=2,
                    save_path=plot_dir / 'gam_partial_dependence_nonsafety.png'
                )
                plot_partial_dependence(
                    gam,
                    features=[0, 1, 5],
                    title='Partial Dependence - Safety Features',
                    n_cols=3,
                    save_path=plot_dir / 'gam_partial_dependence_safety.png'
                )
        else:
            logger.warning("PyGAM not installed. Skipping GAM training.")
    
    # Train MLP
    if args.model_type in ['mlp', 'both']:
        logger.info("Training MLP model...")
        mlp_trainer = MLPTrainer(
            hidden_dim=args.mlp_hidden,
            learning_rate=args.mlp_lr,
            batch_size=args.mlp_batch,
            n_epochs=args.mlp_epochs
        )
        
        train_losses, val_losses = mlp_trainer.fit(X_train, y_train, X_test, y_test)
        
        # Evaluate
        train_pred = mlp_trainer.predict(X_train)
        test_pred = mlp_trainer.predict(X_test)
        
        train_metrics = compute_metrics(y_train, train_pred)
        test_metrics = compute_metrics(y_test, test_pred)
        
        logger.info("MLP Results:")
        logger.info(f"  Train - MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
        logger.info(f"  Test  - MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")
        
        # Save model
        mlp_path = output_dir / 'mlp_model.pt'
        mlp_trainer.save_model(mlp_path)
        
        # Plot training curves
        if args.plot and val_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.title('MLP Training Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            curve_path = plot_dir / 'mlp_training_curves.png'
            plt.savefig(curve_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training curves saved to {curve_path}")
            plt.show()
    
    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()