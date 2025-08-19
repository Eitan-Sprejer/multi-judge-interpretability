#!/usr/bin/env python3
"""
Bias Robustness Experiment

Test how different systematic biases affect Judge Mean vs GAM performance.
Uses strength factors to control bias intensity and measure degradation curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import PyGAM for proper GAM implementation
try:
    from pygam import LinearGAM, s
    HAS_GAM = True
except ImportError:
    HAS_GAM = False
    print("PyGAM not installed. GAM training will not be available.")

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
        spline_terms = [s(i, n_splines=self.n_splines, lam=self.lam) for i in range(X.shape[1])]
        if len(spline_terms) > 1:
            terms = spline_terms[0]
            for term in spline_terms[1:]:
                terms += term
        else:
            terms = spline_terms[0]
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

class BiasTransformer:
    """Applies systematic biases to judge scores with parameterizable strength."""
    
    @staticmethod
    def bottom_heavy(scores, strength):
        """
        Bottom-heavy bias: Compresses high scores, expands low scores.
        Strength 0 = no change, strength 1 = extreme compression of high scores.
        """
        if strength == 0:
            return scores
        
        # Use power transformation: scores^(1 + strength*factor)
        # Higher strength makes high scores get compressed more
        power = 1 + strength * 3  # Strength 1 → power of 4
        transformed = (scores / 4.0) ** power * 4.0
        return np.clip(transformed, 0, 4)
    
    @staticmethod
    def top_heavy(scores, strength):
        """
        Top-heavy bias: Compresses low scores, expands high scores.
        Strength 0 = no change, strength 1 = extreme compression of low scores.
        """
        if strength == 0:
            return scores
        
        # Invert, apply bottom-heavy, invert back
        inverted = 4.0 - scores
        transformed_inverted = BiasTransformer.bottom_heavy(inverted, strength)
        return 4.0 - transformed_inverted
    
    @staticmethod
    def middle_heavy(scores, strength):
        """
        Middle-heavy bias: Compresses extreme scores toward center (2.0).
        Strength 0 = no change, strength 1 = everything pulled to center.
        """
        if strength == 0:
            return scores
        
        center = 2.0
        deviation = scores - center
        # Compress deviations by strength factor
        compressed_deviation = deviation * (1 - strength * 0.8)
        return center + compressed_deviation
    
    @staticmethod
    def systematic_shift(scores, strength):
        """
        Systematic shift: Linear translation of entire distribution.
        Strength 0 = no shift, strength 1 = shift by +1.0, strength -1 = shift by -1.0.
        """
        if strength == 0:
            return scores
        
        shift_amount = strength * 1.0  # +/- 1 point max shift
        return np.clip(scores + shift_amount, 0, 4)
    
    @staticmethod
    def get_available_biases():
        """Return list of available bias transformations."""
        return ['bottom_heavy', 'top_heavy', 'middle_heavy', 'systematic_shift_positive', 'systematic_shift_negative']
    
    @staticmethod
    def apply_bias(scores, bias_type, strength):
        """Apply specified bias with given strength."""
        if bias_type == 'bottom_heavy':
            return BiasTransformer.bottom_heavy(scores, strength)
        elif bias_type == 'top_heavy':
            return BiasTransformer.top_heavy(scores, strength)
        elif bias_type == 'middle_heavy':
            return BiasTransformer.middle_heavy(scores, strength)
        elif bias_type == 'systematic_shift_positive':
            return BiasTransformer.systematic_shift(scores, strength)
        elif bias_type == 'systematic_shift_negative':
            return BiasTransformer.systematic_shift(scores, -strength)
        else:
            raise ValueError(f"Unknown bias type: {bias_type}")

def load_experiment_data():
    """Load original judge scores and human ground truth."""
    # Load judge scores
    results_dir = Path(__file__).parent / 'results_full_20250818_215910'
    scores_path = results_dir / 'restructured_scores_fixed.pkl'
    
    print(f"Loading judge scores from {scores_path}")
    with open(scores_path, 'rb') as f:
        scores_df = pickle.load(f)
    
    # Get original judge scores only
    original_cols = [col for col in scores_df.columns if col.endswith('_original')]
    judge_scores = scores_df[original_cols].values
    judge_names = [col.replace('_original', '') for col in original_cols]
    
    print(f"Loaded {judge_scores.shape[0]} samples x {judge_scores.shape[1]} judges")
    print(f"Judge names: {judge_names}")
    
    # Load human ground truth
    project_root = Path(__file__).parent.parent.parent
    possible_gt_paths = [
        project_root / 'dataset' / 'data_with_judge_scores.pkl',
        project_root / 'results' / 'full_experiments' / 'baseline_ultrafeedback_2000samples_20250816_213023' / 'data' / 'data_with_judge_scores.pkl'
    ]
    
    human_scores = None
    for gt_path in possible_gt_paths:
        if gt_path.exists():
            print(f"Loading ground truth from {gt_path}")
            with open(gt_path, 'rb') as f:
                gt_data = pickle.load(f)
            
            # Extract human scores using balanced persona sampling
            human_scores = []
            np.random.seed(42)
            
            sample_data = gt_data['human_feedback'].values[0]
            if isinstance(sample_data, dict) and 'personas' in sample_data:
                available_personas = list(sample_data['personas'].keys())
                samples_per_persona = 1000 // len(available_personas)
                remaining_samples = 1000 % len(available_personas)
                
                persona_assignment = []
                for persona in available_personas:
                    persona_assignment.extend([persona] * samples_per_persona)
                for _ in range(remaining_samples):
                    persona_assignment.append(np.random.choice(available_personas))
                np.random.shuffle(persona_assignment)
                
                for idx, score_data in enumerate(gt_data['human_feedback'].values[:1000]):
                    assigned_persona = persona_assignment[idx]
                    if isinstance(score_data, dict) and 'personas' in score_data:
                        personas = score_data['personas']
                        if (assigned_persona in personas and 
                            isinstance(personas[assigned_persona], dict) and 
                            'score' in personas[assigned_persona]):
                            score = float(personas[assigned_persona]['score'])
                            human_scores.append(score)
                        else:
                            human_scores.append(5.0)  # Fallback
                    else:
                        human_scores.append(5.0)
            break
    
    if human_scores is None:
        raise FileNotFoundError("Could not find human ground truth data")
    
    human_scores = np.array(human_scores)
    
    # Ensure same length
    min_length = min(len(judge_scores), len(human_scores))
    judge_scores = judge_scores[:min_length]
    human_scores = human_scores[:min_length]
    
    print(f"Final dataset: {len(judge_scores)} samples")
    print(f"Human scores range: [{human_scores.min():.1f}, {human_scores.max():.1f}], mean: {human_scores.mean():.2f}")
    
    return judge_scores, human_scores, judge_names

def train_models(X_train, X_test, y_train, y_test):
    """Train all aggregator models and judge mean baseline."""
    results = {}
    
    # 1. GAM Model (Proper Generalized Additive Model with splines)
    try:
        gam_model = GAMAggregator(n_splines=10, lam=0.6)
        gam_model.fit(X_train, y_train)
        gam_predictions = gam_model.predict(X_test)
        
        results['gam_r2'] = r2_score(y_test, gam_predictions)
        results['gam_mae'] = mean_absolute_error(y_test, gam_predictions)
        
    except Exception as e:
        print(f"GAM training failed: {e}")
        results['gam_r2'] = np.nan
        results['gam_mae'] = np.nan
    
    # 2. MLP Model
    try:
        mlp_model = MLPRegressor(
            hidden_layer_sizes=(64,),
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10
        )
        mlp_model.fit(X_train, y_train)
        mlp_predictions = mlp_model.predict(X_test)
        
        results['mlp_r2'] = r2_score(y_test, mlp_predictions)
        results['mlp_mae'] = mean_absolute_error(y_test, mlp_predictions)
        
    except Exception as e:
        print(f"MLP training failed: {e}")
        results['mlp_r2'] = np.nan
        results['mlp_mae'] = np.nan
    
    # 3. Linear Regression Aggregator
    try:
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_predictions = lr_model.predict(X_test)
        
        results['lr_r2'] = r2_score(y_test, lr_predictions)
        results['lr_mae'] = mean_absolute_error(y_test, lr_predictions)
        
    except Exception as e:
        print(f"Linear Regression training failed: {e}")
        results['lr_r2'] = np.nan
        results['lr_mae'] = np.nan
    
    # 4. Judge Mean Baseline (Simple interval scaling: 0-4 → 0-10)
    judge_mean_raw = np.mean(X_test, axis=1)
    judge_mean_predictions = (judge_mean_raw / 4.0) * 10.0  # Simple interval scaling
    
    results['judge_mean_r2'] = r2_score(y_test, judge_mean_predictions)
    results['judge_mean_mae'] = mean_absolute_error(y_test, judge_mean_predictions)
    
    return results

def run_bias_experiment(judge_scores, human_scores, bias_type, strength_range):
    """Run bias experiment across range of strengths."""
    print(f"\n🧪 Testing {bias_type} bias...")
    
    results = []
    
    for strength in strength_range:
        print(f"  Strength {strength:.2f}...", end=' ')
        
        # Apply bias transformation to ALL judges
        biased_scores = np.zeros_like(judge_scores)
        for j in range(judge_scores.shape[1]):
            biased_scores[:, j] = BiasTransformer.apply_bias(judge_scores[:, j], bias_type, strength)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            biased_scores, human_scores, test_size=0.2, random_state=42
        )
        
        # Train models
        model_results = train_models(X_train, X_test, y_train, y_test)
        
        # Store results
        result = {
            'bias_type': bias_type,
            'strength': strength,
            **model_results
        }
        results.append(result)
        
        print(f"GAM: {model_results['gam_r2']:.3f}, MLP: {model_results['mlp_r2']:.3f}, LR: {model_results['lr_r2']:.3f}, Judge Mean: {model_results['judge_mean_r2']:.3f}")
    
    return results

def create_robustness_plots(all_results, output_dir):
    """Create plots showing performance degradation vs bias strength."""
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Create subplot for each bias type
    bias_types = df['bias_type'].unique()
    n_bias = len(bias_types)
    
    fig, axes = plt.subplots(1, n_bias, figsize=(5*n_bias, 6))
    if n_bias == 1:
        axes = [axes]
    
    colors = ['#2E86AB', '#A23B72', '#8B4513', '#F18F01']  # Blue, Purple, Brown, Orange
    
    for i, bias_type in enumerate(bias_types):
        bias_data = df[df['bias_type'] == bias_type]
        
        # R² vs Strength plot
        ax = axes[i]
        
        ax.plot(bias_data['strength'], bias_data['gam_r2'], 'o-', 
                color=colors[0], linewidth=2, markersize=6, label='GAM (Splines)')
        ax.plot(bias_data['strength'], bias_data['mlp_r2'], 'o-', 
                color=colors[1], linewidth=2, markersize=6, label='MLP')
        ax.plot(bias_data['strength'], bias_data['lr_r2'], 'o-', 
                color=colors[2], linewidth=2, markersize=6, label='Linear Regression')
        ax.plot(bias_data['strength'], bias_data['judge_mean_r2'], 'o-', 
                color=colors[3], linewidth=2, markersize=6, label='Judge Mean')
        
        ax.set_xlabel('Bias Strength')
        ax.set_ylabel('R² Score')
        ax.set_title(f'{bias_type.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-0.8, 1.0)
        
        # Add performance annotation
        final_gam = bias_data['gam_r2'].iloc[-1]
        final_judge = bias_data['judge_mean_r2'].iloc[-1]
        
        if final_judge > 0:
            advantage = final_gam / final_judge
            ax.text(0.02, 0.98, f'Final GAM: {final_gam:.3f}\nFinal Judge: {final_judge:.3f}\nAdvantage: {advantage:.1f}x', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        else:
            ax.text(0.02, 0.98, f'Final GAM: {final_gam:.3f}\nFinal Judge: {final_judge:.3f}\nJudge Mean Failed', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    
    plt.suptitle('🎯 Bias Robustness: Performance vs Bias Strength', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'bias_robustness_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Robustness plots saved to: {plot_path}")
    
    return plot_path

def analyze_results(all_results):
    """Analyze and summarize the bias robustness results."""
    df = pd.DataFrame(all_results)
    
    print(f"\n{'='*80}")
    print("🎯 BIAS ROBUSTNESS ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    for bias_type in df['bias_type'].unique():
        bias_data = df[df['bias_type'] == bias_type]
        
        print(f"\n📊 {bias_type.upper().replace('_', ' ')}:")
        
        # Performance at maximum bias
        max_strength_data = bias_data[bias_data['strength'] == bias_data['strength'].max()]
        
        if len(max_strength_data) > 0:
            row = max_strength_data.iloc[0]
            print(f"  At max strength ({row['strength']:.1f}):")
            print(f"    GAM R²: {row['gam_r2']:.3f}")
            print(f"    MLP R²: {row['mlp_r2']:.3f}")
            print(f"    Linear Regression R²: {row['lr_r2']:.3f}")
            print(f"    Judge Mean R²: {row['judge_mean_r2']:.3f}")
        
        # Calculate degradation for all models
        baseline = bias_data[bias_data['strength'] == 0]
        max_bias = bias_data[bias_data['strength'] == bias_data['strength'].max()]
        
        if len(baseline) > 0 and len(max_bias) > 0:
            gam_degradation = baseline['gam_r2'].iloc[0] - max_bias['gam_r2'].iloc[0]
            mlp_degradation = baseline['mlp_r2'].iloc[0] - max_bias['mlp_r2'].iloc[0]
            lr_degradation = baseline['lr_r2'].iloc[0] - max_bias['lr_r2'].iloc[0]
            judge_degradation = baseline['judge_mean_r2'].iloc[0] - max_bias['judge_mean_r2'].iloc[0]
            
            print(f"    Performance degradation:")
            print(f"      GAM: -{gam_degradation:.3f}")
            print(f"      MLP: -{mlp_degradation:.3f}")
            print(f"      Linear Regression: -{lr_degradation:.3f}")
            print(f"      Judge Mean: -{judge_degradation:.3f}")
            
            # Find most robust learned aggregator
            learned_degradations = [gam_degradation, mlp_degradation, lr_degradation]
            min_learned_degradation = min(learned_degradations)
            
            if judge_degradation > min_learned_degradation * 1.5:
                print(f"    ✅ Learned aggregators are MORE ROBUST than Judge Mean")
            elif min_learned_degradation > judge_degradation * 1.5:
                print(f"    ❌ Judge Mean is MORE ROBUST than learned aggregators")
            else:
                print(f"    ≈ Similar robustness between learned and baseline methods")

def main():
    """Main experimental function."""
    print("🚀 Starting Bias Robustness Experiment...")
    
    # Create output directory
    output_dir = Path(__file__).parent / 'results_full_20250818_215910' / 'bias_robustness'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    judge_scores, human_scores, judge_names = load_experiment_data()
    
    # Define bias types and strength ranges to test
    bias_experiments = {
        'bottom_heavy': np.linspace(0, 1.0, 6),      # 0, 0.2, 0.4, 0.6, 0.8, 1.0
        'top_heavy': np.linspace(0, 1.0, 6),
        'middle_heavy': np.linspace(0, 1.0, 6),
        'systematic_shift_positive': np.linspace(0, 1.0, 6),
        'systematic_shift_negative': np.linspace(0, 1.0, 6)
    }
    
    # Run experiments
    all_results = []
    
    for bias_type, strength_range in bias_experiments.items():
        bias_results = run_bias_experiment(judge_scores, human_scores, bias_type, strength_range)
        all_results.extend(bias_results)
    
    # Create visualizations
    create_robustness_plots(all_results, output_dir)
    
    # Analyze results
    analyze_results(all_results)
    
    # Save raw results
    results_path = output_dir / 'bias_robustness_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n✅ Bias robustness experiment complete!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📈 Check bias_robustness_analysis.png for key plots")

if __name__ == "__main__":
    main()