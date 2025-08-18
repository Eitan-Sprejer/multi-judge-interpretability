#!/usr/bin/env python3
"""
GAM Hyperparameter Tuning for Multi-Judge Interpretability

Uses existing experiment data to test different GAM configurations
and find optimal hyperparameters for improving R² scores with interpretable models.
"""

import json
import pickle
import random
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import itertools
from scipy import stats

# Import GAM dependencies
try:
    from pygam import LinearGAM, s, f, te
    HAS_GAM = True
except ImportError:
    HAS_GAM = False
    print("❌ PyGAM not installed. Install with: pip install pygam")
    exit(1)

# Import project modules
from pipeline.core.aggregator_training import GAMAggregator, compute_metrics, FEATURE_LABELS
from pipeline.core.persona_simulation import PERSONAS


class GAMHyperparameterTuner:
    """
    Hyperparameter tuning specifically for GAM (Generalized Additive Models) aggregation models.
    """
    
    def __init__(
        self,
        experiment_data_path: str,
        output_dir: str = "gam_hyperparameter_tuning_results",
        test_size: float = 0.2,
        random_seed: int = 42
    ):
        self.experiment_data_path = experiment_data_path
        self.output_dir = Path(output_dir) if not str(output_dir).startswith("results/") else Path(output_dir)
        # If default output dir, use organized structure
        if output_dir == "gam_hyperparameter_tuning_results":
            self.output_dir = Path("results/gam_hyperparameter_search")
        self.test_size = test_size
        self.random_seed = random_seed
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"gam_tuning_run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 GAM hyperparameter tuning output: {self.run_dir}")
    
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from completed experiment and prepare for GAM training."""
        # Try different possible data file locations
        possible_paths = [
            Path(self.experiment_data_path) / "data_with_judge_scores.pkl",
            Path(self.experiment_data_path) / "data" / "data_with_judge_scores.pkl",
            Path(self.experiment_data_path)
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists() and path.is_file():
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(f"Could not find judge scores data in: {possible_paths}")
        
        print(f"📂 Loading experiment data from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        print(f"✅ Loaded {len(data)} samples with judge scores and persona feedback")
        
        # Prepare training data with uniform persona sampling
        X_list = []
        y_list = []
        
        # Uniform persona sampling for consistency
        available_personas = list(PERSONAS.keys())
        samples_per_persona = len(data) // len(available_personas)
        remaining_samples = len(data) % len(available_personas)
        
        persona_assignment = []
        for persona in available_personas:
            persona_assignment.extend([persona] * samples_per_persona)
        for _ in range(remaining_samples):
            persona_assignment.append(random.choice(available_personas))
        random.shuffle(persona_assignment)
        
        # Extract features and targets
        for idx, (row, assigned_persona) in enumerate(zip(data.iterrows(), persona_assignment)):
            row = row[1]
            
            if ('human_feedback' not in row or 'personas' not in row['human_feedback'] or
                'judge_scores' not in row or not isinstance(row['judge_scores'], list)):
                continue
            
            personas_feedback = row['human_feedback']['personas']
            if assigned_persona not in personas_feedback or 'score' not in personas_feedback[assigned_persona]:
                continue
            
            selected_score = personas_feedback[assigned_persona]['score']
            judge_scores = row['judge_scores']
            
            if selected_score is None or len(judge_scores) != 10:
                continue
            
            X_list.append(judge_scores)
            y_list.append(selected_score)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"✅ Prepared {len(X)} training samples")
        return X, y
    
    def define_gam_hyperparameter_grid(self) -> Dict[str, List]:
        """
        Define comprehensive GAM hyperparameter search grid.
        
        Optimized based on analysis showing:
        - Low lambda (0.1) + high splines (25) perform poorly
        - Low effective DOF + low splines give better results
        """
        return {
            # Number of splines per feature (controls complexity)
            # Reduced range based on effective DOF analysis - lower splines perform better
            'n_splines': [5, 8, 10, 12, 15],
            
            # Lambda regularization (controls smoothness) 
            # Increased range based on heatmap showing poor performance at lambda=0.1
            'lam': [2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            
            # Feature interaction terms (subset of features to interact)
            # Each tuple represents indices of judges that should interact
            # Testing 0-3 interaction combinations based on domain knowledge
            'interaction_features': [
                [],  # 0 interactions: Independent effects only
                
                # 1 interaction pair: Test core safety relationships
                [(0, 1)],  # Truthfulness & Harmlessness (safety core)
                [(0, 2)],  # Truthfulness & Helpfulness (quality core)  
                [(1, 2)],  # Harmlessness & Helpfulness (user benefit)
                [(6, 7)],  # Clarity & Conciseness (communication)
                [(8, 9)],  # Logic & Creativity (reasoning style)
                
                # 2 interaction pairs: Test domain clusters
                [(0, 1), (6, 7)],  # Safety + Communication clarity
                [(0, 2), (8, 9)],  # Core quality + Reasoning depth
                [(1, 2), (6, 7)],  # User benefit + Clear communication
                
                # 3 interaction pairs: Complex multi-domain relationships
                [(0, 1), (6, 7), (8, 9)],  # Safety + Communication + Reasoning
            ],
            
            # Max iterations for convergence
            'max_iter': [100, 200, 300],
            
            # Tolerance for convergence (balance between precision and training time)
            # 1e-4 = loose (faster), 1e-6 = tight (slower but more precise)
            'tol': [1e-4, 1e-5]
        }
    
    def create_gam_model(self, config: Dict) -> LinearGAM:
        """Create GAM model with specified configuration."""
        # Build terms for each feature
        terms = []
        
        # Add individual spline terms for each judge
        for i in range(10):  # 10 judges
            terms.append(s(i, n_splines=config['n_splines'], lam=config['lam']))
        
        # Add interaction terms if specified
        for interaction in config['interaction_features']:
            if len(interaction) == 2:
                # Pairwise interaction
                terms.append(te(interaction[0], interaction[1], 
                               n_splines=max(5, config['n_splines']//2), 
                               lam=config['lam']))
            elif len(interaction) == 3:
                # Three-way interaction (more complex)
                terms.append(te(interaction[0], interaction[1], interaction[2],
                               n_splines=max(5, config['n_splines']//3),
                               lam=config['lam']))
        
        # Combine all terms
        if len(terms) == 0:
            # No terms - create a simple linear model
            gam_terms = s(0, n_splines=config['n_splines'], lam=config['lam'])
        elif len(terms) == 1:
            gam_terms = terms[0]
        else:
            # Sum all terms together
            gam_terms = terms[0]
            for term in terms[1:]:
                gam_terms = gam_terms + term
        
        # Create GAM model
        gam = LinearGAM(
            gam_terms,
            fit_intercept=True,
            max_iter=config['max_iter'],
            tol=config['tol']
        )
        
        return gam
    
    def evaluate_gam_config(
        self, 
        config: Dict, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """Evaluate a single GAM configuration."""
        try:
            # Normalize data if requested
            if normalize:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train.copy()
                X_test_scaled = X_test.copy()
            
            # Create and fit GAM model
            gam = self.create_gam_model(config)
            gam.fit(X_train_scaled, y_train)
            
            # Make predictions
            train_pred = gam.predict(X_train_scaled)
            test_pred = gam.predict(X_test_scaled)
            
            # Compute metrics
            train_metrics = compute_metrics(y_train, train_pred)
            test_metrics = compute_metrics(y_test, test_pred)
            
            # GAM-specific metrics
            try:
                # Calculate deviance manually if method not available
                test_loglik = gam.loglikelihood(X_test_scaled, y_test)
                null_loglik = np.sum(stats.norm.logpdf(y_test, loc=np.mean(y_test), scale=np.std(y_test)))
                deviance = -2 * (test_loglik - null_loglik)
            except:
                deviance = np.nan
            
            gam_metrics = {
                'aic': gam.statistics_['AIC'],
                'deviance': deviance,
                'edof': gam.statistics_['edof'],  # Effective degrees of freedom
                'gcv': gam.statistics_['GCV'],   # Generalized cross-validation
                'n_terms': len(gam.terms)
            }
            
            # Feature importance (using p-values)
            try:
                p_values = gam.statistics_['p_values']
                feature_importance = {}
                for i, label in enumerate(FEATURE_LABELS):
                    if i < len(p_values):
                        # Convert p-value to importance (lower p-value = higher importance)
                        feature_importance[label] = max(0, 1.0 - p_values[i])
                    else:
                        feature_importance[label] = 0.0
            except:
                feature_importance = {}
            
            result = {
                'config': config,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'gam_metrics': gam_metrics,
                'feature_importance': feature_importance,
                'model': gam,
                'scaler': scaler if normalize else None,
                'normalize': normalize,
                'success': True
            }
            
            return result
            
        except Exception as e:
            return {
                'config': config,
                'error': str(e),
                'success': False
            }
    
    def random_search(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_trials: int = 50,
        normalize: bool = True
    ) -> List[Dict]:
        """Perform random hyperparameter search for GAM models."""
        print(f"🔍 Starting GAM random search with {n_trials} trials")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed
        )
        
        hyperparams = self.define_gam_hyperparameter_grid()
        results = []
        successful_trials = 0
        
        for trial in range(n_trials):
            # Sample random hyperparameters
            config = {
                'n_splines': random.choice(hyperparams['n_splines']),
                'lam': random.choice(hyperparams['lam']),
                'interaction_features': random.choice(hyperparams['interaction_features']),
                'max_iter': random.choice(hyperparams['max_iter']),
                'tol': random.choice(hyperparams['tol'])
            }
            
            print(f"Trial {trial + 1}/{n_trials}: splines={config['n_splines']}, "
                  f"λ={config['lam']:.2f}, interactions={len(config['interaction_features'])}")
            
            # Evaluate configuration
            result = self.evaluate_gam_config(config, X_train, y_train, X_test, y_test, normalize)
            
            if result['success']:
                results.append(result)
                successful_trials += 1
                print(f"  ✅ R² = {result['test_metrics']['r2']:.4f}, "
                      f"AIC = {result['gam_metrics']['aic']:.2f}, "
                      f"GCV = {result['gam_metrics']['gcv']:.4f}")
            else:
                print(f"  ❌ Trial failed: {result['error']}")
        
        print(f"📊 Completed {successful_trials}/{n_trials} successful trials")
        
        # Sort by test R²
        results.sort(key=lambda x: x['test_metrics']['r2'], reverse=True)
        
        return results
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze GAM hyperparameter tuning results."""
        if not results:
            return {}
        
        # Best overall result
        best_result = results[0]
        
        # Extract all test R² scores
        r2_scores = [r['test_metrics']['r2'] for r in results if r['success']]
        aic_scores = [r['gam_metrics']['aic'] for r in results if r['success']]
        gcv_scores = [r['gam_metrics']['gcv'] for r in results if r['success']]
        
        analysis = {
            'best_config': best_result['config'],
            'best_r2': best_result['test_metrics']['r2'],
            'best_mae': best_result['test_metrics']['mae'],
            'best_aic': best_result['gam_metrics']['aic'],
            'best_gcv': best_result['gam_metrics']['gcv'],
            'best_edof': best_result['gam_metrics']['edof'],
            'mean_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores),
            'mean_aic': np.mean(aic_scores),
            'mean_gcv': np.mean(gcv_scores),
            'top_5_configs': [r['config'] for r in results[:5]],
            'top_5_r2': [r['test_metrics']['r2'] for r in results[:5]],
            'top_5_aic': [r['gam_metrics']['aic'] for r in results[:5]],
            'feature_importance_best': best_result['feature_importance'],
            'successful_trials': len(results),
            'model_complexity_stats': {
                'mean_n_terms': np.mean([r['gam_metrics']['n_terms'] for r in results]),
                'mean_edof': np.mean([r['gam_metrics']['edof'] for r in results]),
                'complexity_vs_performance': [
                    (r['gam_metrics']['edof'], r['test_metrics']['r2']) for r in results
                ]
            }
        }
        
        return analysis
    
    def create_gam_visualizations(self, results: List[Dict], analysis: Dict):
        """Create comprehensive GAM-specific visualization plots."""
        if not results:
            return
        
        # Main analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. R² distribution
        r2_scores = [r['test_metrics']['r2'] for r in results]
        axes[0, 0].hist(r2_scores, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        axes[0, 0].axvline(analysis['best_r2'], color='green', linestyle='--', label='Best R²')
        axes[0, 0].axvline(analysis['mean_r2'], color='red', linestyle='--', label='Mean R²')
        axes[0, 0].set_xlabel('Test R² Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of GAM R² Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Lambda vs R²
        lambda_vs_r2 = [(r['config']['lam'], r['test_metrics']['r2']) for r in results]
        lambdas, r2s = zip(*lambda_vs_r2)
        axes[0, 1].scatter(lambdas, r2s, alpha=0.6, color='orange')
        axes[0, 1].set_xlabel('Lambda (Regularization)')
        axes[0, 1].set_ylabel('Test R² Score')
        axes[0, 1].set_title('Regularization vs Performance')
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. N_splines vs R²
        splines_vs_r2 = [(r['config']['n_splines'], r['test_metrics']['r2']) for r in results]
        splines, r2s = zip(*splines_vs_r2)
        axes[0, 2].scatter(splines, r2s, alpha=0.6, color='green')
        axes[0, 2].set_xlabel('Number of Splines')
        axes[0, 2].set_ylabel('Test R² Score')
        axes[0, 2].set_title('Model Complexity vs Performance')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. AIC vs R² (model selection trade-off)
        aic_vs_r2 = [(r['gam_metrics']['aic'], r['test_metrics']['r2']) for r in results]
        aics, r2s = zip(*aic_vs_r2)
        axes[1, 0].scatter(aics, r2s, alpha=0.6, color='purple')
        axes[1, 0].set_xlabel('AIC (Lower is Better)')
        axes[1, 0].set_ylabel('Test R² Score')
        axes[1, 0].set_title('AIC vs R² Trade-off')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Effective degrees of freedom vs R²
        edof_vs_r2 = [(r['gam_metrics']['edof'], r['test_metrics']['r2']) for r in results]
        edofs, r2s = zip(*edof_vs_r2)
        axes[1, 1].scatter(edofs, r2s, alpha=0.6, color='red')
        axes[1, 1].set_xlabel('Effective Degrees of Freedom')
        axes[1, 1].set_ylabel('Test R² Score')
        axes[1, 1].set_title('Model Complexity vs Performance')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Top 10 configurations
        top_10 = results[:10]
        config_names = [f"Config {i+1}" for i in range(len(top_10))]
        top_r2s = [r['test_metrics']['r2'] for r in top_10]
        
        bars = axes[1, 2].bar(config_names, top_r2s, color='lightcoral')
        axes[1, 2].set_xlabel('Configuration')
        axes[1, 2].set_ylabel('Test R² Score')
        axes[1, 2].set_title('Top 10 GAM Configurations')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # Color top bars
        for i, bar in enumerate(bars):
            if i < 3:
                bar.set_color('gold')
            elif i < 5:
                bar.set_color('orange')
            else:
                bar.set_color('lightcoral')
        
        plt.tight_layout()
        
        # Save main analysis plot
        analysis_path = self.run_dir / 'gam_hyperparameter_analysis.png'
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create GAM-specific heatmaps
        self.create_gam_heatmaps(results, analysis)
        
        # Create partial dependence plots for best model
        self.create_partial_dependence_plots(results[0])
        
        print(f"📊 GAM analysis plots saved to {analysis_path}")
    
    def create_gam_heatmaps(self, results: List[Dict], analysis: Dict):
        """Create GAM-specific heatmaps for hyperparameter analysis."""
        # Extract data for heatmaps
        data_rows = []
        for result in results:
            config = result['config']
            data_rows.append({
                'n_splines': config['n_splines'],
                'lam': config['lam'],
                'n_interactions': len(config['interaction_features']),
                'max_iter': config['max_iter'],
                'tol': config['tol'],
                'test_r2': result['test_metrics']['r2'],
                'aic': result['gam_metrics']['aic'],
                'edof': result['gam_metrics']['edof'],
                'gcv': result['gam_metrics']['gcv']
            })
        
        df = pd.DataFrame(data_rows)
        
        # Create GAM heatmap visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Heatmap 1: Best R² by N_splines vs Lambda
        pivot_r2 = df.pivot_table(
            values='test_r2', 
            index='lam', 
            columns='n_splines', 
            aggfunc='max'
        )
        
        sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Best Test R²'}, ax=ax1)
        ax1.set_title('Best Test R² by Configuration\n(N_splines vs Lambda)')
        ax1.set_xlabel('Number of Splines')
        ax1.set_ylabel('Lambda (Regularization)')
        
        # Heatmap 2: AIC by N_splines vs Lambda (lower is better)
        pivot_aic = df.pivot_table(
            values='aic', 
            index='lam', 
            columns='n_splines', 
            aggfunc='min'  # Lower AIC is better
        )
        
        sns.heatmap(pivot_aic, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                   cbar_kws={'label': 'Best AIC (Lower is Better)'}, ax=ax2)
        ax2.set_title('Best AIC by Configuration\n(N_splines vs Lambda)')
        ax2.set_xlabel('Number of Splines')
        ax2.set_ylabel('Lambda (Regularization)')
        
        # Heatmap 3: Model Complexity (EDOF) vs Performance
        # Bin EDOF for better visualization
        df['edof_binned'] = pd.cut(df['edof'], bins=5, labels=['Low', 'Med-Low', 'Medium', 'Med-High', 'High'])
        pivot_complexity = df.pivot_table(
            values='test_r2',
            index='edof_binned',
            columns='n_splines',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_complexity, annot=True, fmt='.3f', cmap='RdYlGn',
                   cbar_kws={'label': 'Mean Test R²'}, ax=ax3)
        ax3.set_title('Performance vs Model Complexity\n(EDOF vs N_splines)')
        ax3.set_xlabel('Number of Splines')
        ax3.set_ylabel('Effective Degrees of Freedom')
        
        # Heatmap 4: Interaction effects vs tolerance
        pivot_interactions = df.pivot_table(
            values='test_r2',
            index='n_interactions',
            columns='tol',
            aggfunc='max'
        )
        
        sns.heatmap(pivot_interactions, annot=True, fmt='.3f', cmap='RdYlGn',
                   cbar_kws={'label': 'Best Test R²'}, ax=ax4)
        ax4.set_title('Best R² by Feature Interactions\n(N_interactions vs Tolerance)')
        ax4.set_xlabel('Tolerance')
        ax4.set_ylabel('Number of Interaction Terms')
        
        plt.tight_layout()
        
        # Add overall title with best config info
        best_config = results[0]['config']
        best_r2 = analysis['best_r2']
        fig.suptitle(f'GAM Hyperparameter Analysis - Best: R²={best_r2:.3f} '
                    f'(Splines={best_config["n_splines"]}, λ={best_config["lam"]:.2f}, '
                    f'AIC={analysis["best_aic"]:.1f})', 
                    fontsize=14, y=0.98)
        
        # Save heatmap
        heatmap_path = self.run_dir / "gam_hyperparameter_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🔥 GAM heatmap saved to {heatmap_path}")
    
    def create_partial_dependence_plots(self, best_result: Dict):
        """Create partial dependence plots for the best GAM model."""
        if not best_result['success']:
            return
        
        gam_model = best_result['model']
        
        # Create partial dependence plots for all features
        n_features = 10
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes_flat = axes.flatten()
        
        for i in range(n_features):
            ax = axes_flat[i]
            
            try:
                # Generate grid for partial dependence
                XX = gam_model.generate_X_grid(term=i, meshgrid=False)
                x_values = XX[:, i]
                y_values = gam_model.partial_dependence(term=i, X=XX)
                
                # Plot partial dependence
                ax.plot(x_values, y_values, 'b-', linewidth=2, label='Partial Dependence')
                
                # Add confidence intervals if available
                try:
                    # Generate confidence intervals for partial dependence
                    conf_int = gam_model.confidence_intervals(X=XX, width=0.95)
                    # Extract confidence intervals for this specific term
                    if conf_int.shape[1] >= 2:
                        ax.fill_between(x_values, conf_int[:, 0], conf_int[:, 1], 
                                       alpha=0.3, color='blue', label='95% CI')
                except Exception as e:
                    # If confidence intervals fail, add a note
                    ax.text(0.05, 0.05, 'CI unavailable', transform=ax.transAxes,
                           fontsize=8, alpha=0.7)
                
                # Add trend analysis
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
                
                ax.set_title(f'{FEATURE_LABELS[i]}', fontsize=10)
                ax.set_xlabel('Judge Score')
                ax.set_ylabel('Effect on Prediction')
                ax.grid(True, alpha=0.3)
                
                if i == 0:
                    ax.legend(fontsize=8)
                    
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes,
                       ha='center', va='center')
                ax.set_title(f'{FEATURE_LABELS[i]} (Error)', fontsize=10)
        
        # Hide unused subplots
        for i in range(n_features, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('GAM Partial Dependence Plots - Best Model', fontsize=16, y=1.02)
        
        # Save partial dependence plots
        pdp_path = self.run_dir / "gam_partial_dependence_plots.png"
        plt.savefig(pdp_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Partial dependence plots saved to {pdp_path}")
    
    def save_results(self, results: List[Dict], analysis: Dict):
        """Save all GAM tuning results to files."""
        # Save detailed results (without model objects for JSON compatibility)
        results_for_json = []
        for result in results:
            if result['success']:
                json_result = {
                    'config': result['config'],
                    'train_metrics': result['train_metrics'],
                    'test_metrics': result['test_metrics'],
                    'gam_metrics': result['gam_metrics'],
                    'feature_importance': result['feature_importance'],
                    'normalize': result['normalize']
                }
                results_for_json.append(json_result)
        
        results_path = self.run_dir / 'gam_detailed_results.json'
        with open(results_path, 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        # Save analysis summary
        analysis_path = self.run_dir / 'gam_analysis_summary.json'
        with open(analysis_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_analysis = {}
            for key, value in analysis.items():
                if isinstance(value, dict):
                    serializable_analysis[key] = {k: float(v) if isinstance(v, np.number) else v for k, v in value.items()}
                elif isinstance(value, (np.ndarray, list)):
                    serializable_analysis[key] = [float(x) if isinstance(x, np.number) else x for x in value]
                else:
                    serializable_analysis[key] = float(value) if isinstance(value, np.number) else value
            json.dump(serializable_analysis, f, indent=2)
        
        # Save best model using pickle
        if results and results[0]['success']:
            best_model_path = self.run_dir / 'best_gam_model.pkl'
            with open(best_model_path, 'wb') as f:
                pickle.dump({
                    'model': results[0]['model'],
                    'scaler': results[0]['scaler'],
                    'config': results[0]['config'],
                    'metrics': results[0]['test_metrics'],
                    'feature_importance': results[0]['feature_importance']
                }, f)
            print(f"🏆 Best GAM model saved to {best_model_path}")
        
        print(f"💾 GAM results saved to {self.run_dir}")
    
    def run_tuning(self, n_trials: int = 50, normalize: bool = True) -> Dict:
        """Run complete GAM hyperparameter tuning."""
        print(f"🚀 Starting GAM hyperparameter search with {n_trials} trials")
        
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Run random search
        results = self.random_search(X, y, n_trials=n_trials, normalize=normalize)
        
        if not results:
            print("❌ No successful trials found")
            return {}
        
        # Analyze results
        analysis = self.analyze_results(results)
        
        # Create visualizations
        self.create_gam_visualizations(results, analysis)
        
        # Save results
        self.save_results(results, analysis)
        
        return analysis


def main():
    parser = argparse.ArgumentParser(description="GAM hyperparameter tuning for multi-judge interpretability")
    parser.add_argument('--experiment-path', required=True,
                        help='Path to completed experiment directory or data file')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials for random search')
    parser.add_argument('--output-dir', default='gam_hyperparameter_tuning_results',
                        help='Output directory for results')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize input features (default: True)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Run GAM tuning
    tuner = GAMHyperparameterTuner(
        experiment_data_path=args.experiment_path,
        output_dir=args.output_dir,
        random_seed=args.random_seed
    )
    
    analysis = tuner.run_tuning(
        n_trials=args.n_trials,
        normalize=args.normalize
    )
    
    if analysis:
        # Print summary
        print("\n" + "="*80)
        print("🎯 GAM HYPERPARAMETER TUNING COMPLETE!")
        print("="*80)
        print(f"🏆 Best R² Score: {analysis['best_r2']:.4f}")
        print(f"📊 Best AIC: {analysis['best_aic']:.2f}")
        print(f"📈 Best GCV: {analysis['best_gcv']:.4f}")
        print(f"🧠 Effective DOF: {analysis['best_edof']:.1f}")
        print(f"🔧 Best Configuration:")
        for key, value in analysis['best_config'].items():
            print(f"   {key}: {value}")
        
        print(f"\n📊 Summary Statistics:")
        print(f"   Mean R²: {analysis['mean_r2']:.4f}")
        print(f"   Std R²: {analysis['std_r2']:.4f}")
        print(f"   Successful Trials: {analysis['successful_trials']}")
        print(f"   Top 5 R² scores: {[f'{r:.4f}' for r in analysis['top_5_r2']]}")
        
        print(f"\n🎯 Top Feature Importance:")
        sorted_importance = sorted(analysis['feature_importance_best'].items(), 
                                 key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_importance[:5]:
            print(f"   {feature}: {importance:.3f}")
        
        print("="*80)
    else:
        print("❌ GAM hyperparameter tuning failed")


if __name__ == "__main__":
    main()