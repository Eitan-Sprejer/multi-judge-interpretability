#!/usr/bin/env python3
"""
Baseline Multi-Judge Interpretability Experiment

This script runs the complete baseline experiment pipeline:
1. Load base UltraFeedback data subset
2. Run ALL personas on each sample (not random selection)
3. Generate judge scores for each sample
4. Train aggregation models (GAM/MLP) with random persona selection
5. Evaluate performance

Key change: All personas evaluate each sample, random selection happens during training.

Usage:
  python run_baseline_experiment.py --data-size 100 --dry-run  # small test
  python run_baseline_experiment.py --data-size 10000        # full experiment
"""

import asyncio
import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Pipeline imports
from pipeline.core.persona_simulation import PersonaSimulator, PERSONAS
from pipeline.core.judge_evaluation import JudgeEvaluator  
from pipeline.core.aggregator_training import MLPTrainer, GAMAggregator, compute_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_DATA_PATH = "dataset/data.pkl"
OUTPUT_DIR = Path("baseline_experiment_results")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


class BaselineExperiment:
    """Runs the complete baseline experiment with all personas evaluation."""
    
    def __init__(
        self,
        data_size: int = 10000,
        test_size: float = 0.2,
        random_seed: int = 42,
        concurrency: int = 5,  # Lower for small test
        checkpoint_interval: int = 50  # More frequent for small test
    ):
        self.data_size = data_size
        self.test_size = test_size
        self.random_seed = random_seed
        self.concurrency = concurrency
        self.checkpoint_interval = checkpoint_interval
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create output directories
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.persona_simulator = PersonaSimulator()
        self.judge_evaluator = None  # Will be initialized when needed
        
    def load_base_data(self) -> pd.DataFrame:
        """Load and sample base UltraFeedback data."""
        logger.info(f"Loading base data from {BASE_DATA_PATH}")
        
        with open(BASE_DATA_PATH, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        # Sample data
        if len(data) > self.data_size:
            data = data.sample(n=self.data_size, random_state=self.random_seed)
            logger.info(f"Sampled {self.data_size} examples from {len(data)} total")
        
        # Ensure required columns
        required_cols = ['instruction', 'answer']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Loaded {len(data)} samples with columns: {list(data.columns)}")
        return data.reset_index(drop=True)
    
    async def run_persona_simulation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run ALL personas on each sample and store all feedback.
        
        This is the key change: instead of random persona selection,
        we run all personas and store all their feedback.
        """
        logger.info("Running persona simulation with ALL personas on each sample")
        
        # Check if already done
        persona_file = OUTPUT_DIR / "data_with_all_personas.pkl"
        if persona_file.exists():
            logger.info(f"Found existing persona data at {persona_file}")
            with open(persona_file, 'rb') as f:
                return pickle.load(f)
        
        # Run simulation
        data_with_personas = await self.persona_simulator.simulate_dataset(
            data,
            question_col='instruction',
            answer_col='answer',
            concurrency=self.concurrency,
            checkpoint_interval=self.checkpoint_interval,
            checkpoint_dir=CHECKPOINT_DIR
        )
        
        # Save results
        with open(persona_file, 'wb') as f:
            pickle.dump(data_with_personas, f)
        
        logger.info(f"Completed persona simulation. Saved to {persona_file}")
        return data_with_personas
    
    def get_judge_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate judge scores for each sample using mock evaluation.
        
        For the full experiment, this should use the real Martian API.
        For this baseline test, we'll use a simplified mock evaluator.
        """
        logger.info("Generating judge scores for each sample")
        
        # Check if already done
        judge_file = OUTPUT_DIR / "data_with_judge_scores.pkl"
        if judge_file.exists():
            logger.info(f"Found existing judge scores at {judge_file}")
            with open(judge_file, 'rb') as f:
                return pickle.load(f)
        
        # Initialize mock judge evaluator
        if self.judge_evaluator is None:
            self.judge_evaluator = MockJudgeEvaluator()
        
        # Generate scores
        judge_scores = []
        for idx, row in data.iterrows():
            scores = self.judge_evaluator.evaluate(row['instruction'], row['answer'])
            judge_scores.append(scores)
            
            if (idx + 1) % 20 == 0:  # More frequent logging for small test
                logger.info(f"Generated judge scores for {idx + 1}/{len(data)} samples")
        
        # Add judge scores to data
        data['judge_scores'] = judge_scores
        
        # Save results
        with open(judge_file, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Completed judge scoring. Saved to {judge_file}")
        return data
    
    def prepare_training_data(self, data: pd.DataFrame) -> tuple:
        """
        Prepare training data with random persona selection.
        
        This is where we implement the random persona selection:
        - Each sample has feedback from ALL personas
        - For training, we randomly select one persona per sample
        """
        logger.info("Preparing training data with random persona selection")
        
        X_list = []
        y_list = []
        persona_selections = []
        
        for idx, row in data.iterrows():
            # Get all persona feedback
            if 'human_feedback' not in row or 'personas' not in row['human_feedback']:
                logger.warning(f"Row {idx} missing persona feedback, skipping")
                continue
            
            personas_feedback = row['human_feedback']['personas']
            judge_scores = row['judge_scores']
            
            # Randomly select one persona for this sample
            available_personas = [p for p in personas_feedback.keys() if 'score' in personas_feedback[p]]
            if not available_personas:
                logger.warning(f"Row {idx} has no valid persona scores, skipping")
                continue
            
            selected_persona = random.choice(available_personas)
            selected_score = personas_feedback[selected_persona]['score']
            
            X_list.append(judge_scores)
            y_list.append(selected_score)
            persona_selections.append(selected_persona)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Prepared {len(X)} training samples")
        logger.info(f"Persona selection distribution: {pd.Series(persona_selections).value_counts().to_dict()}")
        
        return X, y, persona_selections
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train both GAM and MLP aggregation models."""
        logger.info("Training aggregation models")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        results = {}
        
        # Train GAM
        try:
            logger.info("Training GAM model...")
            gam = GAMAggregator(n_splines=10, lam=0.6)
            gam.fit(X_train, y_train)
            
            # Evaluate
            train_pred = gam.predict(X_train)
            test_pred = gam.predict(X_test)
            
            gam_results = {
                'train_metrics': compute_metrics(y_train, train_pred),
                'test_metrics': compute_metrics(y_test, test_pred),
                'feature_importance': gam.get_feature_importance()
            }
            
            results['gam'] = gam_results
            
            # Save model
            gam_path = OUTPUT_DIR / 'gam_model.pkl'
            with open(gam_path, 'wb') as f:
                pickle.dump(gam, f)
            
            logger.info("GAM Results:")
            logger.info(f"  Train - R²: {gam_results['train_metrics']['r2']:.4f}, MAE: {gam_results['train_metrics']['mae']:.4f}")
            logger.info(f"  Test  - R²: {gam_results['test_metrics']['r2']:.4f}, MAE: {gam_results['test_metrics']['mae']:.4f}")
            
        except ImportError:
            logger.warning("PyGAM not installed. Skipping GAM training.")
        except Exception as e:
            logger.error(f"GAM training failed: {e}")
        
        # Train MLP
        try:
            logger.info("Training MLP model...")
            mlp_trainer = MLPTrainer(
                hidden_dim=64,
                learning_rate=0.001,
                batch_size=32,
                n_epochs=100
            )
            
            train_losses, val_losses = mlp_trainer.fit(X_train, y_train, X_test, y_test)
            
            # Evaluate
            train_pred = mlp_trainer.predict(X_train)
            test_pred = mlp_trainer.predict(X_test)
            
            mlp_results = {
                'train_metrics': compute_metrics(y_train, train_pred),
                'test_metrics': compute_metrics(y_test, test_pred),
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            results['mlp'] = mlp_results
            
            # Save model
            mlp_path = OUTPUT_DIR / 'mlp_model.pt'
            mlp_trainer.save_model(mlp_path)
            
            logger.info("MLP Results:")
            logger.info(f"  Train - R²: {mlp_results['train_metrics']['r2']:.4f}, MAE: {mlp_results['train_metrics']['mae']:.4f}")
            logger.info(f"  Test  - R²: {mlp_results['test_metrics']['r2']:.4f}, MAE: {mlp_results['test_metrics']['mae']:.4f}")
            
        except Exception as e:
            logger.error(f"MLP training failed: {e}")
        
        return results
    
    async def run_full_experiment(self) -> Dict[str, Any]:
        """Run the complete baseline experiment."""
        logger.info("Starting baseline experiment")
        
        # Step 1: Load base data
        data = self.load_base_data()
        
        # Step 2: Run persona simulation (ALL personas)
        data_with_personas = await self.run_persona_simulation(data)
        
        # Step 3: Generate judge scores
        data_with_judges = self.get_judge_scores(data_with_personas)
        
        # Step 4: Prepare training data (random persona selection)
        X, y, persona_selections = self.prepare_training_data(data_with_judges)
        
        # Step 5: Train models
        results = self.train_models(X, y)
        
        # Step 6: Generate report
        experiment_results = {
            'experiment_config': {
                'data_size': self.data_size,
                'test_size': self.test_size,
                'random_seed': self.random_seed,
                'concurrency': self.concurrency
            },
            'data_stats': {
                'total_samples': len(data_with_judges),
                'training_samples': len(X),
                'persona_distribution': pd.Series(persona_selections).value_counts().to_dict()
            },
            'model_results': results
        }
        
        # Save experiment results
        results_path = OUTPUT_DIR / 'experiment_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(experiment_results, f)
        
        logger.info(f"Experiment completed. Results saved to {OUTPUT_DIR}")
        return experiment_results


class MockJudgeEvaluator:
    """
    Mock judge evaluator for testing.
    
    In the full implementation, this would use the Martian API
    to evaluate samples with the 10 specialized judges.
    """
    
    def __init__(self):
        self.n_judges = 10
        
    def evaluate(self, instruction: str, answer: str) -> List[float]:
        """Generate mock judge scores based on answer characteristics."""
        # Mock scoring based on answer length and content
        answer_len = len(answer)
        
        # Simulate different judge behaviors
        scores = []
        for i in range(self.n_judges):
            # Base score on answer length (longer = higher quality, with noise)
            base_score = min(4.0, max(1.0, (answer_len / 100) + 1.0))
            
            # Add judge-specific bias and noise
            judge_bias = (i / self.n_judges) * 0.5  # Some judges are stricter
            noise = random.gauss(0, 0.3)  # Random noise
            
            score = base_score - judge_bias + noise
            score = max(1.0, min(4.0, score))  # Clamp to 1-4 range
            scores.append(score)
        
        return scores


def create_experiment_report(results: Dict[str, Any]) -> str:
    """Generate a formatted experiment report."""
    report = []
    report.append("=" * 80)
    report.append("BASELINE MULTI-JUDGE INTERPRETABILITY EXPERIMENT REPORT")
    report.append("=" * 80)
    
    # Configuration
    config = results['experiment_config']
    report.append(f"\nExperiment Configuration:")
    report.append(f"  Data Size: {config['data_size']}")
    report.append(f"  Test Split: {config['test_size']}")
    report.append(f"  Random Seed: {config['random_seed']}")
    report.append(f"  Concurrency: {config['concurrency']}")
    
    # Data statistics
    stats = results['data_stats']
    report.append(f"\nData Statistics:")
    report.append(f"  Total Samples: {stats['total_samples']}")
    report.append(f"  Training Samples: {stats['training_samples']}")
    report.append(f"  Persona Distribution:")
    for persona, count in stats['persona_distribution'].items():
        report.append(f"    {persona}: {count}")
    
    # Model results
    model_results = results['model_results']
    report.append(f"\nModel Performance:")
    
    if 'gam' in model_results:
        gam = model_results['gam']
        report.append(f"\n  GAM (Generalized Additive Model):")
        report.append(f"    Train R²: {gam['train_metrics']['r2']:.4f}")
        report.append(f"    Test R²:  {gam['test_metrics']['r2']:.4f}")
        report.append(f"    Train MAE: {gam['train_metrics']['mae']:.4f}")
        report.append(f"    Test MAE:  {gam['test_metrics']['mae']:.4f}")
        
        report.append(f"\n    Feature Importance (Top 5):")
        importance = gam['feature_importance']
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_importance[:5]):
            report.append(f"      {i+1}. {feature}: {score:.3f}")
    
    if 'mlp' in model_results:
        mlp = model_results['mlp']
        report.append(f"\n  MLP (Multi-Layer Perceptron):")
        report.append(f"    Train R²: {mlp['train_metrics']['r2']:.4f}")
        report.append(f"    Test R²:  {mlp['test_metrics']['r2']:.4f}")
        report.append(f"    Train MAE: {mlp['train_metrics']['mae']:.4f}")
        report.append(f"    Test MAE:  {mlp['test_metrics']['mae']:.4f}")
    
    # Baseline comparison
    report.append(f"\nBaseline Comparison:")
    report.append(f"  Random Baseline R²: ~0.00")
    report.append(f"  Mean Prediction R²: ~0.10-0.20")
    report.append(f"  Target Performance: R² > 0.50")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Multi-Judge Interpretability Experiment")
    parser.add_argument('--data-size', type=int, default=10000,
                        help='Number of samples to use (default: 10000)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set fraction (default: 0.2)')
    parser.add_argument('--concurrency', type=int, default=10,
                        help='API concurrency (default: 10)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run with small dataset for testing')
    
    args = parser.parse_args()
    
    # Adjust settings for dry run
    if args.dry_run:
        args.data_size = min(args.data_size, 100)
        args.concurrency = min(args.concurrency, 5)
        logger.info("DRY RUN MODE: Using smaller dataset and lower concurrency")
    
    # Create and run experiment
    experiment = BaselineExperiment(
        data_size=args.data_size,
        test_size=args.test_size,
        random_seed=args.random_seed,
        concurrency=args.concurrency,
        checkpoint_interval=50 if args.dry_run else 100
    )
    
    try:
        results = await experiment.run_full_experiment()
        
        # Generate and save report
        report = create_experiment_report(results)
        report_path = OUTPUT_DIR / 'experiment_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\nDetailed results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())