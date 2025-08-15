#!/usr/bin/env python3
"""
Experiment 2: UltraFeedback Validation
- Merge your base + human feedback + UF judge scores
- Train MLP and/or GAM aggregators

Usage:
  python -m experiments.2_ultrafeedback_validation.run_experiment \
    --base /path/to/base.pkl \
    --human /path/to/human_feedback.pkl \
    --judges experiments/2_ultrafeedback_validation/results/judge_scores_uf.pkl \
    --out-dir experiments/2_ultrafeedback_validation/results \
    --train-mlp --train-gam
"""
import logging
from pathlib import Path
import pickle
import pandas as pd

from pipeline.utils.data_merger import DataMerger
from pipeline.core.aggregator_training import (
    MLPTrainer, GAMAggregator, compute_metrics
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="UltraFeedback Validation Experiment")
    parser.add_argument('--base', required=True, help='Base data (instruction, answer) .pkl')
    parser.add_argument('--human', required=True, help='Human feedback .pkl')
    parser.add_argument('--judges', required=True, help='UF judge scores .pkl')
    parser.add_argument('--out-dir', default='experiments/2_ultrafeedback_validation/results', help='Output dir')
    parser.add_argument('--train-mlp', action='store_true', help='Train MLP aggregator')
    parser.add_argument('--train-gam', action='store_true', help='Train GAM aggregator (requires pygam)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    merger = DataMerger()
    base = merger.load_base_data(args.base)
    human = merger.load_human_feedback(args.human)
    judges = merger.load_judge_scores(args.judges)

    # Normalize UF judge columns to consistent naming: store scores under 'judges'
    if 'judges' not in judges.columns:
        # if saved as 'judge_scores' or similar, try to auto-detect
        for c in ['judge_scores', 'scores', 'evaluations']:
            if c in judges.columns:
                judges = judges.rename(columns={c: 'judges'})
                break
    
    # Merge and save merged
    merged = merger.merge_datasets(base, human, judges, output_path=out_dir / 'merged_uf.pkl')

    # Prepare training data (will normalize: judges to [0,1], human_score to [0,1])
    X, y = merger.prepare_training_data(merged, normalize=True, drop_incomplete=True)

    # Train GAM
    if args.train_gam:
        try:
            logger.info("Training GAM aggregator...")
            gam = GAMAggregator(n_splines=10, lam=0.6)
            gam.fit(X, y)
            pred = gam.predict(X)
            metrics = compute_metrics(y, pred)
            logger.info(f"GAM metrics (train set): {metrics}")
            with open(out_dir / 'gam_model.pkl', 'wb') as f:
                pickle.dump(gam, f)
        except Exception as e:
            logger.warning(f"GAM training skipped/failed: {e}")

    # Train MLP
    if args.train_mlp:
        logger.info("Training MLP aggregator...")
        trainer = MLPTrainer(hidden_dim=64, learning_rate=1e-3, batch_size=32, n_epochs=50)
        trainer.fit(X, y)
        pred = trainer.predict(X)
        metrics = compute_metrics(y, pred)
        logger.info(f"MLP metrics (train set): {metrics}")
        trainer.save_model(out_dir / 'mlp_model.pt')

    logger.info("Experiment completed. Outputs saved in %s", out_dir)


if __name__ == '__main__':
    main()
