#!/usr/bin/env python3
"""
Re-run Analysis Only Script

This script re-runs only the analysis phase (Steps 3-5) using the already
collected score data, without making any new API calls.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.experiment_runner_v2 import RubricSensitivityExperimentV2


async def rerun_analysis(experiment_dir: str):
    """
    Re-run analysis using existing data from a completed experiment.
    
    Args:
        experiment_dir: Path to experiment directory with cached data
    """
    experiment_path = Path(experiment_dir)
    
    # Check required files exist
    required_files = [
        "raw_scores.pkl",
        "variant_scores_cache.pkl", 
        "config.json"
    ]
    
    for file_name in required_files:
        file_path = experiment_path / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    print(f"🔄 Re-running analysis using data from: {experiment_dir}")
    print("=" * 60)
    
    # Create a new experiment instance pointing to existing data
    experiment = RubricSensitivityExperimentV2(
        data_path="../../dataset/data_with_judge_scores.pkl",
        output_dir=experiment_dir,  # Use same directory  
        n_examples=1000,
        use_real_api=False  # We already have the data
    )
    
    try:
        # Load existing scores 
        import pickle
        scores_path = experiment_path / "raw_scores.pkl"
        with open(scores_path, 'rb') as f:
            scores_df = pickle.load(f)
        
        print(f"✅ Loaded existing scores: {scores_df.shape}")
        
        # Step 3: Analyze robustness (fixed version)
        print("🔍 Step 3: Running robustness analysis...")
        robustness_report = experiment.analyze_robustness(scores_df)
        
        # Step 4: Generate visualizations (fixed version)  
        print("📊 Step 4: Generating visualizations...")
        try:
            experiment.analyzer.create_robustness_plots(
                output_dir=str(experiment_path / "plots"),
                report=robustness_report
            )
            print("✅ Visualizations created successfully")
        except Exception as e:
            print(f"⚠️ Visualization generation failed: {e}")
        
        # Step 5: Generate summary (fixed version)
        print("📝 Step 5: Generating summary...")
        summary = experiment.generate_summary(robustness_report)
        
        print("=" * 60)
        print("🎉 ANALYSIS COMPLETE!")
        print("=" * 60)
        print(summary)
        
        return robustness_report
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rerun_analysis.py <experiment_directory>")
        print("Example: python rerun_analysis.py ../results_full_20250818_215910")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    asyncio.run(rerun_analysis(experiment_dir))