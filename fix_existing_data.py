#!/usr/bin/env python3
"""
Fix Existing Experiment Data

This script:
1. Loads the existing experiment data
2. Removes Ethicist persona scores from the data
3. Saves the cleaned data back to the same location
4. Re-runs the analysis with corrected baselines
"""

import pickle
import pandas as pd
from pathlib import Path
import json

def remove_ethicist_from_data(experiment_dir: str):
    """Remove Ethicist persona from existing experiment data."""
    experiment_path = Path(experiment_dir)
    data_path = experiment_path / "data" / "data_with_judge_scores.pkl"
    
    print(f"🔍 Loading data from {data_path}")
    
    # Load the data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, list):
        data = pd.DataFrame(data)
    
    print(f"📊 Loaded {len(data)} samples")
    
    # Process each row to remove Ethicist
    ethicist_removed_count = 0
    
    for idx, row in data.iterrows():
        if 'human_feedback' in row and isinstance(row['human_feedback'], dict):
            if 'personas' in row['human_feedback']:
                personas = row['human_feedback']['personas']
                if 'Ethicist' in personas:
                    del personas['Ethicist']
                    ethicist_removed_count += 1
                
                # Recalculate average score without Ethicist
                valid_scores = [p['score'] for p in personas.values() 
                              if isinstance(p, dict) and 'score' in p and p['score'] is not None]
                
                if valid_scores:
                    new_avg = sum(valid_scores) / len(valid_scores)
                    row['human_feedback']['average_score'] = new_avg
                    row['human_feedback']['score'] = new_avg  # For compatibility
                    row['human_feedback']['valid_personas'] = len(valid_scores)
    
    print(f"✅ Removed Ethicist from {ethicist_removed_count} samples")
    print(f"📝 Updated average scores for all samples")
    
    # Save the cleaned data
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"💾 Saved cleaned data back to {data_path}")
    
    return data

def main():
    experiment_dir = "/Users/eitu/Documents/Eitu/AI Safety/AIS_hackathons/model_routing/multi-judge-interpretability/results/full_experiments/full_experiment_10000_enhanced"
    
    print("🚀 Starting data cleanup process...")
    
    # Remove Ethicist from the data
    cleaned_data = remove_ethicist_from_data(experiment_dir)
    
    # Re-run the analysis
    print("\n🔬 Re-running analysis with corrected baselines...")
    
    from analyze_existing_experiment import ExistingExperimentAnalyzer
    
    analyzer = ExistingExperimentAnalyzer(
        experiment_dir=experiment_dir,
        random_seed=42
    )
    
    analyzer.run_analysis()
    
    print("\n🎉 Data cleanup and analysis complete!")

if __name__ == "__main__":
    main()