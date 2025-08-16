#!/usr/bin/env python3
"""
Quick script to inspect the persona data structure from the baseline experiment
"""

import pickle
import pandas as pd
import json
from pathlib import Path

def inspect_persona_data():
    """Inspect the persona data structure."""
    data_path = "baseline_experiment_results/data_with_all_personas.pkl"
    
    print(f"=== Inspecting {data_path} ===")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, pd.DataFrame):
        print(f"DataFrame with {len(data)} rows and {len(data.columns)} columns")
        print(f"Columns: {list(data.columns)}")
        
        # Look at first sample
        if len(data) > 0:
            print(f"\nFirst sample:")
            sample = data.iloc[0]
            print(f"Instruction: {sample['instruction'][:100]}...")
            print(f"Answer: {sample['answer'][:100]}...")
            
            if 'human_feedback' in sample:
                hf = sample['human_feedback']
                print(f"\nHuman feedback type: {type(hf)}")
                
                if isinstance(hf, dict):
                    print(f"Human feedback keys: {list(hf.keys())}")
                    
                    if 'personas' in hf:
                        personas = hf['personas']
                        print(f"\nNumber of personas: {len(personas)}")
                        print(f"Persona names: {list(personas.keys())}")
                        
                        # Show sample persona feedback
                        for i, (persona_name, feedback) in enumerate(personas.items()):
                            if i < 3:  # Show first 3 personas
                                print(f"\n{persona_name}:")
                                print(f"  Score: {feedback.get('score', 'N/A')}")
                                print(f"  Analysis: {feedback.get('analysis', 'N/A')[:80]}...")
                    
                    if 'average_score' in hf:
                        print(f"\nAverage score: {hf['average_score']}")
                        
            # Count valid personas per sample
            valid_personas_counts = []
            for idx, row in data.iterrows():
                if 'human_feedback' in row and 'personas' in row['human_feedback']:
                    personas = row['human_feedback']['personas']
                    valid_count = sum(1 for p in personas.values() if 'score' in p)
                    valid_personas_counts.append(valid_count)
            
            if valid_personas_counts:
                print(f"\nPersona statistics across all samples:")
                print(f"  Average personas per sample: {sum(valid_personas_counts) / len(valid_personas_counts):.1f}")
                print(f"  Min personas per sample: {min(valid_personas_counts)}")
                print(f"  Max personas per sample: {max(valid_personas_counts)}")

if __name__ == "__main__":
    inspect_persona_data()