#!/usr/bin/env python3
"""
Quick script to inspect the data structure to understand current format
"""

import pickle
import pandas as pd
import json
from pathlib import Path

def inspect_data(file_path):
    """Inspect data structure from a pickle file."""
    print(f"\n=== Inspecting {file_path} ===")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, pd.DataFrame):
            print(f"DataFrame with {len(data)} rows and {len(data.columns)} columns")
            print(f"Columns: {list(data.columns)}")
            
            # Look at first few rows
            if len(data) > 0:
                print(f"\nFirst row keys: {data.iloc[0].keys() if hasattr(data.iloc[0], 'keys') else 'Not dict-like'}")
                
                # Check human_feedback structure
                if 'human_feedback' in data.columns:
                    hf = data.iloc[0]['human_feedback']
                    print(f"human_feedback type: {type(hf)}")
                    if isinstance(hf, dict):
                        print(f"human_feedback keys: {list(hf.keys())}")
                        if 'personas' in hf:
                            print(f"Number of personas: {len(hf['personas'])}")
                            print(f"Persona names: {list(hf['personas'].keys())}")
                            first_persona = list(hf['personas'].values())[0]
                            print(f"First persona structure: {first_persona}")
                    else:
                        print(f"human_feedback content: {hf}")
                
                # Check if judge scores exist
                if 'judge_scores' in data.columns:
                    js = data.iloc[0]['judge_scores']
                    print(f"judge_scores type: {type(js)}")
                    print(f"judge_scores length: {len(js) if hasattr(js, '__len__') else 'No length'}")
                    print(f"judge_scores sample: {js}")
                
        else:
            print(f"Data type: {type(data)}")
            if hasattr(data, '__len__'):
                print(f"Length: {len(data)}")
            if isinstance(data, list) and len(data) > 0:
                print(f"First item type: {type(data[0])}")
                print(f"First item: {data[0]}")
                
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Check existing datasets
dataset_dir = Path("dataset")
if dataset_dir.exists():
    for file_path in dataset_dir.glob("*.pkl"):
        inspect_data(file_path)
    for file_path in dataset_dir.glob("*.pickle"):
        inspect_data(file_path)
else:
    print("Dataset directory not found")