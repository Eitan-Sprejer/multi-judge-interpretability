#!/usr/bin/env python3
"""
Data Loading and Preparation for Experiment 2b: Aggregator Validation with Less Varied Data

This module provides functions to load and prepare different ground truth targets
for testing the hypothesis that R² scores are limited by ground truth variance.
"""

import pickle
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from pipeline.core.persona_simulation import PERSONAS


def load_experiment_data(data_path: str) -> pd.DataFrame:
    """
    Load the experiment data with judge scores and ultrafeedback scores.
    
    Args:
        data_path: Path to the pickle file with judge scores and ultrafeedback
        
    Returns:
        DataFrame with the loaded data
    """
    print(f"📂 Loading experiment data from {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, list):
        data = pd.DataFrame(data)
    
    print(f"✅ Loaded {len(data)} samples")
    
    # Validate data structure
    required_columns = ['judge_scores', 'human_feedback', 'ultrafeedback_overall_score']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"   Columns available: {list(data.columns)}")
    return data


def extract_judge_scores(data: pd.DataFrame) -> np.ndarray:
    """
    Extract judge scores matrix (X) from the data.
    
    Args:
        data: DataFrame with experiment data
        
    Returns:
        Judge scores matrix of shape (n_samples, n_judges)
    """
    print("🎯 Extracting judge scores...")
    
    judge_scores_list = []
    
    for idx, row in data.iterrows():
        if 'judge_scores' in row and isinstance(row['judge_scores'], list):
            judge_scores_list.append(row['judge_scores'])
        else:
            print(f"⚠️  Warning: Missing or invalid judge scores for sample {idx}")
    
    X = np.array(judge_scores_list)
    
    print(f"✅ Extracted judge scores matrix: shape {X.shape}")
    print(f"   Range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   Mean: {X.mean():.2f}, Std: {X.std():.2f}")
    
    return X


def prepare_mixed_personas_baseline(data: pd.DataFrame, random_seed: int = 42) -> np.ndarray:
    """
    Prepare mixed persona scores using uniform sampling (matching baseline methodology).
    
    Args:
        data: DataFrame with experiment data
        random_seed: Random seed for reproducibility
        
    Returns:
        Mixed persona scores array
    """
    print("🎭 Preparing mixed persona scores (baseline methodology)...")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Uniform persona sampling (same as baseline)
    available_personas = list(PERSONAS.keys())
    samples_per_persona = len(data) // len(available_personas)
    remaining_samples = len(data) % len(available_personas)
    
    persona_assignment = []
    for persona in available_personas:
        persona_assignment.extend([persona] * samples_per_persona)
    for _ in range(remaining_samples):
        persona_assignment.append(random.choice(available_personas))
    random.shuffle(persona_assignment)
    
    # Extract scores using assignment
    y_mixed = []
    valid_samples = 0
    
    for idx, (row, assigned_persona) in enumerate(zip(data.iterrows(), persona_assignment)):
        row_data = row[1]
        
        if ('human_feedback' not in row_data or 'personas' not in row_data['human_feedback']):
            continue
            
        personas_feedback = row_data['human_feedback']['personas']
        if assigned_persona not in personas_feedback or 'score' not in personas_feedback[assigned_persona]:
            continue
            
        selected_score = personas_feedback[assigned_persona]['score']
        if selected_score is None:
            continue
            
        y_mixed.append(selected_score)
        valid_samples += 1
    
    y_mixed = np.array(y_mixed)
    
    print(f"✅ Prepared mixed persona scores: {len(y_mixed)} samples")
    print(f"   Range: [{y_mixed.min():.2f}, {y_mixed.max():.2f}]")
    print(f"   Mean: {y_mixed.mean():.2f}, Std: {y_mixed.std():.2f}")
    print(f"   Variance: {np.var(y_mixed):.3f}")
    
    return y_mixed


def extract_ultrafeedback_scores(data: pd.DataFrame) -> np.ndarray:
    """
    Extract UltraFeedback overall_score as ground truth.
    
    Args:
        data: DataFrame with experiment data
        
    Returns:
        UltraFeedback scores array
    """
    print("🌟 Extracting UltraFeedback overall scores...")
    
    if 'ultrafeedback_overall_score' not in data.columns:
        raise ValueError("ultrafeedback_overall_score column not found in data")
    
    # Convert to numeric array, handling any non-numeric values
    y_ultrafeedback = pd.to_numeric(data['ultrafeedback_overall_score'], errors='coerce').values
    
    # Remove any NaN values
    valid_mask = ~np.isnan(y_ultrafeedback)
    y_ultrafeedback = y_ultrafeedback[valid_mask]
    
    print(f"✅ Extracted UltraFeedback scores: {len(y_ultrafeedback)} samples")
    print(f"   Range: [{y_ultrafeedback.min():.2f}, {y_ultrafeedback.max():.2f}]")
    print(f"   Mean: {y_ultrafeedback.mean():.2f}, Std: {y_ultrafeedback.std():.2f}")
    print(f"   Variance: {np.var(y_ultrafeedback):.3f}")
    
    return y_ultrafeedback


def extract_single_persona_scores(data: pd.DataFrame, persona_name: str) -> np.ndarray:
    """
    Extract scores for a single persona.
    
    Args:
        data: DataFrame with experiment data
        persona_name: Name of the persona to extract
        
    Returns:
        Single persona scores array
    """
    print(f"👤 Extracting {persona_name} persona scores...")
    
    if persona_name not in PERSONAS:
        raise ValueError(f"Unknown persona: {persona_name}")
    
    y_persona = []
    
    for idx, row in data.iterrows():
        if ('human_feedback' not in row or 'personas' not in row['human_feedback']):
            continue
            
        personas_feedback = row['human_feedback']['personas']
        if persona_name not in personas_feedback or 'score' not in personas_feedback[persona_name]:
            continue
            
        score = personas_feedback[persona_name]['score']
        if score is None:
            continue
            
        y_persona.append(score)
    
    y_persona = np.array(y_persona)
    
    print(f"✅ Extracted {persona_name} scores: {len(y_persona)} samples")
    print(f"   Range: [{y_persona.min():.2f}, {y_persona.max():.2f}]")
    print(f"   Mean: {y_persona.mean():.2f}, Std: {y_persona.std():.2f}")
    print(f"   Variance: {np.var(y_persona):.3f}")
    
    return y_persona


def calculate_mean_of_all_personas(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate mean scores across all personas for each sample (bonus experiment).
    
    Args:
        data: DataFrame with experiment data
        
    Returns:
        Mean persona scores array
    """
    print("📊 Calculating mean of all persona scores...")
    
    y_mean_personas = []
    
    for idx, row in data.iterrows():
        if ('human_feedback' not in row or 'personas' not in row['human_feedback']):
            continue
            
        personas_feedback = row['human_feedback']['personas']
        sample_scores = []
        
        for persona_name in PERSONAS.keys():
            if persona_name in personas_feedback and 'score' in personas_feedback[persona_name]:
                score = personas_feedback[persona_name]['score']
                if score is not None:
                    sample_scores.append(score)
        
        if len(sample_scores) > 0:
            y_mean_personas.append(np.mean(sample_scores))
    
    y_mean_personas = np.array(y_mean_personas)
    
    print(f"✅ Calculated mean persona scores: {len(y_mean_personas)} samples")
    print(f"   Range: [{y_mean_personas.min():.2f}, {y_mean_personas.max():.2f}]")
    print(f"   Mean: {y_mean_personas.mean():.2f}, Std: {y_mean_personas.std():.2f}")
    print(f"   Variance: {np.var(y_mean_personas):.3f}")
    
    return y_mean_personas


def prepare_all_targets(data: pd.DataFrame, random_seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Prepare all ground truth targets for the experiment.
    
    Args:
        data: DataFrame with experiment data
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with all target arrays
    """
    print("🎯 Preparing all ground truth targets...")
    
    targets = {}
    
    # Extract judge scores (same for all experiments)
    targets['X'] = extract_judge_scores(data)
    
    # Mixed personas (baseline)
    targets['y_mixed'] = prepare_mixed_personas_baseline(data, random_seed)
    
    # UltraFeedback scores
    targets['y_ultrafeedback'] = extract_ultrafeedback_scores(data)
    
    # Individual persona scores
    targets['y_personas'] = {}
    for persona_name in PERSONAS.keys():
        targets['y_personas'][persona_name] = extract_single_persona_scores(data, persona_name)
    
    # Mean of all personas (bonus)
    targets['y_persona_mean'] = calculate_mean_of_all_personas(data)
    
    print(f"\n📋 Summary of prepared targets:")
    print(f"   Judge scores (X): {targets['X'].shape}")
    print(f"   Mixed personas: {len(targets['y_mixed'])} samples")
    print(f"   UltraFeedback: {len(targets['y_ultrafeedback'])} samples")
    print(f"   Individual personas: {len(targets['y_personas'])} personas")
    print(f"   Persona mean: {len(targets['y_persona_mean'])} samples")
    
    return targets


def validate_data_alignment(targets: Dict[str, np.ndarray]) -> bool:
    """
    Validate that all target arrays have compatible sample counts.
    
    Args:
        targets: Dictionary with all target arrays
        
    Returns:
        True if data is aligned, raises exception otherwise
    """
    print("🔍 Validating data alignment...")
    
    X_samples = len(targets['X'])
    
    # For mixed personas and ultrafeedback, we need to ensure they're aligned with X
    min_samples = min(
        len(targets['y_mixed']),
        len(targets['y_ultrafeedback']),
        len(targets['y_persona_mean'])
    )
    
    print(f"   Judge scores: {X_samples} samples")
    print(f"   Mixed personas: {len(targets['y_mixed'])} samples")
    print(f"   UltraFeedback: {len(targets['y_ultrafeedback'])} samples")
    print(f"   Persona mean: {len(targets['y_persona_mean'])} samples")
    print(f"   Minimum aligned samples: {min_samples}")
    
    # Check individual personas
    persona_samples = {name: len(scores) for name, scores in targets['y_personas'].items()}
    print(f"   Individual persona samples: {persona_samples}")
    
    if min_samples < 100:
        raise ValueError(f"Insufficient aligned samples: {min_samples}")
    
    print("✅ Data alignment validated")
    return True


if __name__ == "__main__":
    # Test the data preparation functions
    data_path = "/Users/eitu/Documents/Eitu/AI Safety/AIS_hackathons/model_routing/multi-judge-interpretability/results/full_experiments/baseline_ultrafeedback_2000samples_20250816_213023/data/data_with_judge_scores_and_ultrafeedback.pkl"
    
    print("🧪 Testing data preparation functions...")
    
    # Load data
    data = load_experiment_data(data_path)
    
    # Prepare all targets
    targets = prepare_all_targets(data)
    
    # Validate alignment
    validate_data_alignment(targets)
    
    print("\n✅ Data preparation functions tested successfully!")