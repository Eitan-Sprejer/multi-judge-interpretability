#!/usr/bin/env python3
"""
Create Transformation Visualization for Appendix

Shows how different bias transformations affect judge score distributions
at various strength levels. This helps readers understand what each 
transformation actually does to the data.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

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

def load_sample_judge_scores():
    """Load a sample of judge scores for visualization."""
    results_dir = Path(__file__).parent / 'results_full_20250818_215910'
    scores_path = results_dir / 'restructured_scores_fixed.pkl'
    
    with open(scores_path, 'rb') as f:
        scores_df = pickle.load(f)
    
    # Get original judge scores (first 500 samples for faster plotting)
    original_cols = [col for col in scores_df.columns if col.endswith('_original')]
    judge_scores = scores_df[original_cols].values[:500]
    judge_names = [col.replace('_original', '') for col in original_cols]
    
    return judge_scores, judge_names

def create_transformation_overview():
    """Create comprehensive transformation visualization."""
    
    judge_scores, judge_names = load_sample_judge_scores()
    
    # Select a few representative judges for cleaner visualization
    selected_judges = [0, 1, 4, 7]  # truthfulness, harmlessness, explanatory-depth, conciseness
    selected_names = [judge_names[i] for i in selected_judges]
    
    # Transformation configurations
    transformations = {
        'Bottom Heavy': ('bottom_heavy', [0.0, 0.4, 0.8, 1.0]),
        'Top Heavy': ('top_heavy', [0.0, 0.4, 0.8, 1.0]),
        'Middle Heavy': ('middle_heavy', [0.0, 0.4, 0.8, 1.0]),
        'Systematic Shift +': ('systematic_shift_positive', [0.0, 0.3, 0.6, 1.0]),
        'Systematic Shift -': ('systematic_shift_negative', [0.0, 0.3, 0.6, 1.0])
    }
    
    # Create large figure
    n_transformations = len(transformations)
    n_strengths = 4
    fig, axes = plt.subplots(n_transformations, n_strengths, figsize=(16, 20))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(selected_judges)))
    
    for t_idx, (transform_name, (transform_type, strengths)) in enumerate(transformations.items()):
        for s_idx, strength in enumerate(strengths):
            ax = axes[t_idx, s_idx]
            
            # Apply transformation to selected judges
            for j_idx, judge_idx in enumerate(selected_judges):
                original_scores = judge_scores[:, judge_idx]
                
                if transform_type == 'systematic_shift_positive':
                    transformed_scores = BiasTransformer.systematic_shift(original_scores, strength)
                elif transform_type == 'systematic_shift_negative':
                    transformed_scores = BiasTransformer.systematic_shift(original_scores, -strength)
                else:
                    transformer_method = getattr(BiasTransformer, transform_type)
                    transformed_scores = transformer_method(original_scores, strength)
                
                # Create histogram
                ax.hist(transformed_scores, bins=20, alpha=0.6, color=colors[j_idx], 
                       label=selected_names[j_idx].replace('-judge', '').title(), density=True)
            
            # Formatting
            ax.set_xlim(-0.5, 4.5)
            ax.set_xlabel('Judge Score')
            ax.set_ylabel('Density')
            
            if t_idx == 0:  # Top row
                ax.set_title(f'Strength = {strength}', fontsize=12, fontweight='bold')
            
            if s_idx == 0:  # Left column
                ax.text(-0.1, 0.5, transform_name, transform=ax.transAxes, 
                       fontsize=14, fontweight='bold', rotation=90, ha='right', va='center')
            
            if t_idx == 0 and s_idx == 0:  # Top-left only
                ax.legend(bbox_to_anchor=(0, 1.15), loc='lower left', ncol=2, fontsize=10)
            
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            if strength > 0:
                mean_change = np.mean([np.mean(transformed_scores) - np.mean(original_scores) 
                                     for original_scores, transformed_scores in 
                                     [(judge_scores[:, j], transformed_scores) 
                                      for j, transformed_scores in 
                                      [(judge_idx, transformer_method(judge_scores[:, judge_idx], 
                                                                     strength if transform_type != 'systematic_shift_negative' 
                                                                     else -strength) 
                                        if transform_type.startswith('systematic_shift') 
                                        else getattr(BiasTransformer, transform_type)(judge_scores[:, judge_idx], strength))
                                       for judge_idx in selected_judges]]])
                
                var_change = np.mean([np.var(transformed_scores) - np.var(original_scores)
                                     for original_scores, transformed_scores in 
                                     [(judge_scores[:, j], transformed_scores) 
                                      for j, transformed_scores in 
                                      [(judge_idx, transformer_method(judge_scores[:, judge_idx], 
                                                                     strength if transform_type != 'systematic_shift_negative' 
                                                                     else -strength) 
                                        if transform_type.startswith('systematic_shift') 
                                        else getattr(BiasTransformer, transform_type)(judge_scores[:, judge_idx], strength))
                                       for judge_idx in selected_judges]]])
                
                # Simplified stats calculation
                sample_original = judge_scores[:, selected_judges[0]]
                if transform_type == 'systematic_shift_positive':
                    sample_transformed = BiasTransformer.systematic_shift(sample_original, strength)
                elif transform_type == 'systematic_shift_negative':
                    sample_transformed = BiasTransformer.systematic_shift(sample_original, -strength)
                else:
                    transformer_method = getattr(BiasTransformer, transform_type)
                    sample_transformed = transformer_method(sample_original, strength)
                
                mean_change = np.mean(sample_transformed) - np.mean(sample_original)
                var_change = np.var(sample_transformed) - np.var(sample_original)
                
                stats_text = f'Δμ: {mean_change:+.2f}\nΔσ²: {var_change:+.2f}'
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                       fontsize=9, ha='right', va='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.suptitle('🔧 Bias Transformation Effects on Judge Score Distributions\n' + 
                 'Appendix Figure: Understanding How Each Transformation Modifies Judge Scores', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    output_dir = Path(__file__).parent / 'results_full_20250818_215910' / 'appendix_figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_dir / 'transformation_effects_appendix.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Transformation effects visualization saved to: {plot_path}")
    return plot_path

def create_before_after_comparison():
    """Create before/after scatter plots for each transformation."""
    
    judge_scores, judge_names = load_sample_judge_scores()
    
    # Use one representative judge for cleaner visualization
    judge_idx = 0  # truthfulness-judge
    original_scores = judge_scores[:, judge_idx]
    
    transformations = {
        'Bottom Heavy (0.8)': ('bottom_heavy', 0.8),
        'Top Heavy (0.8)': ('top_heavy', 0.8),
        'Middle Heavy (0.8)': ('middle_heavy', 0.8),
        'Systematic Shift + (0.6)': ('systematic_shift', 0.6),
        'Systematic Shift - (0.6)': ('systematic_shift', -0.6)
    }
    
    fig, axes = plt.subplots(1, len(transformations), figsize=(20, 4))
    
    for idx, (transform_name, (transform_type, strength)) in enumerate(transformations.items()):
        ax = axes[idx]
        
        # Apply transformation
        if transform_type == 'systematic_shift':
            transformed_scores = BiasTransformer.systematic_shift(original_scores, strength)
        else:
            transformer_method = getattr(BiasTransformer, transform_type)
            transformed_scores = transformer_method(original_scores, strength)
        
        # Create scatter plot
        ax.scatter(original_scores, transformed_scores, alpha=0.6, s=20, color='steelblue')
        
        # Perfect line (no transformation)
        ax.plot([0, 4], [0, 4], 'r--', alpha=0.7, linewidth=2, label='No Change')
        
        # Formatting
        ax.set_xlim(-0.1, 4.1)
        ax.set_ylim(-0.1, 4.1)
        ax.set_xlabel('Original Score')
        ax.set_ylabel('Transformed Score')
        ax.set_title(transform_name)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if idx == 0:
            ax.legend()
        
        # Add correlation
        correlation = np.corrcoef(original_scores, transformed_scores)[0, 1]
        ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes, 
               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.suptitle(f'📊 Before vs After Transformation: {judge_names[judge_idx].replace("-judge", "").title()} Judge\n' +
                 'Appendix Figure: Scatter Plots Showing Score Mapping for Each Transformation', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent / 'results_full_20250818_215910' / 'appendix_figures'
    plot_path = output_dir / 'transformation_scatter_plots_appendix.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Before/after scatter plots saved to: {plot_path}")
    return plot_path

def main():
    """Create transformation visualization figures for appendix."""
    print("🎨 Creating transformation visualization for appendix...")
    
    # Create comprehensive overview
    overview_path = create_transformation_overview()
    
    # Create before/after comparison  
    scatter_path = create_before_after_comparison()
    
    print(f"\n✅ Appendix figures created:")
    print(f"📊 Transformation Overview: {overview_path}")
    print(f"📈 Before/After Scatter: {scatter_path}")
    print(f"\n💡 These figures help readers understand:")
    print(f"   • How each transformation changes score distributions")
    print(f"   • The effect of different bias strengths")
    print(f"   • Why certain transformations break Judge Mean performance")
    print(f"   • The relationship between original and transformed scores")

if __name__ == "__main__":
    main()