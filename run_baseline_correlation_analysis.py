#!/usr/bin/env python3
"""
Standalone script to run cross-correlation analysis on existing baseline experiment results.

This script can be used to add correlation heatmaps to any completed baseline experiment.
It integrates the correlation analysis into the existing results structure.
"""

import argparse
import sys
from pathlib import Path
from correlation_analysis import CorrelationAnalyzer


def main():
    """Run correlation analysis on a baseline experiment."""
    parser = argparse.ArgumentParser(
        description="Add cross-correlation analysis to baseline experiment results"
    )
    parser.add_argument(
        'results_dir', 
        help='Path to baseline experiment results directory'
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Overwrite existing correlation analysis files'
    )
    
    args = parser.parse_args()
    
    # Validate results directory
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Check for required data files
    data_file = results_dir / "data" / "data_with_judge_scores.pkl"
    if not data_file.exists():
        print(f"❌ Judge scores data not found: {data_file}")
        print("   This script requires a completed baseline experiment with judge scores.")
        sys.exit(1)
    
    # Check if analysis already exists
    existing_analysis = results_dir / "results" / "cross_correlation_analysis.json"
    existing_plot = results_dir / "plots" / "cross_correlation_heatmaps.png"
    
    if (existing_analysis.exists() or existing_plot.exists()) and not args.force:
        print("⚠️  Cross-correlation analysis files already exist.")
        print("   Use --force to overwrite existing files.")
        print(f"   Analysis: {existing_analysis.exists()}")
        print(f"   Plot: {existing_plot.exists()}")
        sys.exit(1)
    
    # Run the analysis
    print(f"🔍 Running cross-correlation analysis on: {results_dir.name}")
    print(f"📁 Results directory: {results_dir}")
    
    try:
        analyzer = CorrelationAnalyzer(results_dir)
        results = analyzer.run_correlation_analysis()
        
        print("\n" + "="*60)
        print("✅ CROSS-CORRELATION ANALYSIS COMPLETE")
        print("="*60)
        print(f"📊 Analyzed {len(results['judge_names'])} judges and {len(results['persona_names'])} personas")
        print(f"📈 Judge matrix shape: {results['judge_matrix'].shape}")
        print(f"📈 Persona matrix shape: {results['persona_matrix'].shape}")
        print(f"🎨 Heatmaps saved to: plots/cross_correlation_heatmaps.png")
        print(f"📄 Analysis saved to: results/cross_correlation_analysis.json")
        
        # Create insights report
        insights_file = results_dir / "cross_correlation_insights.md"
        if not insights_file.exists() or args.force:
            create_insights_report(results_dir, results)
            print(f"📝 Insights report: cross_correlation_insights.md")
        
        print("\n💡 Key Insights:")
        
        # Quick summary
        judge_corr_mean = analyzer._calculate_summary_stats(
            results['judge_corr_matrix'], "Judge-Judge"
        )['mean']
        persona_corr_mean = analyzer._calculate_summary_stats(
            results['persona_corr_matrix'], "Persona-Persona"  
        )['mean']
        cross_corr_mean = analyzer._calculate_summary_stats(
            results['judge_persona_corr_matrix'], "Judge-Persona"
        )['mean']
        
        print(f"   📊 Judge-Judge correlations: {judge_corr_mean:.3f} (mean)")
        print(f"   👥 Persona-Persona correlations: {persona_corr_mean:.3f} (mean)")
        print(f"   🔗 Judge-Persona correlations: {cross_corr_mean:.3f} (mean)")
        
        if judge_corr_mean > 0.6:
            print("   ✅ Strong judge alignment - good evaluation consistency")
        if persona_corr_mean > 0.7:
            print("   ✅ Strong persona alignment - human preferences converge")
        if cross_corr_mean > 0.6:
            print("   ✅ Strong judge-persona alignment - judges match human preferences")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


def create_insights_report(results_dir: Path, results: dict):
    """Create a basic insights report."""
    insights_content = f"""# Cross-Correlation Analysis Results

## Summary

- **Judges analyzed**: {len(results['judge_names'])}
- **Personas analyzed**: {len(results['persona_names'])}
- **Total samples**: {results['judge_matrix'].shape[0]}

## Files Generated

- `plots/cross_correlation_heatmaps.png` - Comprehensive correlation heatmaps
- `results/cross_correlation_analysis.json` - Detailed correlation matrices and statistics

## Judge Names
{', '.join(results['judge_names'])}

## Persona Names  
{', '.join(results['persona_names'])}

For detailed insights and interpretation, see the main cross-correlation analysis documentation.
"""

    with open(results_dir / "cross_correlation_insights.md", 'w') as f:
        f.write(insights_content)


if __name__ == "__main__":
    main()