#!/usr/bin/env python3
"""
Experiment 1A: Judge Contamination

Tests how well the aggregator performs when some judges are deliberately flawed.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.inverted_judge_creator import InvertedJudgeCreator
from src.contamination_experiment import ContaminationExperiment


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 1A: Judge Contamination")
    parser.add_argument("--create-judges", action="store_true", help="Create contaminated judges")
    parser.add_argument("--run-experiment", action="store_true", help="Run the contamination experiment")
    parser.add_argument("--quick", action="store_true", help="Run in quick test mode")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--n-contaminated", type=int, default=1, 
                        help="Number of contaminated judges to include (1, 2, or 3)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("EXPERIMENT 1A: JUDGE CONTAMINATION")
    print("="*60)
    
    if args.create_judges:
        print("\nCreating contaminated judges...")
        creator = InvertedJudgeCreator(
            output_dir=output_dir,
            config_path=args.config
        )
        
        if args.n_contaminated > 0:
            judges = creator.create_experiment_1a_judges(args.n_contaminated)
            print(f"✅ Created {len(judges)} contaminated judges for Experiment 1A")
        else:
            judges = creator.create_all_contaminated_judges()
            print(f"✅ Created {len(judges)} contaminated judges")
        
    elif args.run_experiment:
        print(f"\nRunning contamination experiment with {args.n_contaminated} contaminated judges...")
        
        # Step 1: Ensure contaminated judges exist
        print("Step 1: Checking for contaminated judges...")
        creator = InvertedJudgeCreator(
            output_dir=output_dir,
            config_path=args.config
        )
        
        # Create judges if they don't exist
        existing_judges = len(creator.created_judges)
        if existing_judges < args.n_contaminated:
            print(f"Creating {args.n_contaminated - existing_judges} additional contaminated judges...")
            creator.create_experiment_1a_judges(args.n_contaminated)
        
        # Step 2: Run the experiment
        print("Step 2: Running contamination experiment...")
        experiment = ContaminationExperiment(
            output_dir=output_dir,
            config_path=args.config,
            quick_mode=args.quick
        )
        
        import asyncio
        async def run():
            return await experiment.run_contamination_experiment(args.n_contaminated)
        
        results = asyncio.run(run())
        
        # Step 3: Print results
        print("\n" + "="*50)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*50)
        
        if 'key_metrics' in results:
            metrics = results['key_metrics']
            print(f"Performance Degradation: {metrics['performance_degradation']:.2%}")
            print(f"Contamination Detection Score: {metrics['contamination_detection_score']:.2%}")
            print(f"Robustness Score: {metrics['robustness_score']:.2%}")
            
            success = metrics['success_criteria']
            print(f"\nSuccess Criteria:")
            print(f"  Degradation < 10%: {'✅' if success['degradation_below_10_percent'] else '❌'}")
            print(f"  Detection > 70%: {'✅' if success['detection_above_70_percent'] else '❌'}")
            print(f"  Robustness > 80%: {'✅' if success['robustness_above_80_percent'] else '❌'}")
        
        print(f"\nDetailed results saved to: {output_dir}")
        
    else:
        print("\nRunning complete Experiment 1A pipeline...")
        
        # Step 1: Create contaminated judges
        print("Step 1: Creating contaminated judges...")
        creator = InvertedJudgeCreator(
            output_dir=output_dir,
            config_path=args.config
        )
        judges = creator.create_experiment_1a_judges(args.n_contaminated)
        print(f"✅ Created {len(judges)} contaminated judges")
        
        # Step 2: Run the experiment
        print(f"\nStep 2: Running contamination experiment...")
        experiment = ContaminationExperiment(
            output_dir=output_dir,
            config_path=args.config,
            quick_mode=args.quick
        )
        
        import asyncio
        async def run():
            return await experiment.run_contamination_experiment(args.n_contaminated)
        
        results = asyncio.run(run())
        print(f"✅ Experiment completed")
        
        # Print summary
        print("\n" + "="*50)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*50)
        
        if 'key_metrics' in results:
            metrics = results['key_metrics']
            print(f"Performance Degradation: {metrics['performance_degradation']:.2%}")
            print(f"Contamination Detection Score: {metrics['contamination_detection_score']:.2%}")
            print(f"Robustness Score: {metrics['robustness_score']:.2%}")
            
            success = metrics['success_criteria']
            print(f"\nSuccess Criteria:")
            print(f"  Degradation < 10%: {'✅' if success['degradation_below_10_percent'] else '❌'}")
            print(f"  Degradation < 10%: {'✅' if success['degradation_below_10_percent'] else '❌'}")
            print(f"  Detection > 70%: {'✅' if success['detection_above_70_percent'] else '❌'}")
            print(f"  Robustness > 80%: {'✅' if success['robustness_above_80_percent'] else '❌'}")
        
        print(f"\nDetailed results saved to: {output_dir}")


if __name__ == "__main__":
    main()
