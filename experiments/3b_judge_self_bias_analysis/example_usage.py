#!/usr/bin/env python3
"""
Example usage of the created multi-LLM judges.

This script demonstrates how to use the judges created by create_judges.py
to evaluate responses from different models.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the pipeline directory to the path (repo_root/pipeline)
sys.path.append(str(Path(__file__).resolve().parents[2] / "pipeline"))

def example_judge_evaluation():
    """Example of using the created judges to evaluate responses."""
    try:
        from martian_apart_hack_sdk import martian_client
        
        # Load environment variables
        load_dotenv()
        
        # Check for API credentials
        api_key = os.getenv("MARTIAN_API_KEY")
        api_url = os.getenv("MARTIAN_API_URL")
        if not api_key:
            print("❌ MARTIAN_API_KEY environment variable not set")
            return
        if not api_url:
            print("❌ MARTIAN_API_URL environment variable not set")
            return

        # Create client (requires both api_url and api_key)
        client = martian_client.MartianClient(api_url=api_url, api_key=api_key)
        print("✅ Connected to Martian API")
        
        # Example prompt and response
        prompt = "Explain how to make a cup of coffee."
        response = "To make a cup of coffee, you'll need coffee beans, hot water, and a coffee maker. First, grind the beans to your preferred consistency. Then, add the ground coffee to your coffee maker and pour hot water over it. Let it brew for a few minutes, then pour into a cup and enjoy!"
        
        # Example judge IDs (these would be created by create_judges.py)
        example_judge_ids = [
            "openai-harmlessness-judge",
            "anthropic-factual-accuracy-judge", 
            "google-conciseness-redundancy-judge"
        ]
        
        print(f"\n📝 Example Evaluation")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print(f"\n🔍 Evaluating with example judges...")
        
        # Note: This is a demonstration - actual evaluation would require
        # the judges to be created first
        for judge_id in example_judge_ids:
            print(f"\n  Judge: {judge_id}")
            print(f"  Status: Would evaluate response using this judge")
            print(f"  Expected: Score between 0.0-4.0 based on rubric")
        
        print(f"\n💡 To actually run evaluations:")
        print(f"1. First run: python create_judges.py")
        print(f"2. Then use the Martian API to evaluate responses")
        print(f"3. Example API call:")
        print(f"   result = client.judges.evaluate(")
        print(f"       judge_id='openai-harmlessness-judge',")
        print(f"       prompt=prompt,")
        print(f"       response=response")
        print(f"   )")
        
    except Exception as e:
        print(f"❌ Error in example: {e}")


def example_batch_evaluation():
    """Example of batch evaluation using multiple judges."""
    print(f"\n📊 Batch Evaluation Example")
    print(f"With 50 judges, you could evaluate responses like this:")
    
    example_code = '''
# Example batch evaluation
prompts = ["Explain AI", "Write a poem", "Solve math problem"]
responses = ["AI is...", "Roses are red...", "2+2=4"]

# Evaluate each response with all judges
all_scores = {}
for prompt, response in zip(prompts, responses):
    prompt_scores = {}
    for judge_id in all_judge_ids:  # 50 judges
        result = client.judges.evaluate(
            judge_id=judge_id,
            prompt=prompt,
            response=response
        )
        prompt_scores[judge_id] = result.score
    
    all_scores[prompt] = prompt_scores

# Analyze scores across different LLM providers
provider_scores = {}
for provider in ['openai', 'anthropic', 'google', 'together', 'meta']:
    provider_judges = [j for j in all_judge_ids if j.startswith(provider)]
    provider_scores[provider] = {
        'harmlessness': [all_scores[p][f'{provider}-harmlessness-judge'] for p in prompts],
        'factual_accuracy': [all_scores[p][f'{provider}-factual-accuracy-judge'] for p in prompts],
        # ... other metrics
    }
'''
    
    print(example_code)


def main():
    """Main function."""
    print("🚀 Multi-LLM Judge Usage Examples")
    print("=" * 60)
    
    # Show basic usage
    example_judge_evaluation()
    
    # Show batch evaluation
    example_batch_evaluation()
    
    print(f"\n🎯 Key Benefits of Multi-LLM Judges:")
    print(f"• Diversity: 5 different LLM providers")
    print(f"• Specialization: 10 different evaluation criteria")
    print(f"• Consistency: Same rubrics across providers")
    print(f"• Scalability: Easy to add more providers or criteria")
    
    print(f"\n📚 Next Steps:")
    print(f"1. Run: python test_setup.py (to verify setup)")
    print(f"2. Run: python create_judges.py (to create judges)")
    print(f"3. Run: python list_judges.py (to inspect judges)")
    print(f"4. Use the judges in your evaluation pipelines!")


if __name__ == "__main__":
    main()
