#!/usr/bin/env python3
"""
Script to list and inspect created judges.

This script connects to the Martian API and lists all available judges,
with special focus on the multi-LLM judges created by this experiment.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add the pipeline directory to the path
sys.path.append(str(Path(__file__).parent.parent / "pipeline"))

def list_all_judges():
    """List all available judges from the Martian API."""
    try:
        from martian_apart_hack_sdk import martian_client
        
        # Load environment variables
        load_dotenv()
        
        # Check for API key
        api_key = os.getenv("MARTIAN_API_KEY")
        if not api_key:
            print("❌ MARTIAN_API_KEY environment variable not set")
            return
        
        # Create client
        client = martian_client.MartianClient(api_url="https://withmartian.com/api", api_key=api_key)
        print("✅ Connected to Martian API")
        
        # Get all judges
        judges = client.judges.list()
        print(f"\n📋 Found {len(judges)} total judges")
        
        # Categorize judges
        multi_llm_judges = []
        other_judges = []
        
        for judge in judges:
            judge_id = judge.id
            if any(provider in judge_id for provider in ['openai', 'anthropic', 'google', 'together', 'meta']):
                multi_llm_judges.append(judge)
            else:
                other_judges.append(judge)
        
        # Display multi-LLM judges
        if multi_llm_judges:
            print(f"\n🎯 Multi-LLM Judges ({len(multi_llm_judges)} found):")
            print("-" * 60)
            
            # Group by provider
            providers = {}
            for judge in multi_llm_judges:
                judge_id = judge.id
                provider = judge_id.split('-')[0]
                if provider not in providers:
                    providers[provider] = []
                providers[provider].append(judge)
            
            for provider, provider_judges in sorted(providers.items()):
                print(f"\n{provider.upper()} ({len(provider_judges)} judges):")
                for judge in sorted(provider_judges, key=lambda x: x.id):
                    print(f"  • {judge.id}")
                    if hasattr(judge, 'description') and judge.description:
                        print(f"    {judge.description}")
        else:
            print("\n⚠️ No multi-LLM judges found. Run create_judges.py first.")
        
        # Display other judges
        if other_judges:
            print(f"\n🔍 Other Judges ({len(other_judges)} found):")
            print("-" * 60)
            for judge in sorted(other_judges, key=lambda x: x.id):
                print(f"  • {judge.id}")
                if hasattr(judge, 'description') and judge.description:
                    print(f"    {judge.description}")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"  Total judges: {len(judges)}")
        print(f"  Multi-LLM judges: {len(multi_llm_judges)}")
        print(f"  Other judges: {len(other_judges)}")
        
        # Check if we have the expected 50 judges
        expected_count = 50
        if len(multi_llm_judges) == expected_count:
            print(f"🎉 All {expected_count} expected judges are present!")
        elif len(multi_llm_judges) > 0:
            print(f"⚠️ Found {len(multi_llm_judges)} multi-LLM judges, expected {expected_count}")
        else:
            print(f"❌ No multi-LLM judges found. Expected {expected_count}")
        
        return judges
        
    except Exception as e:
        print(f"❌ Error listing judges: {e}")
        return None


def get_judge_details(judge_id: str):
    """Get detailed information about a specific judge."""
    try:
        from martian_apart_hack_sdk import martian_client
        
        # Load environment variables
        load_dotenv()
        
        # Check for API key
        api_key = os.getenv("MARTIAN_API_KEY")
        if not api_key:
            print("❌ MARTIAN_API_KEY environment variable not set")
            return
        
        # Create client
        client = martian_client.MartianClient(api_url="https://withmartian.com/api", api_key=api_key)
        
        # Get judge details
        judge = client.judges.get(judge_id)
        
        print(f"\n🔍 Details for judge: {judge_id}")
        print("-" * 60)
        print(f"Judge ID: {judge.id}")
        print(f"Version: {judge.version}")
        print(f"Created: {judge.createTime}")
        print(f"Updated: {judge.createTime}")
        
        if hasattr(judge, 'description') and judge.description:
            print(f"Description: {judge.description}")
        
        if hasattr(judge, 'llm_model') and judge.llm_model:
            print(f"LLM Model: {judge.llm_model}")
        
        if hasattr(judge, 'min_score') and judge.min_score is not None:
            print(f"Min Score: {judge.min_score}")
        
        if hasattr(judge, 'max_score') and judge.max_score is not None:
            print(f"Max Score: {judge.max_score}")
        
        return judge
        
    except Exception as e:
        print(f"❌ Error getting judge details: {e}")
        return None


def main():
    """Main function."""
    print("🔍 Multi-LLM Judge Inspector")
    print("=" * 50)
    
    # List all judges
    judges = list_all_judges()
    
    if not judges:
        print("\n❌ Failed to retrieve judges")
        return 1
    
    # Interactive mode
    print("\n" + "=" * 50)
    print("Interactive Mode")
    print("=" * 50)
    print("Enter a judge ID to see details (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\nJudge ID (or 'quit'): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Try to get judge details
            judge = get_judge_details(user_input)
            if not judge:
                print(f"❌ Judge '{user_input}' not found or error occurred")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
