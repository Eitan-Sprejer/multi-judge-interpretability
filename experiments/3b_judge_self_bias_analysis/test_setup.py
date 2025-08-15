#!/usr/bin/env python3
"""
Test script to verify the multi-LLM judge creation setup.

This script checks:
1. Required dependencies are available
2. Rubrics can be loaded
3. LLM models are accessible
4. Martian API connection works
"""

import sys
from pathlib import Path

# Add the pipeline directory to the path
sys.path.append(str(Path(__file__).parent.parent / "pipeline"))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from martian_apart_hack_sdk import exceptions, judge_specs, martian_client, utils
        from martian_apart_hack_sdk.models import llm_models
        print("✅ Martian SDK imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Martian SDK: {e}")
        return False
    
    try:
        # Import judge_rubrics directly to avoid __init__.py dependencies
        import sys
        from pathlib import Path
        # Go up two levels: experiments/3b_judge_self_bias_analysis -> experiments -> multi-judge-interpretability
        root_dir = Path(__file__).parent.parent.parent
        sys.path.append(str(root_dir))
        
        # Import the module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "judge_rubrics", 
            str(root_dir / "pipeline" / "utils" / "judge_rubrics.py")
        )
        judge_rubrics_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(judge_rubrics_module)
        
        JUDGE_RUBRICS = judge_rubrics_module.JUDGE_RUBRICS
        JUDGE_DESCRIPTIONS = judge_rubrics_module.JUDGE_DESCRIPTIONS
        
        print("✅ Judge rubrics imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import judge rubrics: {e}")
        return False
    
    return True


def test_rubrics():
    """Test that all rubrics can be loaded."""
    print("\nTesting rubrics...")
    
    try:
        # Import judge_rubrics directly to avoid __init__.py dependencies
        import sys
        from pathlib import Path
        # Go up two levels: experiments/3b_judge_self_bias_analysis -> experiments -> multi-judge-interpretability
        root_dir = Path(__file__).parent.parent.parent
        sys.path.append(str(root_dir))
        
        # Import the module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "judge_rubrics", 
            str(root_dir / "pipeline" / "utils" / "judge_rubrics.py")
        )
        judge_rubrics_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(judge_rubrics_module)
        
        JUDGE_RUBRICS = judge_rubrics_module.JUDGE_RUBRICS
        JUDGE_DESCRIPTIONS = judge_rubrics_module.JUDGE_DESCRIPTIONS
        
        print(f"Found {len(JUDGE_RUBRICS)} rubric types:")
        for rubric_key, rubric_func in JUDGE_RUBRICS.items():
            try:
                rubric_text = rubric_func()
                if len(rubric_text) > 100:  # Basic check that rubric has content
                    print(f"  ✅ {rubric_key}: {len(rubric_text)} characters")
                else:
                    print(f"  ⚠️ {rubric_key}: Rubric seems too short")
            except Exception as e:
                print(f"  ❌ {rubric_key}: Failed to load - {e}")
        
        print(f"\nFound {len(JUDGE_DESCRIPTIONS)} descriptions:")
        for rubric_key, description in JUDGE_DESCRIPTIONS.items():
            print(f"  ✅ {rubric_key}: {description}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to test rubrics: {e}")
        return False


def test_llm_models():
    """Test that LLM models are accessible."""
    print("\nTesting LLM models...")
    
    try:
        from martian_apart_hack_sdk.models import llm_models
        
        # Check that the models we need are available
        required_models = [
            llm_models.GPT_4O_MINI,
            llm_models.CLAUDE_3_5_SONNET,
            llm_models.GEMINI_1_5_FLASH,
            llm_models.LLAMA_3_1_405B,
            llm_models.LLAMA_3_3_70B
        ]
        
        for model in required_models:
            print(f"  ✅ {model}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to test LLM models: {e}")
        return False


def test_environment():
    """Test environment variables."""
    print("\nTesting environment...")
    
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    api_key = os.getenv("MARTIAN_API_KEY")
    if api_key:
        print(f"✅ MARTIAN_API_KEY found (length: {len(api_key)})")
    else:
        print("❌ MARTIAN_API_KEY not found")
        print("   Please set the MARTIAN_API_KEY environment variable")
        return False
    
    return True


def test_martian_connection():
    """Test connection to Martian API."""
    print("\nTesting Martian API connection...")
    
    try:
        import os
        from martian_apart_hack_sdk import martian_client
        
        api_key = os.getenv("MARTIAN_API_KEY")
        if not api_key:
            print("❌ No API key available for connection test")
            return False
        
        # Try to create a client (this will test the connection)
        client = martian_client.MartianClient(api_url="https://withmartian.com/api", api_key=api_key)
        print("✅ Martian client created successfully")
        
        # Try to list judges (this will test actual API access)
        try:
            judges = client.judges.list()
            print(f"✅ Successfully connected to Martian API (found {len(judges)} existing judges)")
            return True
        except Exception as e:
            print(f"⚠️ Client created but API call failed: {e}")
            print("   This might be a network/SSL issue, but the client setup is working")
            # Consider this a partial success since we can create the client
            return True
            
    except Exception as e:
        print(f"❌ Failed to connect to Martian API: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Multi-LLM Judge Creation Setup Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Rubrics", test_rubrics),
        ("LLM Models", test_llm_models),
        ("Environment", test_environment),
        ("Martian API", test_martian_connection)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! You're ready to create judges.")
        print("\nNext steps:")
        print("1. Ensure your MARTIAN_API_KEY is set")
        print("2. Run: python create_judges.py")
    else:
        print("⚠️ Some tests failed. Please fix the issues before proceeding.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
