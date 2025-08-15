#!/usr/bin/env python3
"""
Multi-LLM Judge Creation Script

Creates 50 specialized judges using 5 different LLM providers.
Each provider creates 10 judges based on the available rubrics.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Add the pipeline directory to the path
sys.path.append(str(Path(__file__).parent.parent / "pipeline"))

try:
    from martian_apart_hack_sdk import exceptions, judge_specs, martian_client, utils
    from martian_apart_hack_sdk.models import llm_models
    
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
    
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you have the martian-apart-hack-sdk installed and the pipeline directory is accessible")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('judge_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# LLM Provider Configuration
LLM_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "model": llm_models.GPT_4O_MINI,
        "description": "GPT-4o-mini based judge"
    },
    "anthropic": {
        "name": "Anthropic", 
        "model": llm_models.CLAUDE_3_5_SONNET,
        "description": "Claude-3.5-sonnet based judge"
    },
    "google": {
        "name": "Google",
        "model": llm_models.GEMINI_1_5_FLASH,
        "description": "Gemini-1.5-flash based judge"
    },
    "together": {
        "name": "Together",
        "model": llm_models.LLAMA_3_1_405B,
        "description": "Llama-3.1-405B based judge"
    },
    "meta": {
        "name": "Meta",
        "model": llm_models.LLAMA_3_3_70B,
        "description": "Llama-3.3-70B based judge"
    }
}

# Judge configuration
MIN_SCORE = 0.0
MAX_SCORE = 4.0


def create_judge_spec(
    rubric: str,
    llm_model: str,
    min_score: float = MIN_SCORE,
    max_score: float = MAX_SCORE
) -> judge_specs.RubricJudgeSpec:
    """
    Create a judge specification with the given rubric and LLM model.
    
    Args:
        rubric: The scoring rubric text
        llm_model: The LLM model to use for the judge
        min_score: Minimum score value
        max_score: Maximum score value
        
    Returns:
        Configured judge specification
    """
    return judge_specs.RubricJudgeSpec(
        model_type="rubric_judge",
        rubric=rubric,
        model=llm_model,
        min_score=min_score,
        max_score=max_score
    )


def create_or_update_judge(
    client: martian_client.MartianClient,
    judge_id: str,
    judge_spec: judge_specs.RubricJudgeSpec,
    description: str
) -> Optional[object]:
    """
    Create a new judge or update an existing one.
    
    Args:
        client: Martian API client
        judge_id: Unique identifier for the judge
        judge_spec: Judge specification with rubric and configuration
        description: Human-readable description of the judge
        
    Returns:
        Judge object if successful, None otherwise
    """
    try:
        judge = client.judges.create_judge(
            judge_id=judge_id,
            judge_spec=judge_spec,
            description=description
        )
        logger.info(f"✅ Created judge {judge_id}")
        return judge
    except exceptions.ResourceAlreadyExistsError:
        try:
            existing_judge = client.judges.get(judge_id)
            client.judges.update_judge(judge_id=judge_id, judge_spec=judge_spec)
            logger.info(f"ℹ️ Updated judge {judge_id} to v{existing_judge.version}")
            return existing_judge
        except Exception as e:
            logger.error(f"❌ Failed to update judge {judge_id}: {e}")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to create judge {judge_id}: {e}")
        return None


def create_provider_judges(
    client: martian_client.MartianClient,
    provider_key: str,
    provider_config: Dict
) -> Dict[str, object]:
    """
    Create all 10 judges for a specific LLM provider.
    
    Args:
        client: Martian API client
        provider_key: Key identifier for the provider
        provider_config: Configuration for the provider
        
    Returns:
        Dictionary mapping judge IDs to judge objects
    """
    logger.info(f"Creating judges for {provider_config['name']} ({provider_key})")
    
    judges = {}
    provider_name = provider_config['name'].lower()
    
    for rubric_key, rubric_func in JUDGE_RUBRICS.items():
        # Create unique judge ID: provider-rubric (e.g., openai-harmlessness-judge)
        judge_id = f"{provider_key}-{rubric_key}"
        
        # Get the rubric text
        rubric_text = rubric_func()
        
        # Create judge specification
        judge_spec = create_judge_spec(
            rubric=rubric_text,
            llm_model=provider_config['model']
        )
        
        # Create description
        description = f"{provider_config['description']} for {JUDGE_DESCRIPTIONS[rubric_key]}"
        
        # Create or update the judge
        judge = create_or_update_judge(client, judge_id, judge_spec, description)
        
        if judge:
            judges[judge_id] = judge
        else:
            logger.error(f"Failed to create judge {judge_id}")
    
    logger.info(f"Created {len(judges)}/10 judges for {provider_config['name']}")
    return judges


def main():
    """Main function to create all judges."""
    logger.info("Starting multi-LLM judge creation")
    
    # Check for API key
    api_key = os.getenv("MARTIAN_API_KEY")
    if not api_key:
        logger.error("MARTIAN_API_KEY environment variable not set")
        sys.exit(1)
    
    try:
        # Initialize Martian client
        client = martian_client.MartianClient(api_url="https://withmartian.com/api", api_key=api_key)
        logger.info("✅ Connected to Martian API")
        
        # Create judges for each provider
        all_judges = {}
        total_created = 0
        
        for provider_key, provider_config in LLM_PROVIDERS.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {provider_config['name']}")
            logger.info(f"{'='*50}")
            
            provider_judges = create_provider_judges(client, provider_key, provider_config)
            all_judges.update(provider_judges)
            total_created += len(provider_judges)
            
            logger.info(f"Progress: {total_created}/50 judges created")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("JUDGE CREATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total judges created/updated: {total_created}/50")
        
        if total_created == 50:
            logger.info("🎉 All judges successfully created!")
        else:
            logger.warning(f"⚠️ Only {total_created}/50 judges were created. Check logs for errors.")
        
        # List all created judges
        logger.info("\nCreated judges:")
        for judge_id in sorted(all_judges.keys()):
            logger.info(f"  - {judge_id}")
            
    except Exception as e:
        logger.error(f"Fatal error during judge creation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
