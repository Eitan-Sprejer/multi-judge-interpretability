"""
Judge Creation Pipeline

Creates 10 specialized judges via Martian API for evaluating different quality dimensions.
Each judge evaluates on a 0-4 scale with detailed scoring rubrics.
"""

import logging
from typing import Dict, Optional
from martian_apart_hack_sdk import exceptions, judge_specs, martian_client, utils
from martian_apart_hack_sdk.models import llm_models

from utils.judge_rubrics import JUDGE_RUBRICS, JUDGE_DESCRIPTIONS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Judge configuration
JUDGE_MODEL = llm_models.GPT_4O_MINI
MIN_SCORE = 0.0
MAX_SCORE = 4.0

# Judge IDs for easy reference
JUDGE_IDS = list(JUDGE_RUBRICS.keys())


def create_or_update_judge(
    client: martian_client.MartianClient,
    judge_id: str,
    judge_spec: judge_specs.RubricJudgeSpec,
    description: str
) -> Optional[object]:
    """
    Creates a new judge or updates an existing one.
    
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
            logger.info(f"ℹ️ Updated judge {judge_id} to v{existing_judge.version}!")
            return existing_judge
        except Exception as e:
            logger.error(f"❌ Failed to update judge {judge_id}: {e}")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to create judge {judge_id}: {e}")
        return None


def create_all_judges(config_path: Optional[str] = None) -> Dict[str, object]:
    """
    Creates or updates all 10 judges.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Dictionary mapping judge IDs to judge objects
    """
    # Load configuration
    if config_path:
        # TODO: Load from custom config file
        logger.info(f"Loading config from {config_path}")
        config = utils.load_config()  # For now, use default
    else:
        config = utils.load_config()
    
    # Initialize client
    client = martian_client.MartianClient(
        api_url=config.api_url,
        api_key=config.api_key,
    )
    
    logger.info("Starting judge creation/update process...")
    
    judges = {}
    
    for judge_id in JUDGE_IDS:
        # Get rubric and description
        rubric_func = JUDGE_RUBRICS[judge_id]
        rubric = rubric_func()
        description = JUDGE_DESCRIPTIONS[judge_id]
        
        # Create judge specification
        judge_spec = judge_specs.RubricJudgeSpec(
            model_type="rubric_judge",
            rubric=rubric,
            model=JUDGE_MODEL,
            min_score=MIN_SCORE,
            max_score=MAX_SCORE,
        )
        
        # Create or update judge
        judge = create_or_update_judge(client, judge_id, judge_spec, description)
        if judge:
            judges[judge_id] = judge
    
    logger.info(f"Successfully created/updated {len(judges)}/{len(JUDGE_IDS)} judges")
    return judges


def get_judges_from_api(config_path: Optional[str] = None) -> Dict[str, object]:
    """
    Retrieves existing judges from the Martian API.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Dictionary mapping judge IDs to judge objects
    """
    # Load configuration
    if config_path:
        logger.info(f"Loading config from {config_path}")
        config = utils.load_config()  # For now, use default
    else:
        config = utils.load_config()
    
    # Initialize client
    client = martian_client.MartianClient(
        api_url=config.api_url,
        api_key=config.api_key,
    )
    
    judges = {}
    for judge_id in JUDGE_IDS:
        try:
            judge = client.judges.get(judge_id=judge_id)
            judges[judge_id] = judge
            logger.info(f"✅ Retrieved judge {judge_id}")
        except Exception as e:
            logger.error(f"❌ Failed to retrieve judge {judge_id}: {e}")
    
    logger.info(f"Successfully retrieved {len(judges)}/{len(JUDGE_IDS)} judges")
    return judges


def main():
    """Main entry point for judge creation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create or update evaluation judges")
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--list', action='store_true', help='List all judge IDs')
    parser.add_argument('--get', action='store_true', help='Get existing judges from API')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable judges:")
        for i, judge_id in enumerate(JUDGE_IDS, 1):
            print(f"  {i:2}. {judge_id}")
            print(f"      {JUDGE_DESCRIPTIONS[judge_id]}")
    elif args.get:
        judges = get_judges_from_api(args.config)
        print(f"\nRetrieved {len(judges)} judges from API:")
        for judge_id in judges:
            print(f"  - {judge_id}")
    else:
        judges = create_all_judges(args.config)
        print(f"\nCreated/updated {len(judges)} judges:")
        for judge_id in judges:
            print(f"  - {judge_id}")


if __name__ == "__main__":
    main()