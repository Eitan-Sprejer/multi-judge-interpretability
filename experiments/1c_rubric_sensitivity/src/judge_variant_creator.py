"""
Judge Variant Creator

Creates judge variants with different rubric phrasings using the Martian API.
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from martian_apart_hack_sdk import exceptions, judge_specs, martian_client, utils
from martian_apart_hack_sdk.models import llm_models
from pipeline.utils.judge_rubrics import JUDGE_RUBRICS, JUDGE_DESCRIPTIONS
from .rubric_variations import RubricVariationGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Judge configuration (same as original)
JUDGE_MODEL = llm_models.GPT_4O_MINI
MIN_SCORE = 0.0
MAX_SCORE = 4.0


class JudgeVariantCreator:
    """Creates judge variants with different rubric phrasings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the judge variant creator.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        if config_path:
            logger.info(f"Loading config from {config_path}")
            config = utils.load_config()  # For now, use default
        else:
            config = utils.load_config()
        
        # Initialize client
        self.client = martian_client.MartianClient(
            api_url=config.api_url,
            api_key=config.api_key,
        )
        
        # Initialize rubric generator
        self.rubric_generator = RubricVariationGenerator()
        
        # Track created judges
        self.created_judges = {}
    
    def create_judge_variant(
        self,
        base_judge_id: str,
        variant_type: str,
        rubric: str
    ) -> Optional[object]:
        """
        Create a single judge variant.
        
        Args:
            base_judge_id: ID of the original judge
            variant_type: Type of variation (original, formal, casual, restructured)
            rubric: The varied rubric text
            
        Returns:
            Judge object if successful, None otherwise
        """
        # Create unique ID for this variant
        variant_id = f"{base_judge_id}-{variant_type}"
        
        # Get description from original
        description = JUDGE_DESCRIPTIONS.get(
            base_judge_id,
            f"Variant ({variant_type}) of {base_judge_id}"
        )
        description = f"{description} [{variant_type.upper()} variant]"
        
        # Create judge specification
        judge_spec = judge_specs.RubricJudgeSpec(
            model_type="rubric_judge",
            rubric=rubric,
            model=JUDGE_MODEL,
            min_score=MIN_SCORE,
            max_score=MAX_SCORE,
        )
        
        # Create or update judge
        try:
            judge = self.client.judges.create_judge(
                judge_id=variant_id,
                judge_spec=judge_spec,
                description=description
            )
            logger.info(f"✅ Created judge variant: {variant_id}")
            return judge
        except exceptions.ResourceAlreadyExistsError:
            try:
                existing_judge = self.client.judges.get(variant_id)
                self.client.judges.update_judge(
                    judge_id=variant_id,
                    judge_spec=judge_spec
                )
                logger.info(f"ℹ️ Updated judge variant: {variant_id}")
                return existing_judge
            except Exception as e:
                logger.error(f"❌ Failed to update judge {variant_id}: {e}")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to create judge {variant_id}: {e}")
            return None
    
    def create_all_variants_for_judge(
        self,
        judge_id: str,
        variation_types: List[str] = ['original', 'formal', 'casual', 'restructured']
    ) -> Dict[str, object]:
        """
        Create all variants for a single judge.
        
        Args:
            judge_id: ID of the judge to create variants for
            variation_types: Types of variations to create
            
        Returns:
            Dictionary mapping variant IDs to judge objects
        """
        if judge_id not in JUDGE_RUBRICS:
            logger.error(f"Unknown judge ID: {judge_id}")
            return {}
        
        # Get original rubric
        rubric_func = JUDGE_RUBRICS[judge_id]
        original_rubric = rubric_func()
        
        # Generate variations
        variations = self.rubric_generator.generate_variations(original_rubric, judge_id)
        
        # Create judges for each variation
        variants = {}
        for variant_type in variation_types:
            if variant_type not in variations:
                logger.warning(f"Variation type {variant_type} not available for {judge_id}")
                continue
            
            rubric = variations[variant_type]
            judge = self.create_judge_variant(judge_id, variant_type, rubric)
            
            if judge:
                variant_id = f"{judge_id}-{variant_type}"
                variants[variant_id] = judge
                self.created_judges[variant_id] = judge
        
        logger.info(f"Created {len(variants)}/{len(variation_types)} variants for {judge_id}")
        return variants
    
    def create_all_judge_variants(
        self,
        judge_ids: Optional[List[str]] = None,
        variation_types: List[str] = ['original', 'formal', 'casual', 'restructured']
    ) -> Dict[str, object]:
        """
        Create variants for all specified judges.
        
        Args:
            judge_ids: List of judge IDs to create variants for (None = all judges)
            variation_types: Types of variations to create
            
        Returns:
            Dictionary mapping all variant IDs to judge objects
        """
        if judge_ids is None:
            judge_ids = list(JUDGE_RUBRICS.keys())
        
        logger.info(f"Creating {len(variation_types)} variants for {len(judge_ids)} judges...")
        
        all_variants = {}
        for judge_id in judge_ids:
            variants = self.create_all_variants_for_judge(judge_id, variation_types)
            all_variants.update(variants)
        
        total_expected = len(judge_ids) * len(variation_types)
        logger.info(f"Successfully created {len(all_variants)}/{total_expected} judge variants")
        
        return all_variants
    
    def get_variant_id(self, base_judge_id: str, variant_type: str) -> str:
        """Get the ID for a judge variant."""
        return f"{base_judge_id}-{variant_type}"
    
    def get_all_variant_ids(
        self,
        judge_ids: Optional[List[str]] = None,
        variation_types: List[str] = ['original', 'formal', 'casual', 'restructured']
    ) -> List[str]:
        """Get all variant IDs that would be created."""
        if judge_ids is None:
            judge_ids = list(JUDGE_RUBRICS.keys())
        
        variant_ids = []
        for judge_id in judge_ids:
            for variant_type in variation_types:
                variant_ids.append(self.get_variant_id(judge_id, variant_type))
        
        return variant_ids
    
    def cleanup_variants(self):
        """Delete all created judge variants (for cleanup)."""
        logger.info("Cleaning up created judge variants...")
        
        deleted = 0
        for variant_id in self.created_judges:
            try:
                # Skip original variants (don't delete the base judges)
                if variant_id.endswith('-original'):
                    continue
                    
                self.client.judges.delete(variant_id)
                logger.info(f"🗑️ Deleted judge variant: {variant_id}")
                deleted += 1
            except Exception as e:
                logger.error(f"Failed to delete {variant_id}: {e}")
        
        logger.info(f"Deleted {deleted} judge variants")


def main():
    """Main entry point for testing judge variant creation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create judge variants for rubric sensitivity testing")
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--judges', nargs='+', help='Specific judge IDs to create variants for')
    parser.add_argument('--types', nargs='+', 
                       default=['original', 'formal', 'casual', 'restructured'],
                       help='Variation types to create')
    parser.add_argument('--cleanup', action='store_true', 
                       help='Delete all created variants')
    parser.add_argument('--test', action='store_true',
                       help='Test with just one judge')
    
    args = parser.parse_args()
    
    creator = JudgeVariantCreator(args.config)
    
    if args.cleanup:
        creator.cleanup_variants()
    elif args.test:
        # Test with just harmlessness judge
        test_judge = 'harmlessness-judge'
        logger.info(f"Testing with {test_judge}...")
        variants = creator.create_all_variants_for_judge(test_judge, args.types)
        print(f"\nCreated {len(variants)} variants:")
        for variant_id in variants:
            print(f"  - {variant_id}")
    else:
        # Create all requested variants
        judge_ids = args.judges if args.judges else None
        variants = creator.create_all_judge_variants(judge_ids, args.types)
        
        print(f"\nCreated {len(variants)} judge variants:")
        for variant_id in sorted(variants.keys()):
            print(f"  - {variant_id}")


if __name__ == "__main__":
    main()