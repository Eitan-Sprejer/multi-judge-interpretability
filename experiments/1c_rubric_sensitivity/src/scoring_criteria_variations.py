"""
Scoring Criteria Variation Generator

Creates variations in judge scoring criteria - both interval scaling and strictness levels.
Separate from prompting style variations to test fundamental robustness to evaluation changes.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ScoreInterval:
    """Represents a scoring interval with range and description."""
    min_score: float
    max_score: float
    label: str
    description: str
    
    @property
    def range_str(self) -> str:
        """Return formatted range string."""
        if self.max_score == self.min_score:
            return f"{self.min_score}"
        return f"{self.min_score:.1f}–{self.max_score:.1f}"


class ScoringCriteriaVariationGenerator:
    """Generates scoring criteria variations for testing rubric robustness."""
    
    def __init__(self):
        """Initialize the criteria variation generator."""
        self.variation_strategies = {
            # Interval scaling variations
            'linear': self._apply_linear_scaling,
            'bottom_heavy': self._apply_bottom_heavy_scaling,
            'top_heavy': self._apply_top_heavy_scaling,
            
            # Criteria strictness variations
            'standard': self._apply_standard_strictness,
            'strict': self._apply_strict_criteria,
            'lenient': self._apply_lenient_criteria,
        }
    
    def generate_variations(self, original_rubric: str, judge_name: str) -> Dict[str, str]:
        """
        Generate all scoring criteria variations of a rubric.
        
        Args:
            original_rubric: The original rubric text
            judge_name: Name of the judge for context-aware variations
            
        Returns:
            Dictionary with variation types as keys and rubric text as values
        """
        variations = {'original': original_rubric}
        
        for variation_type, strategy in self.variation_strategies.items():
            if variation_type != 'standard':  # Skip standard (same as original)
                try:
                    variations[variation_type] = strategy(original_rubric, judge_name)
                except Exception as e:
                    print(f"Warning: Failed to generate {variation_type} variation for {judge_name}: {e}")
                    variations[variation_type] = original_rubric  # Fallback to original
        
        return variations
    
    def _extract_scoring_intervals(self, rubric: str) -> List[ScoreInterval]:
        """Extract scoring intervals from rubric text."""
        intervals = []
        
        # Pattern to match score ranges like "0.0–0.9 = LABEL" or "4.0 = LABEL"
        pattern = r'([\d\.]+)(?:–|-)?([\d\.]+)?\s*=\s*([A-Z\s\(\)]+)'
        matches = re.findall(pattern, rubric)
        
        for match in matches:
            min_score = float(match[0])
            max_score = float(match[1]) if match[1] else min_score
            label = match[2].strip()
            
            # Extract description (everything until next score range or section)
            description = self._extract_interval_description(rubric, min_score, max_score, label)
            
            intervals.append(ScoreInterval(min_score, max_score, label, description))
        
        return intervals
    
    def _extract_interval_description(self, rubric: str, min_score: float, max_score: float, label: str) -> str:
        """Extract the description for a specific scoring interval."""
        # Find the line with this score range
        if max_score == min_score:
            pattern = f"{min_score}\\s*=\\s*{re.escape(label)}"
        else:
            pattern = f"{min_score:.1f}–{max_score:.1f}\\s*=\\s*{re.escape(label)}"
        
        start_match = re.search(pattern, rubric)
        if not start_match:
            return ""
        
        start_pos = start_match.end()
        
        # Find the next score range or section boundary
        next_score_pattern = r'\n[\d\.]+(?:–[\d\.]+)?\s*='
        next_section_pattern = r'\n[A-Z][a-z]+ [A-Z][a-z]+\n[-=]+'
        
        next_score = re.search(next_score_pattern, rubric[start_pos:])
        next_section = re.search(next_section_pattern, rubric[start_pos:])
        
        # Find the earliest boundary
        end_pos = len(rubric)
        if next_score:
            end_pos = min(end_pos, start_pos + next_score.start())
        if next_section:
            end_pos = min(end_pos, start_pos + next_section.start())
        
        description = rubric[start_pos:end_pos].strip()
        return description
    
    def _rebuild_rubric_with_intervals(self, rubric: str, new_intervals: List[ScoreInterval]) -> str:
        """Rebuild rubric with modified scoring intervals."""
        # Find the scoring criteria section
        criteria_pattern = r'(Scoring Criteria\n[-=]+\n)(.*?)(\n[A-Z][a-z]+ [A-Z][a-z]+\n[-=]+|$)'
        match = re.search(criteria_pattern, rubric, re.DOTALL)
        
        if not match:
            return rubric  # Fallback if pattern not found
        
        header = match.group(1)
        footer = match.group(3) if match.group(3) else ""
        
        # Rebuild criteria section
        new_criteria = []
        for interval in new_intervals:
            new_criteria.append(f"{interval.range_str} = {interval.label}")
            if interval.description:
                new_criteria.append(interval.description)
            new_criteria.append("")  # Add spacing
        
        new_criteria_text = "\n".join(new_criteria).rstrip()
        
        # Replace the criteria section
        before_criteria = rubric[:match.start(1)]
        after_criteria = rubric[match.end(2):]
        
        return before_criteria + header + new_criteria_text + after_criteria
    
    # Interval Scaling Variations
    def _apply_linear_scaling(self, rubric: str, judge_name: str) -> str:
        """Apply linear scaling (original intervals) - no change."""
        return rubric
    
    def _apply_bottom_heavy_scaling(self, rubric: str, judge_name: str) -> str:
        """Apply bottom-heavy scaling - compress low scores, expand high scores."""
        intervals = self._extract_scoring_intervals(rubric)
        if len(intervals) != 5:
            return rubric  # Only works with standard 5-level rubrics
        
        # New intervals: compress bottom, expand top
        new_ranges = [
            (0.0, 0.5),   # Was 0.0-0.9
            (0.6, 1.2),   # Was 1.0-1.9  
            (1.3, 2.5),   # Was 2.0-2.9
            (2.6, 3.5),   # Was 3.0-3.9
            (3.6, 4.0)    # Was 4.0
        ]
        
        new_intervals = []
        for i, (min_score, max_score) in enumerate(new_ranges):
            if i < len(intervals):
                new_intervals.append(ScoreInterval(
                    min_score, max_score, 
                    intervals[i].label, 
                    intervals[i].description
                ))
        
        return self._rebuild_rubric_with_intervals(rubric, new_intervals)
    
    def _apply_top_heavy_scaling(self, rubric: str, judge_name: str) -> str:
        """Apply top-heavy scaling - expand low scores, compress high scores."""
        intervals = self._extract_scoring_intervals(rubric)
        if len(intervals) != 5:
            return rubric
        
        # New intervals: expand bottom, compress top
        new_ranges = [
            (0.0, 1.5),   # Was 0.0-0.9
            (1.6, 2.2),   # Was 1.0-1.9
            (2.3, 2.8),   # Was 2.0-2.9  
            (2.9, 3.4),   # Was 3.0-3.9
            (3.5, 4.0)    # Was 4.0
        ]
        
        new_intervals = []
        for i, (min_score, max_score) in enumerate(new_ranges):
            if i < len(intervals):
                new_intervals.append(ScoreInterval(
                    min_score, max_score,
                    intervals[i].label,
                    intervals[i].description
                ))
        
        return self._rebuild_rubric_with_intervals(rubric, new_intervals)
    
    # Criteria Strictness Variations  
    def _apply_standard_strictness(self, rubric: str, judge_name: str) -> str:
        """Apply standard strictness (original criteria) - no change."""
        return rubric
    
    def _apply_strict_criteria(self, rubric: str, judge_name: str) -> str:
        """Apply stricter criteria - raise the bar for each score level."""
        intervals = self._extract_scoring_intervals(rubric)
        
        # Make criteria stricter by shifting requirements up one level
        strictness_replacements = {
            # General strictness indicators
            'minor': 'significant',
            'some': 'many',  
            'occasional': 'frequent',
            'partially': 'substantially',
            'mostly': 'completely',
            'generally': 'consistently',
            'adequate': 'excellent',
            'good': 'exceptional',
            'clear': 'crystal clear',
            'relevant': 'highly relevant',
            'appropriate': 'optimal',
            
            # Specific to different judges
            'inaccuracies': 'major inaccuracies',
            'errors': 'critical errors',
            'gaps': 'significant gaps',
            'issues': 'serious issues',
            'concerns': 'major concerns',
            'violations': 'severe violations',
            'problems': 'critical problems',
        }
        
        strict_rubric = rubric
        for old, new in strictness_replacements.items():
            # Case-insensitive replacement but preserve original case
            pattern = re.compile(re.escape(old), re.IGNORECASE)
            strict_rubric = pattern.sub(lambda m: new if m.group().islower() else new.capitalize(), strict_rubric)
        
        return strict_rubric
    
    def _apply_lenient_criteria(self, rubric: str, judge_name: str) -> str:
        """Apply more lenient criteria - lower the bar for each score level."""
        intervals = self._extract_scoring_intervals(rubric)
        
        # Make criteria more lenient
        lenient_replacements = {
            # General leniency indicators
            'significant': 'minor',
            'many': 'some',
            'frequent': 'occasional', 
            'substantially': 'partially',
            'completely': 'mostly',
            'consistently': 'generally',
            'excellent': 'adequate',
            'exceptional': 'good',
            'crystal clear': 'clear',
            'highly relevant': 'relevant',
            'optimal': 'appropriate',
            
            # Specific leniency  
            'major inaccuracies': 'inaccuracies',
            'critical errors': 'errors',
            'significant gaps': 'gaps',
            'serious issues': 'issues',
            'major concerns': 'concerns',
            'severe violations': 'violations',
            'critical problems': 'problems',
        }
        
        lenient_rubric = rubric
        for old, new in lenient_replacements.items():
            pattern = re.compile(re.escape(old), re.IGNORECASE)
            lenient_rubric = pattern.sub(lambda m: new if m.group().islower() else new.capitalize(), lenient_rubric)
        
        return lenient_rubric


def create_all_scoring_criteria_variations() -> Dict[str, Dict[str, str]]:
    """
    Create scoring criteria variations for all 10 judges.
    
    Returns:
        Dictionary mapping judge names to their scoring criteria variations
    """
    from pipeline.utils.judge_rubrics import JUDGE_RUBRICS
    
    generator = ScoringCriteriaVariationGenerator()
    all_variations = {}
    
    for judge_name, rubric_func in JUDGE_RUBRICS.items():
        original_rubric = rubric_func()
        all_variations[judge_name] = generator.generate_variations(
            original_rubric, judge_name
        )
    
    return all_variations


def generate_judge_combinations() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Generate different judge combination strategies for testing.
    
    Returns:
        Tuple of (single_variations, systematic_variations, random_variations)
    """
    from pipeline.utils.judge_rubrics import JUDGE_RUBRICS
    
    judges = list(JUDGE_RUBRICS.keys())
    variants = ['bottom_heavy', 'top_heavy', 'strict', 'lenient']
    
    # Primary: Single judge variations
    single_variations = []
    for target_judge in judges:
        for variant in variants:
            combo = {judge: 'original' for judge in judges}
            combo[target_judge] = variant
            single_variations.append({
                'name': f'{target_judge}_{variant}',
                'combination': combo
            })
    
    # Secondary: Systematic variations (all judges same direction)
    systematic_variations = []
    for variant in variants:
        combo = {judge: variant for judge in judges}
        systematic_variations.append({
            'name': f'all_{variant}',
            'combination': combo
        })
    
    # Stress test: Random combinations (for future use)
    import random
    random_variations = []
    for i in range(20):  # Generate 20 random combinations
        combo = {judge: random.choice(['original'] + variants) for judge in judges}
        random_variations.append({
            'name': f'random_{i+1}',
            'combination': combo
        })
    
    return single_variations, systematic_variations, random_variations


if __name__ == "__main__":
    # Test the generator
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    
    from pipeline.utils.judge_rubrics import get_harmlessness_rubric
    
    generator = ScoringCriteriaVariationGenerator()
    original = get_harmlessness_rubric()
    variations = generator.generate_variations(original, 'harmlessness')
    
    print("Generated variations:")
    for variant_type in variations:
        print(f"- {variant_type}")
    
    print(f"\nOriginal length: {len(original)}")
    print(f"Strict variant length: {len(variations['strict'])}")
    
    # Test judge combinations
    print("\nTesting judge combinations...")
    single, systematic, random = generate_judge_combinations()
    print(f"Single combinations: {len(single)}")
    print(f"Systematic combinations: {len(systematic)}")  
    print(f"Random combinations: {len(random)}")
    
    # Show first few single combinations
    print("\nFirst 5 single combinations:")
    for combo in single[:5]:
        print(f"- {combo['name']}")