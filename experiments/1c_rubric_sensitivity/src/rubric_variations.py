"""
Rubric Variation Generator

Creates semantically equivalent but differently phrased versions of judge rubrics.
Three variation types: formal academic, casual conversational, restructured.
"""

import re
from typing import Dict, List, Callable


class RubricVariationGenerator:
    """Generates rubric variations for testing semantic robustness."""
    
    def __init__(self):
        """Initialize the variation generator."""
        self.variation_strategies = {
            'formal': self._make_formal,
            'casual': self._make_casual,
            'restructured': self._restructure
        }
    
    def generate_variations(self, original_rubric: str, judge_name: str) -> Dict[str, str]:
        """
        Generate all variations of a rubric.
        
        Args:
            original_rubric: The original rubric text
            judge_name: Name of the judge for context-aware variations
            
        Returns:
            Dictionary with variation types as keys and rubric text as values
        """
        variations = {'original': original_rubric}
        
        for variation_type, strategy in self.variation_strategies.items():
            variations[variation_type] = strategy(original_rubric, judge_name)
        
        return variations
    
    def _make_formal(self, rubric: str, judge_name: str) -> str:
        """
        Convert rubric to formal academic language.
        
        - Replace casual phrases with formal equivalents
        - Use passive voice where appropriate
        - Add technical terminology
        - Maintain exact same scoring criteria
        """
        formal_rubric = rubric
        
        # Replace judge version with formal designation
        formal_rubric = re.sub(
            r'You are ([A-Z\-]+)-JUDGE-v[\d\.]+',
            r'This evaluation instrument designates the assessor as \1 Evaluation System Version 1.0',
            formal_rubric
        )
        
        # Formal replacements
        replacements = {
            # Common phrases
            'Score how': 'Conduct a quantitative assessment of the degree to which',
            'on a 0.0-4.0 scale': 'utilizing a continuous numerical scale ranging from 0.0 to 4.0',
            'assistant\'s reply': 'system-generated response',
            'assistant\'s response': 'system-generated response',
            
            # Scoring level descriptors  
            'CRITICAL VIOLATIONS': 'CATEGORY I: CRITICAL SEVERITY VIOLATIONS',
            'SERIOUS CONCERNS': 'CATEGORY II: SUBSTANTIAL DEFICIENCIES',
            'MINOR ISSUES': 'CATEGORY III: MINOR DEFICIENCIES',
            'APPROPRIATE': 'CATEGORY IV: ACCEPTABLE STANDARD',
            'EXEMPLARY': 'CATEGORY V: EXCEPTIONAL PERFORMANCE',
            
            # Common verbs
            'Contains': 'Encompasses',
            'Shows': 'Demonstrates',
            'Features': 'Exhibits',
            'Includes': 'Incorporates',
            'Provides': 'Furnishes',
            'Uses': 'Employs',
            'Lacks': 'Exhibits deficiency in',
            
            # Scoring instructions
            'Return ONLY': 'The output shall consist exclusively of',
            'rounded to one decimal place': 'with precision limited to one decimal position',
            'Use one decimal place precision': 'Maintain numerical precision to one decimal position',
            'caps the score at': 'establishes an upper bound for the score at',
            
            # Evaluative terms
            'good': 'satisfactory',
            'bad': 'unsatisfactory', 
            'clear': 'unambiguous',
            'poor': 'inadequate',
            'strong': 'robust',
            'weak': 'insufficient'
        }
        
        for old, new in replacements.items():
            formal_rubric = formal_rubric.replace(old, new)
        
        return formal_rubric
    
    def _make_casual(self, rubric: str, judge_name: str) -> str:
        """
        Convert rubric to casual conversational language.
        
        - Use simpler vocabulary
        - More direct, conversational tone
        - Shorter sentences
        - Maintain exact same scoring criteria
        """
        casual_rubric = rubric
        
        # Replace judge version with casual intro
        casual_rubric = re.sub(
            r'You are ([A-Z\-]+)-JUDGE-v[\d\.]+',
            r"You're the \1 Judge (version 1.0)",
            casual_rubric
        )
        
        # Casual replacements
        replacements = {
            # Formal phrases to casual
            'Score how': 'Rate how',
            'on a 0.0-4.0 scale': 'from 0.0 to 4.0',
            'assistant\'s reply': 'AI\'s answer',
            'assistant\'s response': 'AI\'s answer',
            'Scoring Criteria': 'How to Score',
            'Scoring Guidelines': 'Tips for Scoring',
            'Output Format': 'What to Return',
            
            # Formal words to simple
            'Encompasses': 'Has',
            'Demonstrates': 'Shows',
            'Exhibits': 'Has',
            'Incorporates': 'Includes',
            'Furnishes': 'Gives',
            'Employs': 'Uses',
            'comprising': 'including',
            'pertaining to': 'about',
            'subsequent': 'next',
            'prior': 'before',
            
            # Instructions
            'Return ONLY': 'Just give me',
            'shall': 'should',
            'must': 'needs to',
            'Evaluate': 'Check',
            'Assess': 'Look at',
            'Consider': 'Think about',
            
            # Complex phrases
            'rounded to one decimal place': 'with one decimal (like 2.5)',
            'Use one decimal place precision': 'Use one decimal point',
            'caps the score at': 'limits the score to',
            'regardless of': 'no matter what',
            'pertinent': 'relevant',
            'aforementioned': 'mentioned above'
        }
        
        for old, new in replacements.items():
            casual_rubric = casual_rubric.replace(old, new)
        
        # Make bullet points more casual
        casual_rubric = casual_rubric.replace('•', '-')
        
        return casual_rubric
    
    def _restructure(self, rubric: str, judge_name: str) -> str:
        """
        Restructure the rubric while keeping semantic meaning.
        
        - Reorder sections (put guidelines before criteria)
        - Reorganize bullet points
        - Different grouping of information
        - Maintain exact same scoring criteria
        """
        # Parse the rubric into sections
        sections = self._parse_rubric_sections(rubric)
        
        # Restructure by reordering and regrouping
        restructured = []
        
        # Start with the judge designation (unchanged)
        if 'header' in sections:
            restructured.append(sections['header'])
        
        # Put guidelines before criteria (reversed order)
        if 'guidelines' in sections:
            restructured.append("EVALUATION GUIDELINES\n" + "="*20)
            restructured.append(sections['guidelines'])
        
        # Add output format early
        if 'output' in sections:
            restructured.append("\nRESPONSE FORMAT\n" + "-"*15)
            restructured.append(sections['output'])
        
        # Now add scoring criteria with different organization
        if 'criteria' in sections:
            restructured.append("\nSCORING FRAMEWORK\n" + "="*20)
            # Reorganize scoring bands from high to low (reverse typical order)
            criteria_text = sections['criteria']
            
            # Extract score ranges and their descriptions
            score_blocks = re.findall(
                r'([\d\.]+\-[\d\.]+|[\d\.]+)\s*=\s*([^\n]+)\n((?:[^\d].*\n?)*)',
                criteria_text
            )
            
            if score_blocks:
                # Reverse the order
                score_blocks.reverse()
                restructured.append("SCORING LEVELS (Best to Worst):\n")
                for score_range, title, description in score_blocks:
                    restructured.append(f"\n[{score_range}] {title}")
                    restructured.append(description.strip())
            else:
                # Fallback if pattern doesn't match
                restructured.append(criteria_text)
        
        # Add any special considerations at the end
        if 'special' in sections:
            restructured.append("\nADDITIONAL CONSIDERATIONS\n" + "-"*20)
            restructured.append(sections['special'])
        
        return '\n'.join(restructured)
    
    def _parse_rubric_sections(self, rubric: str) -> Dict[str, str]:
        """Parse rubric into logical sections."""
        sections = {}
        
        # Extract header (judge designation)
        header_match = re.search(r'^(You are .+?)\n', rubric, re.MULTILINE)
        if header_match:
            sections['header'] = header_match.group(0)
        
        # Extract scoring criteria
        criteria_match = re.search(
            r'Scoring Criteria\n[-=]+\n(.*?)(?=Scoring Guidelines|Output Format|Special|$)',
            rubric, re.DOTALL
        )
        if criteria_match:
            sections['criteria'] = criteria_match.group(1).strip()
        
        # Extract guidelines
        guidelines_match = re.search(
            r'Scoring Guidelines\n[-=]+\n(.*?)(?=Output Format|Special|$)',
            rubric, re.DOTALL
        )
        if guidelines_match:
            sections['guidelines'] = guidelines_match.group(1).strip()
        
        # Extract output format
        output_match = re.search(
            r'Output Format\n[-=]+\n(.*?)$',
            rubric, re.DOTALL
        )
        if output_match:
            sections['output'] = output_match.group(1).strip()
        
        # Extract special considerations
        special_match = re.search(
            r'Special (?:Considerations|Cases)\n[-=]+\n(.*?)(?=Output Format|$)',
            rubric, re.DOTALL
        )
        if special_match:
            sections['special'] = special_match.group(1).strip()
        
        return sections


def create_harmlessness_variations() -> Dict[str, str]:
    """Create variations specifically for the harmlessness judge."""
    from pipeline.utils.judge_rubrics import get_harmlessness_rubric
    
    generator = RubricVariationGenerator()
    original = get_harmlessness_rubric()
    return generator.generate_variations(original, 'harmlessness')


def create_all_judge_variations() -> Dict[str, Dict[str, str]]:
    """
    Create variations for all 10 judges.
    
    Returns:
        Dictionary mapping judge names to their rubric variations
    """
    from pipeline.utils.judge_rubrics import JUDGE_RUBRICS
    
    generator = RubricVariationGenerator()
    all_variations = {}
    
    for judge_name, rubric_func in JUDGE_RUBRICS.items():
        original_rubric = rubric_func()
        all_variations[judge_name] = generator.generate_variations(
            original_rubric, judge_name
        )
    
    return all_variations