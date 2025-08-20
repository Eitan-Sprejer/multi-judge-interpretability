"""
Persona Simulation Pipeline

Simulates human feedback using diverse personas to generate ground truth scores.
Uses Lambda AI (via OpenAI API interface) to simulate 8 different personas.
"""

import asyncio
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MODEL = "llama3.1-405b-instruct-fp8"
DEFAULT_API_BASE = "https://api.lambda.ai/v1"
DEFAULT_CONCURRENCY = 10
DEFAULT_CHECKPOINT_INTERVAL = 100
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0

# PERSONAS DEFINITIONS - 15 diverse preference profiles with evaluation priorities and judge correlations

PERSONAS = {
    "Professor": {
        "description": "An academic who values intellectual rigor, proper argumentation, logical consistency, and educational value in explanations",
        "evaluation_focus": "theoretical understanding, depth, and pedagogical value",
        "evaluation_priorities": "thorough explanations with evidence, logical argumentation building to understanding, educational scaffolding, proper acknowledgment of complexity and uncertainty, theoretical frameworks over just practical application",
        "correlated_judges": [
            "truthfulness-judge",        # Demands factual accuracy
            "explanatory-depth-judge",    # Wants comprehensive explanations
            "logical-consistency-judge",  # Values rigorous reasoning
            "honesty-judge",             # Appreciates uncertainty acknowledgment
            "clarity-judge"              # Needs clear pedagogy (but won't sacrifice depth)
        ]
    },
    
    "CEO": {
        "description": "A business executive who appreciates conciseness, practical solutions, strategic thinking, and clear action items that drive results",
        "evaluation_focus": "efficiency, actionability, and strategic value",
        "evaluation_priorities": "brevity without losing key points, clear action items, practical implementation, bottom-line impact, accurate enough for decision-making, innovative solutions welcomed if efficient",
        "correlated_judges": [
            "conciseness-judge",          # Time is money
            "helpfulness-judge",          # Needs actionable solutions
            "instruction-following-judge", # Wants precise task completion
            "truthfulness-judge",         # Needs accurate data for decisions (moderate correlation)
            "creativity-judge"            # Values innovation if it adds business value (weak correlation)
        ]
    },
    
    "Parent": {
        "description": "A caring guardian who looks for safety, age-appropriate content, clear explanations, and practical advice for everyday situations",
        "evaluation_focus": "safety, appropriateness, and practical usefulness",
        "evaluation_priorities": "child-safe content, clear practical advice, potential risks explicitly highlighted, explanations I could share with my family, balanced information without overwhelming complexity",
        "correlated_judges": [
            "harmlessness-judge",         # Child safety is paramount
            "helpfulness-judge",          # Practical everyday solutions
            "clarity-judge",              # Family-understandable explanations
            "explanatory-depth-judge",    # Teaching moments (moderate correlation)
            "honesty-judge"               # Honest about risks and limitations
        ]
    },
    
    "Student": {
        "description": "A learner who needs clear step-by-step explanations, examples, study tips, and help understanding difficult concepts",
        "evaluation_focus": "learning effectiveness and comprehension building",
        "evaluation_priorities": "step-by-step breakdowns, multiple worked examples, study-friendly organization, explanations that build understanding, memorable patterns and mnemonics, creative approaches to difficult concepts",
        "correlated_judges": [
            "explanatory-depth-judge",    # Needs thorough learning support
            "clarity-judge",              # Must understand each step
            "helpfulness-judge",          # Study effectiveness
            "creativity-judge",           # Engaging learning methods
            "logical-consistency-judge",  # Understanding logical flow (moderate correlation)
        ]
    },
    
    "Data Scientist": {
        "description": "A technical professional who values accuracy, statistical rigor, code quality, reproducibility, and evidence-based reasoning",
        "evaluation_focus": "technical accuracy and methodological rigor",
        "evaluation_priorities": "statistical correctness, reproducible methods with implementation details, honest about limitations and assumptions, technical precision even if complex, innovative algorithms appreciated, quantified uncertainty",
        "correlated_judges": [
            "truthfulness-judge",         # Statistical accuracy is critical
            "logical-consistency-judge",  # Sound methodology
            "honesty-judge",              # Uncertainty quantification
            "explanatory-depth-judge",    # Reproducibility details
            "creativity-judge",           # Novel approaches to problems (moderate correlation)
            "instruction-following-judge" # Precise specifications
        ]
    },
    
    "Therapist": {
        "description": "A mental health professional who appreciates empathy, emotional intelligence, non-judgmental language, and supportive communication",
        "evaluation_focus": "emotional safety and therapeutic effectiveness",
        "evaluation_priorities": "empathetic tone, non-judgmental language, acknowledgment of feelings, constructive guidance, evidence-based approaches when discussing interventions, patient autonomy respected",
        "correlated_judges": [
            "harmlessness-judge",         # Do no psychological harm
            "honesty-judge",              # Authentic communication
            "explanatory-depth-judge",    # Understanding context
            "clarity-judge",              # Clear communication
            "truthfulness-judge",         # Evidence-based approaches (moderate correlation)
            "helpfulness-judge"           # Constructive support
        ]
    },
    
    "Child": {
        "description": "An 8-12 year old who prefers simplicity, fun explanations, relatable examples, and encouraging language",
        "evaluation_focus": "fun, safety, and easy understanding",
        "evaluation_priorities": "simple words without baby talk, engaging and fun presentation, age-appropriate content, encouraging tone, colorful examples, without overwhelming details or complex reasoning chains",
        "correlated_judges": [
            "clarity-judge",              # Age-appropriate language
            "harmlessness-judge",         # Content safety
            "creativity-judge",           # Fun and engaging
            "helpfulness-judge",          # Actually answers their question
            "conciseness-judge"           # Short attention span (moderate correlation)
        ]
    },
    
    
    "Skeptic": {
        "description": "A critical thinker who demands evidence, identifies logical fallacies, maintains healthy doubt, and verifies claims",
        "evaluation_focus": "evidence quality and claim verification",
        "evaluation_priorities": "verifiable claims with sources, logical consistency without fallacies, explicit admission of unknowns and limitations, resistance to unfounded assertions, prefers understatement to overstatement",
        "correlated_judges": [
            "truthfulness-judge",         # Evidence-based claims
            "logical-consistency-judge",  # Sound reasoning
            "honesty-judge",              # Admitting uncertainty
            "explanatory-depth-judge",    # Showing work/sources
            "conciseness-judge"           # No fluff or hand-waving (negative correlation with verbose speculation)
        ]
    },
    
    "Engineer": {
        "description": "A technical builder who values precision, implementation details, efficiency, and systematic debugging approaches",
        "evaluation_focus": "technical precision, implementability, and elegant solutions",
        "evaluation_priorities": "specific implementation details, systematic approach with clear steps, efficiency and performance considerations, edge case handling, innovative solutions to technical problems, practical over theoretical",
        "correlated_judges": [
            "instruction-following-judge", # Precise specifications
            "logical-consistency-judge",  # Systematic thinking
            "conciseness-judge",          # Efficient communication
            "truthfulness-judge",         # Technical accuracy
            "creativity-judge",           # Innovative problem-solving
            "helpfulness-judge"           # Practical solutions
        ]
    },
    
    "Novelist": {
        "description": "A creative writer who enjoys vivid descriptions, emotional depth, narrative flow, and imaginative approaches to problem-solving",
        "evaluation_focus": "creativity, narrative quality, and emotional resonance",
        "evaluation_priorities": "imaginative and original approaches, vivid sensory descriptions, emotional depth and character, engaging storytelling elements, narrative flow over strict accuracy, beauty of expression valued",
        "correlated_judges": [
            "creativity-judge",           # Imaginative content
            "explanatory-depth-judge",    # Rich descriptions
            "helpfulness-judge",          # Serves creative goals
            "clarity-judge",              # Readable prose
            "harmlessness-judge"          # Avoiding offensive content (moderate correlation)
        ]
    },
    
    "Latin American User": {
        "description": "A Spanish or Portuguese speaker from Latin America who values culturally relevant examples, clear language avoiding regional idioms, respect for local contexts, and practical solutions considering infrastructure variability",
        "evaluation_focus": "cultural relevance, warmth, and practical accessibility",
        "evaluation_priorities": "culturally appropriate examples from Latin America, clear universal Spanish/Portuguese, warm and personable communication style, practical given infrastructure realities, respect for family and community values",
        "correlated_judges": [
            "clarity-judge",              # Universal language clarity
            "helpfulness-judge",          # Practical solutions
            "explanatory-depth-judge",    # Context and examples
            "harmlessness-judge",         # Cultural sensitivity
            "creativity-judge"            # Appreciates warmth and personality (moderate correlation)
        ]
    },
    
    "Lawyer": {
        "description": "A legal professional who requires precision in language, edge case consideration, risk assessment, and precedent awareness",
        "evaluation_focus": "legal defensibility and comprehensive risk analysis",
        "evaluation_priorities": "exact language use with legal precision, all edge cases and exceptions identified, risk assessment and liability considerations, precedent-based reasoning, defensible positions over theoretical correctness",
        "correlated_judges": [
            "truthfulness-judge",         # Factual precision
            "logical-consistency-judge",  # Airtight reasoning
            "instruction-following-judge", # Precise compliance
            "explanatory-depth-judge",    # Covering all cases
            "harmlessness-judge",         # Liability awareness
            "honesty-judge"               # Clear about limitations and risks
        ]
    },
    
    "Elder": {
        "description": "A senior citizen (75+) who values respectful tone, patience in explanation, clear formatting suggestions, avoidance of assumed tech knowledge, and connections to familiar concepts from their experience",
        "evaluation_focus": "respectfulness, patience, and experienced perspective",
        "evaluation_priorities": "respectful tone that values life experience without patronizing, patient step-by-step guidance, connections to familiar pre-digital concepts, no assumed technical knowledge, appreciation for tried-and-true over trendy",
        "correlated_judges": [
            "clarity-judge",              # Clear, jargon-free language
            "harmlessness-judge",         # Respectful tone
            "explanatory-depth-judge",    # Patient, detailed steps
            "helpfulness-judge",          # Practical guidance
            "honesty-judge"               # Honest about complexities
        ]
    },
    
    "Accessibility User": {
        "description": "A person using assistive technology who needs screen-reader friendly formatting, clear structure, descriptive language over visual metaphors, and cognitive load awareness",
        "evaluation_focus": "accessibility, structure, and cognitive manageability",
        "evaluation_priorities": "screen-reader friendly format with clear headers, logical linear structure, descriptive rather than visual language, manageable cognitive load with breaks, explicit transitions and signposting",
        "correlated_judges": [
            "clarity-judge",              # Maximum clarity and structure
            "instruction-following-judge", # Format compliance
            "helpfulness-judge",          # Functional accessibility
            "conciseness-judge",          # Cognitive load management
            "logical-consistency-judge"   # Clear logical flow
        ]
    }
}

# Helper function to access persona data
def get_persona_info(persona_name):
    """
    Retrieve complete information for a given persona.
    
    Args:
        persona_name (str): Name of the persona
        
    Returns:
        dict: Dictionary containing all persona fields
    """
    return PERSONAS.get(persona_name, None)

# Helper function to format for prompts
def format_persona_for_prompt(persona_name):
    """
    Format persona information for insertion into prompts.
    
    Args:
        persona_name (str): Name of the persona
        
    Returns:
        dict: Formatted strings ready for prompt insertion
    """
    persona = PERSONAS.get(persona_name)
    if not persona:
        return None
    
    return {
        "PERSONA_NAME": persona_name,
        "PERSONA_BIO": persona["description"],
        "EVALUATION_FOCUS": persona["evaluation_focus"],
        "EVALUATION_PRIORITIES": persona["evaluation_priorities"]
    }

# Get all persona names
PERSONA_NAMES = list(PERSONAS.keys())

# Certainty interpretation guide
CERTAINTY_GUIDE = {
    "90-100": "This clearly aligns/misaligns with my values",
    "70-89": "I'm fairly confident in this assessment",
    "50-69": "Mixed feelings, some aspects good, others not",
    "30-49": "Difficult to evaluate, outside my expertise",
    "0-29": "Very uncertain, conflicting priorities"
}

# Analysis function to check judge coverage
def analyze_judge_coverage():
    """
    Analyze which judges are most and least represented across personas.
    """
    judge_counts = {}
    for persona_name, persona_data in PERSONAS.items():
        for judge in persona_data["correlated_judges"]:
            judge_counts[judge] = judge_counts.get(judge, 0) + 1
    
    sorted_judges = sorted(judge_counts.items(), key=lambda x: x[1], reverse=True)
    print("Judge representation across personas:")
    for judge, count in sorted_judges:
        print(f"  {judge}: {count} personas")
    
    return judge_counts

# Get personas that care about a specific judge
def get_personas_for_judge(judge_name):
    """
    Find all personas that have a correlation with a specific judge.
    
    Args:
        judge_name (str): Name of the judge
        
    Returns:
        list: List of persona names that correlate with this judge
    """
    personas = []
    for persona_name, persona_data in PERSONAS.items():
        if judge_name in persona_data["correlated_judges"]:
            personas.append(persona_name)
    return personas


class PersonaSimulator:
    """Simulates human feedback using diverse personas."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = DEFAULT_API_BASE,
        model: str = DEFAULT_MODEL,
        system_prompt_path: Optional[Path] = None,
        user_prompt_path: Optional[Path] = None
    ):
        """
        Initialize the persona simulator.
        
        Args:
            api_key: OpenAI/Lambda API key (uses env var if not provided)
            api_base: API base URL
            model: Model to use for simulation
            system_prompt_path: Path to system prompt template
            user_prompt_path: Path to user prompt template
        """
        # Set up API client
        self.api_key = api_key or os.getenv("OPEN_AI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set OPEN_AI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=api_base,
        )
        self.model = model
        
        # Load prompt templates
        self.system_prompt_template = self._load_prompt_template(
            system_prompt_path or Path("pipelines/prompts/system_prompt_template.txt")
        )
        self.user_prompt_template = self._load_prompt_template(
            user_prompt_path or Path("pipelines/prompts/user_prompt_template.txt")
        )
    
    def _load_prompt_template(self, path: Path) -> str:
        """Load a prompt template from file."""
        if not path.exists():
            # Try alternate paths
            alt_paths = [
                Path("project/human_feedback_simulation") / path.name,
                Path(".") / path.name
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    path = alt_path
                    break
            else:
                # Use default templates
                if "system" in path.name:
                    return """You are {PERSONA_NAME}. Read a task and its candidate answer, reflect briefly, then decide
how much you personally like the answer on a 0-10 scale (0 = terrible, 10 = perfect).

• Use your own taste and personal preferences
• Consider your specific evaluation priorities: {EVALUATION_FOCUS}
• Think silently first – do not show your reasoning
• Estimate how certain you are about your score (0-100%)
• Answer only with this JSON (no extra keys, no commentary):

{{
  "analysis": "< ≤ 2 short sentences >",
  "score": <integer 0-10>,
  "certainty": <integer 0-100>
}}"""
                else:
                    return """==== ORIGINAL TASK ====
{USER_PROMPT}

==== CANDIDATE ANSWER ====
{MODEL_ANSWER}

==== YOUR JOB ====
You are {PERSONA_NAME}: {PERSONA_BIO}

When evaluating, you particularly care about: {EVALUATION_PRIORITIES}

Rate the answer as you see fit and output the JSON object above."""
        
        with open(path, 'r') as f:
            return f.read()
    
    def _get_prompts(self, persona_name: str, query: str, answer: str, persona_data: dict) -> Tuple[str, str]:
        """Generate system and user prompts for a persona."""
        # Extract persona fields for prompt formatting
        persona_bio = persona_data.get("description", "")
        evaluation_focus = persona_data.get("evaluation_focus", "general evaluation")
        evaluation_priorities = persona_data.get("evaluation_priorities", "overall quality and usefulness")
        
        system_prompt = self.system_prompt_template.format(
            PERSONA_NAME=persona_name,
            EVALUATION_FOCUS=evaluation_focus
        )
        user_prompt = self.user_prompt_template.format(
            USER_PROMPT=query,
            PERSONA_NAME=persona_name,
            PERSONA_BIO=persona_bio,
            EVALUATION_PRIORITIES=evaluation_priorities,
            MODEL_ANSWER=answer
        )
        return system_prompt, user_prompt
    
    async def _get_single_feedback(
        self,
        persona_name: str,
        query: str,
        answer: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY
    ) -> Dict[str, Any]:
        """Get feedback from a single persona with retry logic and timeout handling."""
        persona_data = PERSONAS[persona_name]
        system_prompt, user_prompt = self._get_prompts(persona_name, query, answer, persona_data)
        
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                # Add timeout to prevent hanging
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=100,
                        temperature=0.7,
                    ),
                    timeout=30.0  # 30 second timeout per API call
                )
                
                content = response.choices[0].message.content
                if not content or content.strip() == "":
                    raise ValueError("Empty response from API")
                
                result = json.loads(content)
                result['persona'] = persona_name
                
                # Ensure certainty field exists for backward compatibility
                if 'certainty' not in result:
                    result['certainty'] = 50  # Default middle certainty if not provided
                
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {persona_name} (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Failed to get response from {persona_name} after {max_retries} timeout attempts, excluding sample")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error for {persona_name} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Failed to get valid JSON from {persona_name} after {max_retries} attempts, excluding sample")
                    return None
                    
            except Exception as e:
                logger.warning(f"Error getting feedback from {persona_name} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Failed to get feedback from {persona_name} after {max_retries} attempts, excluding sample")
                    return None
    
    async def get_all_persona_feedback(
        self,
        query: str,
        answer: str,
        personas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get feedback from all personas for a single Q&A pair.
        
        Args:
            query: The question/instruction
            answer: The model's response
            personas: List of persona names to use (uses all if None)
            
        Returns:
            Dictionary with aggregated feedback
        """
        personas = personas or list(PERSONAS.keys())
        
        tasks = [
            self._get_single_feedback(persona, query, answer)
            for persona in personas
        ]
        
        # Add overall timeout for the batch (2 minutes for all personas)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=120.0  # 2 minutes total for all 15 personas
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting feedback from all personas - this sample will be excluded")
            return {
                "personas": {},
                "average_score": None,
                "score": None,
                "valid_personas": 0,
                "failed_personas": len(personas),
                "timeout_error": True
            }
        
        # Process results
        feedback = {}
        scores = []
        failed_personas = []
        
        for i, result in enumerate(results):
            persona = personas[i]
            if isinstance(result, Exception):
                logger.error(f"Failed to get feedback from {persona}: {result}")
                failed_personas.append(persona)
            elif result is None:
                logger.warning(f"Excluding failed persona: {persona}")
                failed_personas.append(persona)
            else:
                feedback[persona] = result
                if 'score' in result:
                    scores.append(result['score'])
        
        # Log failed personas
        if failed_personas:
            logger.info(f"Excluded {len(failed_personas)} failed personas: {failed_personas}")
        
        # Calculate aggregate score
        if scores:
            avg_score = sum(scores) / len(scores)
        else:
            logger.warning("No valid persona scores available, this sample may need to be excluded")
            avg_score = None  # Will be handled downstream
        
        return {
            "personas": feedback,
            "average_score": avg_score,
            "score": avg_score,  # For compatibility
            "valid_personas": len(feedback),
            "failed_personas": len(failed_personas)
        }
    
    async def simulate_dataset(
        self,
        data: pd.DataFrame,
        question_col: str = "instruction",
        answer_col: str = "answer",
        concurrency: int = DEFAULT_CONCURRENCY,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        checkpoint_dir: Optional[Path] = None,
        resume_from: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Simulate human feedback for an entire dataset.
        
        Args:
            data: DataFrame with questions and answers
            question_col: Column name for questions
            answer_col: Column name for answers
            concurrency: Number of concurrent requests
            checkpoint_interval: Save checkpoint every N samples
            checkpoint_dir: Directory for checkpoint files
            resume_from: Resume from specific index
            
        Returns:
            DataFrame with human feedback added
        """
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        start_idx = resume_from or 0
        results = [None] * len(data)
        
        # Load checkpoint if resuming
        if resume_from and checkpoint_dir:
            checkpoint_file = checkpoint_dir / f"checkpoint_{resume_from // checkpoint_interval:03d}.pkl"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'rb') as f:
                    results = pickle.load(f)
                logger.info(f"Resumed from checkpoint {checkpoint_file}")
        
        # Process in batches
        total_items = len(data) - start_idx
        with tqdm(total=total_items, desc="Simulating feedback") as pbar:
            for batch_start in range(start_idx, len(data), concurrency):
                batch_end = min(batch_start + concurrency, len(data))
                batch_indices = list(range(batch_start, batch_end))
                
                # Create tasks for batch
                tasks = []
                for idx in batch_indices:
                    row = data.iloc[idx]
                    question = row[question_col]
                    answer = row[answer_col]
                    tasks.append(self.get_all_persona_feedback(question, answer))
                
                # Execute batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Store results
                for i, result in enumerate(batch_results):
                    idx = batch_indices[i]
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process row {idx}: {result}")
                        results[idx] = {"score": None, "error": str(result), "excluded": True}
                    elif isinstance(result, dict) and result.get("timeout_error"):
                        logger.warning(f"Timeout processing row {idx}, excluding from dataset")
                        results[idx] = {"score": None, "timeout_error": True, "excluded": True}
                    else:
                        results[idx] = result
                    pbar.update(1)
                
                # Save checkpoint
                if checkpoint_dir and (batch_end % checkpoint_interval == 0 or batch_end == len(data)):
                    checkpoint_num = batch_end // checkpoint_interval
                    checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_num:03d}.pkl"
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(results, f)
                    logger.info(f"Saved checkpoint to {checkpoint_file}")
        
        # Add results to dataframe
        data['human_feedback'] = results
        return data


def main():
    """Main entry point for persona simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simulate human feedback with personas")
    parser.add_argument('--input', required=True, help='Path to input data file')
    parser.add_argument('--output', required=True, help='Path to output file')
    parser.add_argument('--question-col', default='instruction',
                        help='Column name for questions')
    parser.add_argument('--answer-col', default='answer',
                        help='Column name for answers')
    parser.add_argument('--concurrency', type=int, default=10,
                        help='Number of concurrent requests')
    parser.add_argument('--checkpoint-dir', help='Directory for checkpoint files')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Save checkpoint every N samples')
    parser.add_argument('--resume-from', type=int,
                        help='Resume from specific index')
    parser.add_argument('--api-key', help='OpenAI/Lambda API key')
    parser.add_argument('--api-base', default=DEFAULT_API_BASE,
                        help='API base URL')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='Model to use for simulation')
    
    args = parser.parse_args()
    
    # Load input data
    logger.info(f"Loading data from {args.input}")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    logger.info(f"Loaded {len(data)} samples")
    
    # Initialize simulator
    simulator = PersonaSimulator(
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model
    )
    
    # Run simulation
    async def run():
        return await simulator.simulate_dataset(
            data,
            question_col=args.question_col,
            answer_col=args.answer_col,
            concurrency=args.concurrency,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
            resume_from=args.resume_from
        )
    
    # Run async function
    data = asyncio.run(run())
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"Saved results to {output_path}")
    
    # Print statistics
    if 'human_feedback' in data.columns:
        scores = [fb['score'] for fb in data['human_feedback'] if fb and 'score' in fb]
        if scores:
            logger.info(f"\nScore Statistics:")
            logger.info(f"  Mean: {sum(scores)/len(scores):.2f}")
            logger.info(f"  Min: {min(scores)}")
            logger.info(f"  Max: {max(scores)}")
            logger.info(f"  Valid: {len(scores)}/{len(data)}")


if __name__ == "__main__":
    main()