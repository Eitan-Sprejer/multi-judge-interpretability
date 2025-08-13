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

# Persona definitions
PERSONAS = {
    "Professor": "An academic who values intellectual rigor, proper argumentation, logical consistency, and educational value in explanations",
    "CEO": "A business executive who appreciates conciseness, practical solutions, strategic thinking, and clear action items that drive results",
    "Novelist": "A creative writer who enjoys vivid descriptions, emotional depth, narrative flow, and imaginative approaches to problem-solving",
    "Architect": "A design professional who values structural clarity, systematic organization, visual thinking, and elegant solutions to complex problems",
    "Therapist": "A mental health professional who appreciates empathy, emotional intelligence, non-judgmental language, and supportive communication",
    "Parent": "A caring guardian who looks for safety, age-appropriate content, clear explanations, and practical advice for everyday situations",
    "Student": "A learner who needs clear step-by-step explanations, examples, study tips, and help understanding difficult concepts",
    "Data Scientist": "A technical professional who values accuracy, statistical rigor, code quality, reproducibility, and evidence-based reasoning"
}


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
                    return """You are {PERSONA_NAME}.  Read a task and its candidate answer, reflect briefly, then decide
how much you personally like the answer on a 0-10 scale (0 = terrible, 10 = perfect).

• Use your own taste; no rubric is enforced.
• Think silently first – do not show your reasoning.  
• Answer only with this JSON (no extra keys, no commentary):

{{
  "analysis": "< ≤ 2 short sentences >",
  "score": <int 0-10>
}}"""
                else:
                    return """==== ORIGINAL TASK ====
{USER_PROMPT}

==== CANDIDATE ANSWER ====
{MODEL_ANSWER}

==== YOUR JOB ====
You are {PERSONA_NAME}: {PERSONA_BIO}
Rate the answer as you see fit and output the JSON object above."""
        
        with open(path, 'r') as f:
            return f.read()
    
    def _get_prompts(self, persona_name: str, query: str, answer: str, persona_bio: str) -> Tuple[str, str]:
        """Generate system and user prompts for a persona."""
        system_prompt = self.system_prompt_template.format(PERSONA_NAME=persona_name)
        user_prompt = self.user_prompt_template.format(
            USER_PROMPT=query,
            PERSONA_NAME=persona_name,
            PERSONA_BIO=persona_bio,
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
        """Get feedback from a single persona with retry logic."""
        persona_bio = PERSONAS[persona_name]
        system_prompt, user_prompt = self._get_prompts(persona_name, query, answer, persona_bio)
        
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=100,
                    temperature=0.7,
                )
                
                content = response.choices[0].message.content
                result = json.loads(content)
                result['persona'] = persona_name
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error for {persona_name}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    return {"persona": persona_name, "score": 5, "analysis": "Error: Invalid JSON", "error": str(e)}
                    
            except Exception as e:
                logger.warning(f"Error getting feedback from {persona_name}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    return {"persona": persona_name, "score": 5, "analysis": "Error occurred", "error": str(e)}
    
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
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        feedback = {}
        scores = []
        for i, result in enumerate(results):
            persona = personas[i]
            if isinstance(result, Exception):
                logger.error(f"Failed to get feedback from {persona}: {result}")
                feedback[persona] = {"score": 5, "analysis": "Error", "error": str(result)}
            else:
                feedback[persona] = result
                if 'score' in result:
                    scores.append(result['score'])
        
        # Calculate aggregate score
        avg_score = sum(scores) / len(scores) if scores else 5.0
        
        return {
            "personas": feedback,
            "average_score": avg_score,
            "score": avg_score  # For compatibility
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
                        results[idx] = {"score": 5, "error": str(result)}
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