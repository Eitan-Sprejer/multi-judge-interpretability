from openai import AsyncOpenAI
import os
import pandas as pd
import random
import pickle
from tqdm import tqdm
import json
import asyncio # Added import

from dotenv import load_dotenv
load_dotenv()

# This key right here is a proxy key for lambda.ai
openai_api_key = os.getenv("OPEN_AI_API_KEY")
openai_api_base = "https://api.lambda.ai/v1"

# Initialize the OpenAI client
async_client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Choose the model
model = "llama3.1-405b-instruct-fp8"

# Define the prompts for the simulated "human evaluators"
def get_prompts(persona_name: str, query: str, model_answer: str, persona_bio: str) -> tuple[str, str]:
    
    def get_system_prompt(persona_name: str) -> str:
        # Read the system prompt from the file
        with open("project/human_feedback_simulation/system_prompt_template.txt", "r") as file:
            system_prompt = file.read()
        return system_prompt.format(PERSONA_NAME=persona_name)
    def get_user_prompt(query: str, model_answer: str, persona_name: str, persona_bio: str) -> str:
        # Read the user prompt from the file
        with open("project/human_feedback_simulation/user_prompt_template.txt", "r") as file:
            user_prompt = file.read()
        return user_prompt.format(USER_PROMPT=query, PERSONA_NAME=persona_name, PERSONA_BIO=persona_bio, MODEL_ANSWER=model_answer)
    
    system_prompt = get_system_prompt(persona_name)
    user_prompt = get_user_prompt(query, model_answer, persona_name, persona_bio)
    
    return system_prompt, user_prompt

# Async version of get_human_feedback
async def async_get_human_feedback(persona_name: str, query: str, model_answer: str, persona_bio: str) -> str:
    system_prompt, user_prompt = get_prompts(persona_name, query, model_answer, persona_bio)
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    content = response.choices[0].message.content
    return content if content is not None else ""

personas = {  # persona_name - persona_bio
    "Sarah Chen - University Professor": "Dr. Sarah Chen, a rigorous university professor who values accuracy, depth, and proper citations. She appreciates nuanced analysis and gets frustrated by oversimplifications or unsupported claims.",
    "Marcus Rivera - Startup CEO": "Marcus Rivera, a startup CEO who needs information fast and actionable. He values conciseness, practical relevance, and clear next steps. Long-winded explanations annoy him.",
    "Luna Martinez - Novelist": "Luna Martinez, a novelist and poet who appreciates eloquent language, creative analogies, and engaging storytelling. She's drawn to answers that paint vivid pictures and avoid dry, technical jargon.",
    "David Kim - Software Architect": "David Kim, a software architect who questions everything and values logical reasoning. He appreciates technical precision, identifies edge cases, and dislikes vague or overly optimistic answers.",
    "Amara Johnson - Therapist": "Dr. Amara Johnson, a therapist who prioritizes emotional intelligence and human connection. She values answers that show understanding of feelings and consider psychological impact.",
    "Elena Rodriguez - Working Parent": "Elena Rodriguez, a working mother of three who needs information that's immediately useful for real-world situations. She appreciates common-sense advice and gets impatient with theoretical discussions.",
    "Jamie Park - College Student": "Jamie Park, an enthusiastic college junior who loves learning new things. They appreciate clear explanations, interesting examples, and answers that spark further questions.",
    "Robert Thompson - Data Scientist": "Robert Thompson, a data scientist who notices inconsistencies and appreciates comprehensive coverage. He values thoroughness, statistical rigor, and well-structured information."
}

async def async_main():
    N_samples = 10000
    base_data_path = 'dataset/data.pkl'
    save_path = 'dataset/data_with_human_feedback_async.pickle' # Changed save path for async version
    concurrency_limit = 10 # Number of concurrent API calls

    print(f"Loading base data from {base_data_path} for {N_samples} samples...")
    with open(base_data_path, 'rb') as file:
        base_data_list = pickle.load(file)
    
    df = pd.DataFrame(base_data_list[:N_samples])

    df['human_feedback'] = None
    df['human_feedback_score'] = pd.NA
    df['human_feedback_analysis'] = None
    df['persona_name'] = None
    df['persona_bio'] = None

    if os.path.exists(save_path):
        print(f"Attempting to load existing results from {save_path}...")
        try:
            df_saved = pd.read_pickle(save_path)
            common_indices = df.index.intersection(df_saved.index)
            cols_to_load = ['human_feedback', 'human_feedback_score', 'human_feedback_analysis', 'persona_name', 'persona_bio']
            cols_present_in_saved = [col for col in cols_to_load if col in df_saved.columns]

            for col in cols_present_in_saved:
                df.loc[common_indices, col] = df_saved.loc[common_indices, col]
            
            already_processed_count = df['human_feedback_score'].notna().sum()
            print(f"Successfully loaded and merged data. {already_processed_count} out of {N_samples} rows were already processed.")
        except Exception as e:
            print(f"Error loading or merging {save_path}: {e}. Proceeding with initialized/base data.")
    else:
        print(f"No existing results file found at {save_path}. Starting fresh.")

    persona_names = list(personas.keys())
    
    tasks = []
    # Store original DataFrame indices for rows that need processing
    rows_to_process_indices = [] 

    for i, row_data in df.iterrows():
        # Only create tasks for rows not yet processed
        if pd.isna(df.at[i, 'human_feedback_score']):
            query = df.at[i, 'instruction']
            model_answer = df.at[i, 'answer']
            random_persona_name = random.choice(persona_names)
            persona_bio = personas[random_persona_name]
            
            # Store persona info now, as tasks run out of order and we need this for the DataFrame
            df.at[i, 'persona_name'] = random_persona_name
            df.at[i, 'persona_bio'] = persona_bio
            
            tasks.append(async_get_human_feedback(random_persona_name, query, model_answer, persona_bio))
            rows_to_process_indices.append(i) # Keep track of the original index


    newly_processed_count = 0
    total_to_process = len(rows_to_process_indices)
    
    if total_to_process == 0:
        print("No new samples to process. Exiting.")
        return

    print(f"Starting to process {total_to_process} new samples asynchronously...")
    # Process tasks in batches to manage concurrency
    for batch_start in tqdm(range(0, total_to_process, concurrency_limit), desc="Processing samples in batches"):
        batch_end = min(batch_start + concurrency_limit, total_to_process)
        current_batch_tasks = tasks[batch_start:batch_end]
        # Get the original DataFrame indices for the current batch
        current_batch_indices = rows_to_process_indices[batch_start:batch_end]

        # return_exceptions=True allows us to handle individual task failures
        results = await asyncio.gather(*current_batch_tasks, return_exceptions=True)

        for idx, result in enumerate(results):
            original_df_index = current_batch_indices[idx] # Get original DataFrame index
            
            if isinstance(result, Exception):
                print(f"--- API Error for index {original_df_index}: {result} ---")
                df.at[original_df_index, 'human_feedback'] = f"Error: API call failed - {result}"
            elif isinstance(result, str): # Ensure result is a string before json.loads
                feedback_response = result
                feedback_dict_for_print = None # For printing an example
                try:
                    feedback = json.loads(feedback_response)
                    df.at[original_df_index, 'human_feedback'] = feedback
                    # Use .get() for safer access to dictionary keys
                    df.at[original_df_index, 'human_feedback_score'] = feedback.get('score') 
                    df.at[original_df_index, 'human_feedback_analysis'] = feedback.get('analysis')
                    feedback_dict_for_print = feedback
                except json.JSONDecodeError as e:
                    print(f"--- JSON Decode Error for index {original_df_index}: {e} ---")
                    print(f"--- Faulty Feedback String: {feedback_response} ---")
                    df.at[original_df_index, 'human_feedback'] = f"Error: {e}. Original: {feedback_response}"
            else:
                # Should not happen if async_get_human_feedback always returns str or raises Exception
                print(f"--- Unexpected result type for index {original_df_index}: {type(result)} ---")
                df.at[original_df_index, 'human_feedback'] = f"Error: Unexpected result type - {type(result)}"


            newly_processed_count += 1

            # Every 100 newly processed samples, save the data and print checkpoint info
            if newly_processed_count > 0 and newly_processed_count % 100 == 0:
                print(f"\n--- Checkpoint after {newly_processed_count} new samples ---")
                # Fetch some data from the last processed item in the batch for context
                last_query = df.at[original_df_index, 'instruction']
                last_persona = df.at[original_df_index, 'persona_name']
                print(f"--- Example from last batch (index {original_df_index}): Persona: {last_persona}, Query: {last_query[:50]}... ---")
                if feedback_dict_for_print:
                     print(f"--- Feedback (parsed): {feedback_dict_for_print} ---")
                elif isinstance(result, str) and not isinstance(result, Exception) : # only print if not an error and string
                     print(f"--- Feedback (raw/error): {result[:200]}... ---")
                print(f"Saving progress to {save_path}...")
                df.to_pickle(save_path)

    print(f"\nProcessing complete. Total newly processed: {newly_processed_count}. Saving final data to {save_path}...")
    df.to_pickle(save_path)
    print("Done.")

if __name__ == "__main__":
    # Standard way to run asyncio programs
    # Handles "RuntimeError: Event loop is closed" if script is run multiple times in some environments like Jupyter
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(async_main())
