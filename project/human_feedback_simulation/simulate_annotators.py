from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
import random
import pickle
from tqdm import tqdm
import json

# Load environment variables from .env file
openai_api_key = os.getenv("OPEN_AI_API_KEY")
openai_api_base = "https://api.lambda.ai/v1"

# Initialize the OpenAI client
client = OpenAI(
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

def get_human_feedback(persona_name: str, query: str, model_answer: str, persona_bio: str) -> str:
    system_prompt, user_prompt = get_prompts(persona_name, query, model_answer, persona_bio)
    response = client.chat.completions.create(
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

if __name__ == "__main__":
    N_samples = 1000
    base_data_path = 'dataset/data.pkl'
    save_path = 'dataset/data_with_human_feedback.pickle'

    # Load the base data for the target N_samples
    print(f"Loading base data from {base_data_path} for {N_samples} samples...")
    with open(base_data_path, 'rb') as file:
        base_data_list = pickle.load(file)
    
    df = pd.DataFrame(base_data_list[:N_samples])

    # Initialize feedback columns
    df['human_feedback'] = None
    df['human_feedback_score'] = pd.NA
    df['human_feedback_analysis'] = None
    df['persona_name'] = None
    df['persona_bio'] = None

    # Load existing processed data if save_path exists
    if os.path.exists(save_path):
        print(f"Attempting to load existing results from {save_path}...")
        try:
            df_saved = pd.read_pickle(save_path)
            # Ensure we only update relevant rows and columns
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

    # Get a list of persona names
    persona_names = list(personas.keys())

    # Process each query-answer pair
    newly_processed_count = 0 # Counter for new items
    for i, row_data in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        # Skip if already processed (score is not NA)
        if pd.notna(df.at[i, 'human_feedback_score']):
            continue

        query = df.at[i, 'instruction']
        model_answer = df.at[i, 'answer']

        # Randomly select a persona
        random_persona_name = random.choice(persona_names)
        persona_bio = personas[random_persona_name]

        # Get simulated human feedback
        feedback_response = get_human_feedback(random_persona_name, query, model_answer, persona_bio)
        feedback_dict_for_print = None

        try:
            feedback = json.loads(feedback_response)
            df.at[i, 'human_feedback'] = feedback
            df.at[i, 'human_feedback_score'] = feedback['score']
            df.at[i, 'human_feedback_analysis'] = feedback['analysis']
            feedback_dict_for_print = feedback
        except json.JSONDecodeError as e:
            print(f"--- JSON Decode Error for index {i}: {e} ---")
            print(f"--- Faulty Feedback String: {feedback_response} ---")
            df.at[i, 'human_feedback'] = f"Error: {e}. Original: {feedback_response}"

        df.at[i, 'persona_name'] = random_persona_name
        df.at[i, 'persona_bio'] = persona_bio

        newly_processed_count += 1 # Increment after processing

        # Every 100 newly processed samples, save the data and print checkpoint info
        if newly_processed_count > 0 and newly_processed_count % 100 == 0:
            print(f"\n--- Checkpoint after {newly_processed_count} new samples (current index {i}) ---")
            print(f"--- Query: {query} ---")
            print(f"--- Model Answer: {model_answer} ---")
            print(f"--- Persona: {random_persona_name} ({persona_bio}) ---")
            if feedback_dict_for_print:
                 print(f"--- Feedback (parsed): {feedback_dict_for_print} ---")
            else:
                 print(f"--- Feedback (raw/error): {feedback_response[:200]}... ---")
            print(f"Saving progress to {save_path}...")
            df.to_pickle(save_path)

    # Save the final data
    print(f"\nProcessing complete. Total newly processed: {newly_processed_count}. Saving final data to {save_path}...")
    df.to_pickle(save_path)
    print("Done.")