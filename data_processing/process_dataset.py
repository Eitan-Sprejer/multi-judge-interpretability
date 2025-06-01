from datasets import load_dataset
import json

def process_example(example):
    """
    Processes a single example from the UltraFeedback dataset.
    Extracts the instruction and responses from the primary completion list.
    """
    instruction = example['instruction']

    output = {'source': example['source'], 'instruction': instruction}
    
    return output

def main():
    # Configuration
    dataset_name = "openbmb/UltraFeedback"
    split_name = "train"  # You can change this to other available splits if needed
    # Set to a number (e.g., 100) for quick testing on a subset, None to process the whole dataset
    sample_size = 10000
    
    output_jsonl_file_path = "dataset/processed_ultrafeedback_for_inference.jsonl"

    # Load the dataset
    print(f"Loading dataset '{dataset_name}' (split: '{split_name}')...")
    try:
        ds = load_dataset(dataset_name, split=split_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have an internet connection, the 'datasets' library is installed (pip install datasets),")
        print("and the dataset name/split are correct.")
        return

    if sample_size is not None:
        print(f"Selecting a subset of {sample_size} examples for processing.")
        print("Shuffling the dataset before selecting a sample...")
        ds = ds.shuffle(seed=42)  # Add a seed for reproducibility
        ds = ds.select(range(min(sample_size, len(ds))))

    print(f"Dataset loaded. Number of examples to process: {len(ds)}")
    if len(ds) > 0:
        print(f"Original features: {ds.features}")
        print(f"First original example structure (completions field): {ds[0]['completions']}")


    print("\nProcessing dataset...")
    
    # Determine columns to remove. We want to keep only what process_example returns.
    # It's safer to remove only columns that are present.
    original_columns = ds.column_names
    columns_to_remove_potentially = ['models', 'completions', 'correct_answers', 'incorrect_answers']
    actual_columns_to_remove = [col for col in columns_to_remove_potentially if col in original_columns]

    processed_ds = ds.map(
        process_example,
        remove_columns=actual_columns_to_remove
    )

    print("\nProcessed dataset features:")
    print(processed_ds.features)
    
    if len(processed_ds) > 0:
        print("\nFirst example of processed dataset:")
        print(processed_ds[0])
    else:
        print("Processed dataset is empty.")


    # --- Saving the processed dataset ---

    # Save to JSON Lines (JSONL) file
    print(f"\nSaving processed dataset to JSONL file: '{output_jsonl_file_path}'...")
    try:
        with open(output_jsonl_file_path, 'w', encoding='utf-8') as f:
            for example_idx, example in enumerate(processed_ds):
                try:
                    f.write(json.dumps(example) + '\n')
                except TypeError as te:
                    print(f"TypeError serializing example {example_idx} to JSON: {te}. Example: {example}")
                    # Handle or skip problematic examples if necessary
        print(f"Successfully saved to '{output_jsonl_file_path}'")
    except Exception as e:
        print(f"Error saving dataset to JSONL: {e}")

    print("\nFinished.")
    print(f"The output JSONL file is at: {output_jsonl_file_path}")

if __name__ == "__main__":
    main() 