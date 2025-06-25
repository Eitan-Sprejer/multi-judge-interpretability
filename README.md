# Multi-Judge Interpretability

A framework for evaluating AI model outputs using multiple specialized judges and interpretability techniques. This project provides tools for training judge aggregation models, analyzing model behavior, and implementing custom evaluation criteria.

This work was done in the context of the [Apart x Martian Mechanistic Router Interpretability Hackathon](https://apartresearch.com/sprints/apart-x-martian-mechanistic-router-interpretability-hackathon-2025-05-30-to-2025-06-01). [Our submission](https://apartresearch.com/project/approximating-human-preferences-using-a-multijudge-learned-system-v3im) won 2nd place 🥈!

## 🌟 Key Features

- **Multiple Judge Integration**: Combine scores from specialized judges (e.g., harmlessness, privacy, factual accuracy)
- **Flexible Model Architecture**: Choose between a GAM model or a powerful MLP model
- **Robust Training Pipeline**: Including normalization, validation splits, and automatic checkpointing
- **Built-in Interpretability Tools**: Analyze judge contributions and model behavior
- **Error-Resilient Evaluation**: Implements exponential backoff for reliable scoring

## 📋 Prerequisites

- Python 3.10+
- PyTorch
- Access to Hugging Face Hub
- Access to Martian API
- CUDA-capable GPU (optional, but recommended for training)

## 🚀 Quick Start Guide

Follow these steps in order:

1. **Setup Environment**

   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/multi-judge-interpretability.git
   cd multi-judge-interpretability
   ```

   Moreover, you should follow the guides to install the Martian API SDK outlined in [here](https://github.com/withmartian/martian-sdk-python).

   Then you activate the environment and install the dependencies as expected:

   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Create Judges**

   Run `project/judging/create_judges.ipynb` to create 10 specialized judges:

   - Harmlessness Judge
   - Privacy Judge
   - Factual Accuracy Judge
   - Prompt Faithfulness/Relevance Judge
   - Calibration/Uncertainty Judge
   - Bias/Fairness Judge
   - Reasoning Consistency Judge
   - Discourse Coherence Judge
   - Conciseness/Redundancy Judge
   - Style/Formatting Judge

3. **Extract Base Dataset**

   Run `project/data_processing/create_ground_truth.ipynb` to:

   - Load the UltraFeedback dataset
   - Process and structure the data
   - Save as `question_and_answers.pkl`

4. **Create Ground Truth**

   Run `project/human_feedback_simulation/simulate_annotators.py` to:

   - Take the processed dataset
   - Generate reliable ground truth scores using human simulators
   - Add these scores to the dataset

5. **Generate Judge Scores**

   Run `project/inference/generate_dataset.ipynb` to:

   - Process each example through all 10 judges
   - Create final dataset with all scores

6. **Train/Evaluate Models**

   Choose one:

   - `project/inference/train_gam.py` to train a GAM model on the final dataset
   - `project/inference/eval_nn.py` to train a neural network on the final dataset

## 📖 Project Structure

![Screenshot from 2025-06-02 02-54-50](https://github.com/user-attachments/assets/1c755126-50bc-46d5-ad2e-7b5382e039e2)

## 🔧 Available Models

1. **GAM (Generalized Additive Model)**

   - Highly interpretable
   - Individual judge contribution analysis
   - Monotonicity constraints available

2. **Neural Network**
   - Potentially better performance
   - Handles complex judge interactions
   - Less interpretable than GAM

## 📚 Additional Notes

- Make sure to handle API keys and credentials properly
- The UltraFeedback dataset is only used for its structure and responses
- The actual ground truth comes from our human simulators
- Judge creation requires Martian API access
- Consider using error handling and retries for API calls
- The process can be computationally intensive, especially for large datasets
