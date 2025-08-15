JUDGE_MAPPING = {
    'harmlessness': 'harmlessness-judge',
    'privacy': 'privacy-judge',
    'factual_accuracy': 'factual-accuracy-judge',
    'prompt_faithfulness': 'prompt-faithfulness-relevance-judge',
    'calibration': 'calibration-uncertainty-judge',
    'bias_fairness': 'bias-fairness-judge',
    'reasoning_consistency': 'reasoning-consistency-judge',
    'discourse_coherence': 'discourse-coherence-judge',
    'conciseness': 'conciseness-redundancy-judge',
    'style_formatting': 'style-formatting-judge',
}

JUDGE_IDS = list(JUDGE_MAPPING.values())
