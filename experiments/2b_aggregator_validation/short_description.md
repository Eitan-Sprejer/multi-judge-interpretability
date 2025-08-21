When reporting the baseline performance of our aggregator, we report ~0.58 R2 score for our aggregator. This low-score might as well be an artifact of having simulated human preference data that's too varied. That is why we should evaluate the performance of our aggregator in less varied data.
In order to do this, I propose evaluating on:

1. Ultrafeedback's "overall_score", which is the score assigned by GPT-4 after a review.
2. Single-persona simulated preference data.