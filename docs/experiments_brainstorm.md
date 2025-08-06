Taken from “Main”:

**Approach:**

1. **Systematic persona and metric design:**  
   1. Develop principled methods for selecting representative personas that span human preference space.  
   2. Design judge metrics across constrained preference dimensions (building on idea 1.a from the brainstorm).  
   3. Study which characteristics best approximate human preferences and their individual contributions.  
2. **Multi-model comparison framework:**  
   1. Compare GAMs, MLPs, and expanded model architectures (SVMs, random forests, decision trees)  
   2. Test against baselines: naive averaging, constitutional AI methods, existing ensemble techniques (we'd need some lit review)  
   3. Explore dynamic judge selection: learning which judges to use for different query types  
3. **Synthetic dataset creation and validation:**  
   1. Create our own high-quality synthetic preference dataset  
   2. Compare against UltraFeedback to understand quality differences  
   3. Validate synthetic personas against human annotations  
4. **Robustness and generalization experiments:**  
   1. Test if learned models generalize from N personas to a new persona OOD  
   2. Evaluate poisoning resistance: can one "bad preference" persona break the model?  
      1. If not, how many bad personas are needed for the model to break.  
         1. Grok4 is really good for creating bad personas.  
   3. Assess judge contamination: how do "bad judges" affect the system?  
   4. Map Pareto frontier for optimal judges-to-personas ratios  
   5. Preference decomposition: given human preferences, can we infer the underlying evaluation dimensions?  
5. Interpretability

Green: for great ideas, grey: not enough details or can be improved, red: not great ideas/ we have concerns

## Summary of the Most Relevant Ideas:

Judge Robustness: introducing a judge with an inverted metric, see how robust the system is to that judge.

Sensitivity analysis on Rubrics (test different prompting, different definitions of the same metrics, etc).

Are judges biases to evaluating their own responses positively?

can you predict the best judge per query in any better/smarter ways?

Simulated Human Feedback Robustness: Introduce a *troll persona,* see how that affects the score of the aggregator.

Validate with real or improved preference data (ground truth quality). (Ultrafeedback)

Improve and rerun the interp stuff that's already been done.

## Eitan

* From 1.a | 1.b:  
  * Ideation: Explore different approaches to simulating real human personas for feedback, and base the personas on that.  
  * Experimentation:   
    * Having extracted the relevant *preference dimensions*, construct N random personas with some combination of those dimensions and use them to generate human feedback.   
    * Construct the judges metrics basing them directly on each one of the *preference dimensions,* and use them as inputs for the aggregator function.  
  * Expected results:  
    * Increased performance on the reconstructed variance on the simulated human feedback (i.e. R2 score)  
    * Sanity check: Using only one persona for the feedback simulation should define an optimal linear combination of judge's metrics (bc the personas are constructed from linear combinations\!)  
* From 2\.  
  * Experimentation:  
    * Replace GAM and MLP with Random Forest, and compare performance. Interpret through feature relevance.  
    * Segregate the dataset according to different topics, and see if the optimal distribution changes. Then, use that info to train a dynamic judge selection process (2.c.)  
* From 3:  
  * .  
* From 4:  
  * Train the model on feedback from N different personas (predefined personas) and evaluate on another M different personas.  
  * 4.b. Devise a persona that red teams the system: it’s basically a troll. It rates bad responses well, and goodbad responses poorly. We then measure, for different contamination rates, how bad the trained judge aggregation function performs (performance in the sense of generalizing to *true* human preference (a non-contaminated test set)).  
  * 4.c. I can imagine, as a sanity check, introducing a judge that has a really poorly defined metric and checking that the learnt aggregator function does not give that judge any relevance.  
  * 4.d: Having pre-defined those sets of metrics for the judges, see if by combining two or more judges (combining the rubrics) we get a poorer performance than with all the judges.  
  * 4.e. Get a big dataset of human annotated responses, create a rather big list of different personas for the LLM to emulate. Get the annotations from all of those personas, and train a model to approximate the ground truth (human annotation) using the LLM personas annotations. Then, analyze the *feature relevance* on the model. Grab the features (i.e. the *personas*)  that could explain 90% of the recovered variance, and it could be said that we decomposed the human preferences into the main independent *persona dimensions*

## José

* **(1) Play around with preferences**  
  * Get one person preferences x multi-person preferences, test how this changes our model. Particularly, test “bad” person preferences (“bad” here means both “person who is willing to cause harm” and “person who just annotates data poorly”). The cool result would be: “what is the ‘diversity’ of person preferences we should have such that a bad person won’t ‘contaminate’ the system?”.   
    * **Technical details**: Simplifiedly, generate data as we did (‘personas’) and train the models, then compare with base truth (or interp, which could be as simple as “which of the judges is activating the most”). “All else equal, change only the training data”  
* **(2) Play around with judges**  
  * Analogous to the above but with judges: if we insert a ‘bad’ judge, what happens? Does our system learns to be robust? What is the condition such that our system remains robust?  
    * **Technical details**: “All else equal, change only the judges” which means “change the rubrics of the model”  
* Ideally, in (1) and (2), we would do experiments to get a "pareto frontier” of optimality x safety and understand it.   
* **(3) Metrics**  
  * (Maybe not worth it idea because it feels like we’re testing an ‘obvious’ thing, so more of a sanity check): “Metric” is a confusing thing: if we evaluate 10 different “facets” of a metric, let’s say, ‘quality’ is this the same as 10 different metrics (using as metric here our 10 facets)?Mathematically, both formulations seem to be equivalent, but do the conceptual distinction make a difference in the system?.  
    * **Technical details**: Don’t bother, dumb idea, obviously we’ll show those are the same if we use the same conceptual notion. I'm only leaving this idea here in case anyone thinks about the same thing  
* **(4) “Safety lock”**  
  * Can we apply a function h on top of our learned f\_theta and remain having better performance than avg/max/min/cond logic? Can we show that such a function h exists and is such that our system is “more safe”?  
    * **Technical details:** This should be done after “playing around with judges” and “playing around with preferences” (because then we understand the “pareto frontier” between optimality and safety). We test naive functions to be h (like an indicator that only outputs a value if the “safety score” is bigger than some threshold), or we get some mathematical framework (for ex., in routing papers, people show you can always construct some interpolation between routers and this is the “form” of the optimal router). Then we compare it with other baselines  
* **(5) “biased-to-itself model”**  
  * Judges are actually smaller LLMs, are they biased to evaluating more positively or negatively (prompt, answers) generated by the same LLM?  
  * More generally, we can do an LLM agnostic approach (mathematically, a tensor of vectors) or LLM dependent approach (mathematically, a tensor of matrixes). Are those the same empirically? (This is actually easier to test)  
  * **Technical details:** For the second, just test both cases (tables with different LLMs as rows x vectors that are “LLM agnostic”). I have a loose feeling we might be able to provide some *a priori* proofs about when both cases are equivalent here. For the first, if the result of the second bullet point is “yes, LLM agnostic is equivalent to LLM dependent regarding judge scores” then we already solved it. If it is not, then we experiment with different LLMs as judges and input-generators  
* **(6) Judges that aren’t LLMs**  
  * It would be safer and cheaper to use classical models or any other tooling as judges.   
  * **Technical details:** Get a bunch of classic models and test them as judges, see if they perform better than LLMs or if they at least beat LLMs in cost x safety x performance

## Augusto

1. Same content rubric, different text:  
   1. How does creating a different rubric that evaluates the same qualities, affect the final score? Does the way the rubric is written change the score significantly even when its content is essentially the same?  
2. Metrics:  
   1. Analyse which metrics have a high correlation, if 2 or more metrics correlate a lot, perhaps one of them should be dropped to avoid it influencing the routing decision too much (I need to word this idea better)

## Fernando

1. Given the way the rubrics were constructed, there’s a lot of low-hanging fruit to clear:  
   1. Explore a different set of rubrics to decompose the “quality” property of any given corpus of text. The one we ended up using in the hackathon was a working, initial solution but we can systematically evaluate whether pruning or changing some judges leads to better results. The, we plug in Eitan’s ideas for personas and we would be done  
2. We don’t run every judge on every query. Learn a small router from cheap signals (topic, length, embeddings) that picks a minimal, query-specific subset of judges (always include a safety judge) and the combiner. Two modes: (1) cluster top-k—per topic we keep the best judges; (2) sequential—start with 1–2 generalists, add another only if uncertainty actually drops, stop early. Constrain the router (simple rules/monotone), log which judges fired and why. Evaluate vs “use all” and global top-k on error, calibration, cost/latency, and safety overrides. Artifacts: cluster×judge heatmap, cost–accuracy curve, per-query usage log.  
3. .  
4. .  
5. Given that nobody else has spoken about interp meaningfully, here are some experiments:  
   1. Activation/path patching: Take two examples that differ on a target judge. Swap the hidden activations (or just that judge’s path) from the high one into the low one and see if the score jumps. If it jumps, that path matters. I’ll log the “flip sets” per judge and call out any small cluster of units that always does the work (e.g., a safety veto). Output: a tiny causal map per judge and a couple of concrete clamps/constraints.  
   2. Sparse-additive distillation (make the NN readable): Train a very small GAM/EBM to mimic the MLP. Goal: clean per-judge curves and a short list of interactions that explain most of the behavior. Use it as the explainer: flag weird non-monotone shapes, suggest which judges to drop or cap. I’ll report how many terms we need before the mimic is good enough, so we know the cost of simplicity.  
   3. Interaction surfaces (find synergy/antagonism): Fit an EBM/GA²M and pull the top judge×judge interactions. Then poke the MLP along those two axes to check they’re real. This surfaces patterns like “verbosity only helps when quality is already high” or “safety suppresses style past a threshold.” We turn those into simple routing rules and a couple of pairwise training constraints, then show before/after numbers.

## Narmeen

1. Validate with real or improved preference data (ground truth quality)

**Experiment A:** For example, the [UltraFeedback dataset](https://arxiv.org/abs/2310.01377) provides 5-objective absolute ratings (e.g. Instruction Following, Truthfulness, Helpfulness, etc.) for each prompt-response  
rlhflow.github.io  
. One approach is to treat one dimension (such as an Overall Quality score) as the ground truth to predict, and the other dimensions as inputs (simulated judge scores). This mimics a multi-judge scenario with a reliable “true” aggregate score.

2. Validate with baseline multi-dimensional judges

**Experiment B**:an alternative is to use the existing multi-dimensional LLM ratings (like GPT-4 ratings in UltraFeedback) as a proxy for human preferences. Prior research has demonstrated the effectiveness of multi-objective reward models trained on such data, achieving state-of-the-art performance on preference benchmarks ([Blog](https://rlhflow.github.io/posts/2024-05-29-multi-objective-reward-modeling/#:~:text=trained%20an%20ArmoRM%20with%20Llama3,34B%20by%20a%20margin), [Paper](https://arxiv.org/abs/2406.12845)). This experiment would connect our work to that trend, showing that our learned judge aggregator can match or exceed single-model judges (even powerful ones like GPT-4) by intelligently combining smaller specialized metrics.

3. Robustness to biased persona among the mixed personas impact on the aggregator judge

**Experiment C:** Simulate an extreme scenario where one persona has systematically poor preferences (e.g. always gives high scores to unsafe or factually wrong answers). Train an aggregator on a dataset heavily or entirely influenced by this persona. Measure performance on normal data and inspect if the model indeed learned the bad preferences (e.g. it might give high aggregate scores to answers that only the bad persona would approve). This checks the model’s robustness to a biased or low-quality preference signal.

When combining multiple personas, we anticipate a form of preference averaging: the learned function will attempt to minimize error across conflicting preferences, possibly resulting in intermediate behavior. This could dilute extreme biases, but it may also fail to satisfy any one persona perfectly. We will look for signs of preference conflicts – for example, larger prediction errors on any single persona’s data when using the combined model. If one “bad” annotator’s data is mixed in, we can observe whether their influence *poisons* the aggregate model or is drowned out by others. How do we fix this, can we add weights that are proportional to the dataset size?

4. Robustness to biased Judge

   

**Experiment D:**

-  Introduce a Biased Judge**:** Modify one of the existing judge models or scoring rubrics to be systematically flawed. For example, create a judge that rewards irrelevant or harmful content (contrary to the others). This could be done by flipping a rubric (so high scores mean bad quality) or by adding random noise to its outputs. Alternatively, one judge’s rubric could ignore crucial aspects (e.g. a judge that never penalizes factual errors). Keep the other judges and the ground-truth scoring the same as in the original setup.

- Train Aggregator with Outlier Judges: Train the aggregator on a dataset that includes the flawed judge’s scores as part of the input. (The ground truth remains the “correct” preference signal – e.g. a simulated or human score that does *not* share the judge’s flaw.)

- Baseline Comparison: Compare multiple aggregation strategies: (a) the learned aggregator *f* and (b) a naive combination (such as a simple average of judges or a majority vote). Evaluate both on a test set. The expectation is that the naive aggregator will be noticeably degraded by the bad judge – e.g. averaging in an adversarial judge could pull the overall score in the wrong direction– whereas the learned *f* might recognize and discount the unreliable judge to minimize error.

- Inspect Aggregator Behavior: Check the learned model’s parameters or partial dependence for the biased judge’s input. Ideally, we would see that the aggregator has assigned *low or even negative influence* to that judge. For example, in a linear or GAM model, the coefficient or partial dependence curve for the bad judge’s score might be flat or decreasing, indicating the model has learned to invert or ignore that judge’s evaluations. If using an interpretable model like GAM, we can directly plot the effect function for the bad judge to confirm this.

- Multiple Bad Judges Scenario: If resources allow, extend to scenarios with multiple biased judges or varying degrees of judge accuracy. This will test the limits: how many corrupted inputs can the system tolerate before performance suffers? It can help identify if certain aggregation methods (e.g. median vs. mean vs. learned model) are more robust to outliers. For instance, we might find that the learned aggregator behaves somewhat like a weighted median, effectively sidelining extreme judges.

What would be good results to show: “biases of individual judges can cancel out” if combined intelligently  
.A concrete result could be: “With a deliberately mis-specified Judge \#3, the learned aggregator’s error increased only modestly (e.g. \+5%), whereas the simple average’s error jumped significantly (e.g. \+20%)

**Experiment E: Benchmarking against MoJ**  
[Mixture of Judges (MoJ)](https://arxiv.org/pdf/2409.20370) proposes using a mixture of reward models (judges) during RLHF rather than a single reward model.

Each judge $$ J\_k $$ evaluates responses on a particular axis (e.g., helpfulness, harmlessness).

A learned gating function $$ g(x, a) \\in \\Delta^K $$ outputs weights over the judges for a given (prompt, answer).

The reward is:

$$ R(x, a) \= \\sum\_{k=1}^{K} g\_k(x, a) \\cdot J\_k(x, a)$$

This mixture is context-aware, i.e., the weights depend on the prompt and the answer.

Why is this good?

\- Prevent reward hacking by making the reward adaptive    
\- Allow fine-grained control over different preferences    
\- Improve robustness to judge biases  

Experiments:

\- Used as the reward in RLHF training    
\- The mixture gives better performance and fewer safety violations  

Can You Replace MoJ with Your Aggregator?

Your learned aggregator is:

$$f\_\\theta(J\_1(x, a), \\dots, J\_K(x, a))$$

This is a fixed, learned function mapping judge outputs to a scalar reward.

While Xu et al. use a contextual mixture, your aggregator could act as:  
\- A fixed (but learned) mixture: interpretable and simpler to train  
\- A lightweight plug-in for systems that can't afford online learning

Setup: Keep the same judge outputs:  

$$J\_k(x, a) \\in \\mathbb{R}$$

Instead of learning a context-aware gating function, you learn a static aggregator:

$$R(x, a) \= f\_\\theta(J\_1(x, a), \\dots, J\_K(x, a))$$

The function $$ f\_\\theta $$ can be a GAM, MLP, or shallow neural network.

Use Cases

1\. For RLHF Training

\- Replace their reward function $$ R(x, a) $$  with your $$ f\_\\theta $$ in PPO/fine-tuning loop.  
\- Evaluate policy behavior trained with your aggregator vs. theirs.

2\. As a Reward Model for Ranking Responses

\- Use your aggregator to score/rank multiple model outputs for a prompt.  
\- Compare results with MoJ on benchmarks like UltraFeedback, TREx, etc.- Mixture-of-Judges gating  