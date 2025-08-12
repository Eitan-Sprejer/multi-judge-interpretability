# Apart Project Proposal

This application is designed to help our team understand your research. When you are ready to submit your proposal to Apart, use this [form](https://apartresearch.notion.site/19ffcfd1de9d81a9a968eda77f3465f5?pvs=105).

When writing this proposal, keep the following ideas in mind:

* **Motivation** — e.g. the intervention, approach, or tool is a demonstrable solution to an existing need, or a provable improvement to an established approach/use-case.  
    
* **Justification** — for each design decision, explain why the choice you’ve made is the most logical one with clearly marked external sources (citations, hyperlinks, footnotes).  
    
* **Accessibility** — avoid jargon and assumption of knowledge wherever possible.  
    
* **Specificity** — clarity doesn’t mean oversimplification. Provide concrete details to demonstrate subject matter expertise.

### Target output

What will you be working towards during your time in the fellowship **(max. 150 characters)**?

Typically, the core output is an academic paper, in which case you should note the **target venue**, i.e. the name of the conference, workshop, or journal you will be submitting to.

Include submission deadlines, or best approximations, if known.

---

Paper on **Neurips' Interp Workshop**. Submission deadline **Aug. 22nd**

### 

### Summary

Please provide an **abstract-level summary** of your project **(max. 1000 characters)**.

Include the following:

* What problem are you addressing?  
* How are you addressing it? What is your key contribution or approach?  
* Why is your intervention novel and important?  
* What evidence will you provide to validate your claims *(this is necessarily speculative, but requires understanding the experiments you will run, and how to quantify their uncertainty)*?

---

Current approaches to evaluating AI outputs rely on either single judges, which provide limited perspectives, or naive averaging of multiple judges, which assumes all evaluation dimensions are equally important. However, human preferences are diverse: different people weight safety, helpfulness, and accuracy differently depending on context and values. We address this fundamental challenge by learning interpretable aggregation functions that capture these varying preference profiles when combining multiple judge evaluations. We propose to: (1) test robustness against deliberately contaminated judges, (2) validate using UltraFeedback's multi-dimensional ratings as ground truth, (3) compare our learned aggregation approach against existing ensemble methods like Mixture of Judges. This is novel because current work uses fixed aggregation rules or single judges, missing the opportunity to learn optimal combinations that better approximate human preferences while remaining interpretable. Our experiments will demonstrate how learned aggregators can remain robust to judge contamination, generalize to new personas, and reveal which evaluation dimensions matter most for different preference profiles, critical for building truly aligned AI systems.

### 

### High-level background

What are the 3-5 works that are foundational to the problem you’re trying to address? What is your work based on? How does your work build off prior work? How does your work compare to other works with similar goals?

---

* [**Constitutional AI: Harmlessness from AI Feedback**](https://arxiv.org/abs/2212.08073) \- Bai et al.  
* [**UltraFeedback: Boosting Language Models with Scaled AI Feedback**](https://arxiv.org/abs/2310.01377) \- Cui et al.  
* [**The Perfect Blend: Redefining RLHF with Mixture of Judges**](https://arxiv.org/abs/2409.20370) \- Xu et al**.**  
* [**Expert Orchestration: A Framework for Composable AI Systems**](https://arxiv.org/html/2506.00051v1) **\-** Kulkarni et al**.**  
* [**Intelligible Models for Classification and Regression**](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd12.pdf) **\-** Lou et al.  
* [**Multi-Agent-as-Judge: Aligning LLM-Agent-Based Automated Evaluation with Multi-Dimensional Human Evaluation**](https://arxiv.org/pdf/2507.21028) \- Chen et al.  
* [**A Survey on LLM-as-a-Judge**](https://arxiv.org/html/2411.15594v4) \- Gu et al.  
* [**Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**](https://arxiv.org/pdf/2306.05685) \- Zheng et al.  
* [**Snorkel: Rapid Training Data Creation with Weak Supervision**](https://arxiv.org/pdf/1711.10160) \- Alexander et al.  
* [**Improving LLM-as-a-Judge Inference with the Judgment Distribution**](https://arxiv.org/pdf/2503.03064) \- Victor et al.

### Detailed technical background

What are the 1-2 works that your project directly builds on? For these works, provide detailed context, including:

* Core concepts which will also be used in your work,  
* Specific results or methods you’ll be extending,  
* Limitations or gaps which your work will improve upon.

This section should provide enough detail that someone unfamiliar with these specific papers can understand your proposed methodology, and how it differs from these related works.

*NOTE: One approach is to find the most similar existing study, and assess whether your proposed idea is still valuable in light of this contribution. If you decide your work is an improvement, what about the original study makes it insufficient for addressing the problem/question that you’ve identified?*

---

**Expert Orchestration \- Quirke et al. (2024)**

Expert Orchestration proposes a paradigm shift from monolithic AI systems to coordinated networks of specialized models. The framework identifies three core primitives: **Judges** (evaluate model outputs across different dimensions), **Routers** (select appropriate models for tasks), and **Orchestrators** (coordinate multi-step workflows). This addresses fundamental limitations of current AI systems: high computational costs, lack of interpretability, and vulnerability to single points of failure.

**Core concepts we directly implement:** Multi-judge evaluation systems, preference aggregation as a core safety mechanism, interpretable alternatives to monolithic models. **Specific methods we extend:** While Expert Orchestration establishes the judge primitive conceptually, it doesn't address the critical question of optimal judge combination. Most implementations use simple heuristics (averaging, maximum, conditional logic) which are vulnerable to contaminated judges and fail to capture complex preference relationships.

**Key gap our work addresses:** Expert Orchestration identifies judge aggregation as essential but provides limited guidance on implementation. Our learned aggregation approach f\_θ fills this gap by providing a principled, interpretable method for combining multiple judges that can adapt to different preference profiles while remaining robust to adversarial inputs. This validates the Expert Orchestration vision by demonstrating that small, specialized models can effectively coordinate to achieve performance comparable to monolithic systems.

### Proposed methodology

What is your approach, and how will you validate it?

* State your hypothesis and any novel techniques you’ll introduce  
* Describe experimental design, baselines, and method of comparison  
* Provide key assumptions your approach relies on  
* Outline explicitly how you’ll *confirm* or *deny* your hypothesis

See the [Methodology Worksheet](?tab=t.4j72f4qd5jx2) tab for additional guidance.

---

**Hypothesis:** Learned aggregation functions can better approximate diverse human preferences than fixed combination rules, while maintaining interpretability and robustness to adversarial inputs.

**Approach:**

We propose four experimental tracks to systematically validate our hypothesis. We will prioritize Tracks 1 and 2 as core contributions, with Tracks 3 and 4 as secondary/parallel efforts:  
**Important:** See the **\[Work in Progress\] Detailed Experiments** tab for more details on the experiments.

**Track 1: Robustness Analysis (Priority)**

* **Judge Contamination:** Introduce deliberately flawed judges (inverted metrics, random noise, safety-blind rubrics) and measure if learned aggregators assign near-zero weights to bad judges while maintaining performance  
* **Persona Poisoning:** Include "troll" personas that systematically misrate responses at varying contamination rates to identify failure thresholds  
* **Rubric Sensitivity:** Test semantic robustness by evaluating same content with differently phrased but equivalent rubrics

**Track 2: Ground Truth Validation (Priority)**

* **UltraFeedback Integration:** Use UltraFeedback's multi-dimensional ratings as more realistic ground truth, treating Overall Quality as target and other dimensions as judge's scores (aggregator function inputs).  
* **Baseline Comparison:** Compare against single-judge baselines and naive averaging methods

**Track 3: Architectural Comparisons (Secondary)**

* **Mixture of Judges (MoJ) Benchmark:** Compare our static learned aggregator against MoJ's dynamic context-aware gating on performance, computational cost, and interpretability  
* **Judge Self-Bias Analysis:** Test if LLM judges favor responses from same model family and develop correction methods

**Track 4: Interpretability Deep Dive (Secondary)**

* **Learned Function Analysis:** Systematic interpretability of GAM and MLP aggregators using partial dependence plots and feature importance  
* **Sparse Additive Distillation:** Train minimal GAM to mimic MLP behavior, identifying key judges and interaction effects  
* **Framing Effects and Bias Transfer:** Following [Christian et al. (2024),](https://arxiv.org/pdf/2506.07326v1) test whether learned aggregators inherit differential sensitivity to positive/negative sentiment tokens and systematic identity group devaluation, using AFINN-111 lexicon for sentiment analysis

**Validation:**

* **Robustness:** Quantified improvement over baseline aggregation methods under adversarial conditions  
* **Performance:** Competitive accuracy on established preference datasets compared to single-judge systems  
* **Efficiency:** Computational cost analysis and inference time comparisons  
* **Interpretability:** Clear identification of judge contributions and decision-making patterns

**Key Assumptions:**

* A set of simulated diverse personas can approximate the human preference distribution  
* Judge specialization improves over monolithic evaluation  
* Interpretable models can achieve competitive performance

### 

### Preliminary experiments

What evidence do you have that your approach is worth investigating further? Briefly describe any initial experiments or analysis you’ve conducted and how the results **support** your proposed direction.

Ideally, these experiments should demonstrate that your key assumptions (mentioned above) are more likely to be true.

*NOTE: If continuing from an Apart Studio blogpost and you think your findings fit this description, summarize them here and include a link to your blogpost. Additional experimentation is also welcome.*

---

Our hackathon experiments (see blogpost [here](https://www.lesswrong.com/collaborateOnPost?postId=uKGJDch2Pf4CywQMy&key=4b92ea84fdaa13051bf8b807440d43)) demonstrate promising results:

* Trained GAM and MLP models on 10,000 examples  
* Achieved 57% R² (vs 24% baseline), 1.35 MAE (vs 1.83)  
* Partial dependency analysis reveals model sensitivity to non-safety vs safety judges  
* Infrastructure ready for scaled experiments

These results support our hypothesis that learned aggregation outperforms fixed rules and suggest rich interpretability potential.

### 

### Potential applications & practical uses

Why are your findings and/or outputs valuable **(max. 1000 characters)**?

Will your findings make models more robust or safe? What failure modes can you address? Is there a **need** that your project would resolve? If so, what is it, and how do you **know** that need exists?

Are there current use-cases which can be made better (efficiency, robustness, accessibility) with your output? Does your work open up any new possibilities? If so, justify why this is the case.

---

Our work directly improves AI safety by making multi-judge evaluation systems more robust against single-point failures and adversarial judges. Current RLHF systems using single evaluators are vulnerable to reward hacking; our approach reduces this risk through learned, interpretable aggregation. This addresses the critical need for scalable oversight as AI systems become more capable. Our work opens new possibilities: enabling personalized AI systems that can adapt to diverse user preferences while maintaining safety guarantees, creating more nuanced evaluation frameworks that capture the full spectrum of human values, and providing interpretable tools to understand which aspects of AI behavior matter most to different communities. By demonstrating that small, interpretable models can effectively orchestrate multiple judges, we validate the Expert Orchestration paradigm as a viable alternative to monolithic AI systems.

### 

### Risks, uncertainties, monitoring, & mitigation

It can be incredibly useful to red-team your idea. Ask yourself *“What is the best reason to not support this work?”* These could include uncertainties in your assumptions, amount of resources required, or something else.

Once you’ve thought of these, address what you will do to decrease their likelihood, monitor whether or not they are becoming a more significant roadblock, and what your contingency plan is in the face of these setbacks.

---

1. **Time constraints** \- August 22 deadline is tight  
     
   * *Mitigation:* Prioritize Track 1 (Robustness) and Track 2 (UltraFeedback validation) as core contributions, defer some Track 3 experiments if needed  
   * *Monitoring:* Weekly progress reviews, milestone tracking  
   * *Contingency:* Submit preliminary results with clear follow-up plan

   

2. **UltraFeedback limitations** \- Dataset may not represent true human preference diversity  
     
   * *Mitigation:* Acknowledge limitations explicitly, complement with synthetic persona validation  
   * *Monitoring:* Compare UltraFeedback results against our synthetic personas for consistency  
   * *Contingency:* Pivot to multi-source validation if UltraFeedback proves insufficient

   

3. **Judge contamination detection failure** \- Learned aggregators might not identify bad judges  
     
   * *Mitigation:* Test multiple contamination types, implement explicit anomaly detection  
   * *Monitoring:* Track learned weights and performance degradation curves  
   * *Contingency:* Fall back to ensemble methods with explicit outlier removal

### 

### Capacity

If accepted, what level of commitment would you be able to **guarantee**, barring unseen circumstances? Answer in hours per week for each individual team member.

Are you currently waiting to hear from other opportunities (jobs, fellowships, etc.)? If so, what are they, and where are you in this process? If you are accepted, how would that impact your involvement with this project?

---

* Eitan:  
  * 5-10hs (at least for the next month or so…)  
* José  
  * 10h minimum, 20h maximum  
* Fernando  
  * 10h minimum, 20h maximum  
* Augusto  
  * up to 15 hrs

### 

### Proposed timeline/milestones

Propose a timeline for your project, including at least 3 milestones. For each milestone, use the template provided below.

---

**Milestone 1: Robustness Framework**

Description: Implement and test judge contamination experiments across all three contamination types (inverted, random, safety-blind). Establish baseline performance metrics and contamination detection capabilities.

Target Completion Date: August 13th, 2025

Key Deliverables:

* Contaminated judge test suite with multiple judge types  
* Performance degradation curves for different contamination rates  
* Learned weight analysis demonstrating bad judge detection  
* Comparison against naive averaging baselines  
* Framing Effects and Bias Transfer (from track 4\)

\-------------------

**Milestone 2: UltraFeedback Integration & Validation**

Description: Integrate UltraFeedback dataset as ground truth, train aggregators on multi-dimensional ratings, and benchmark against single-judge baselines.

Target Completion Date: August 16th, 2025

Key Deliverables:

* UltraFeedback integration pipeline  
* GAM and MLP aggregators trained on multi-dimensional ratings  
* Performance comparison with single-judge baseline  
* Ablation study showing dimension importance

\-------------------

**Milestone 3: Secondary Experiments & Paper Draft**

Description: Complete architectural comparisons (MoJ benchmark), finalize interpretability analysis, and prepare paper submission with all experimental results.

Target Completion Date: August 20th, 2025

Key Deliverables:

* Architectural comparison results (MoJ, self-bias analysis)  
* Comprehensive interpretability analysis (partial dependence plots, feature importance)  
* Complete paper draft with all experimental results

**Milestone 4: Paper Submission**

Description: Finish the paper, with the feedback from Narmeen and Apart.

Target Completion Date: August 22th, 2025

Key Deliverables:

* Submission to NeurIPS Interpretability Workshop

### Anything else you would like us to know about?

Feel free to leave this blank.

---

