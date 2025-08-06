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

### Summary

Please provide an **abstract-level summary** of your project **(max. 1000 characters)**.

Include the following:

* What problem are you addressing?  
* How are you addressing it? What is your key contribution or approach?  
* Why is your intervention novel and important?  
* What evidence will you provide to validate your claims *(this is necessarily speculative, but requires understanding the experiments you will run, and how to quantify their uncertainty)*?

---

We address the fundamental problem of approximating diverse human preferences using multi-judge AI systems. Our approach extends beyond simple aggregation by learning interpretable functions that optimally combine judge evaluations across different metrics and personas. We propose to: (1) systematically design judge metrics that capture preference dimensions, (2) develop methods for selecting diverse, representative personas that span human preference space, (3) compare our learned aggregation approach against existing ensemble methods. This is novel because current work uses fixed aggregation rules or single judges, missing the opportunity to learn optimal combinations that better approximate human preferences while remaining interpretable. Our experiments would demonstrate how learned aggregators can generalize to new personas, remain robust to judge contamination, and reveal which evaluation dimensions matter most for different preference profiles—critical for building truly aligned AI systems.

### High-level background

What are the 3-5 works that are foundational to the problem you’re trying to address? What is your work based on? How does your work build off prior work? How does your work compare to other	 works with similar goals?

---

* **Constitutional AI/RLHF papers** \- Establishing the paradigm of using AI feedback for alignment  
* **UltraFeedback dataset paper** \- Provides the preference data infrastructure we build upon  
* **Expert Orchestration vision papers** \- Framework for using multiple specialized models instead of monolithic AI  
* **Multi-agent evaluation literature** \- Prior work on ensemble methods for model evaluation  
* **Interpretable ML (GAMs) papers** \- Foundation for our interpretable aggregation approach

### Detailed technical background

What are the 1-2 works that your project directly builds on? For these works, provide detailed context, including:

* Core concepts which will also be used in your work,  
* Specific results or methods you’ll be extending,  
* Limitations or gaps which your work will improve upon.

This section should provide enough detail that someone unfamiliar with these specific papers can understand your proposed methodology, and how it differs from these related works.

*NOTE: One approach is to find the most similar existing study, and assess whether your proposed idea is still valuable in light of this contribution. If you decide your work is an improvement, what about the original study makes it insufficient for addressing the problem/question that you’ve identified?*

---

\[To be filled after literature review \- will focus on most similar multi-judge aggregation work and explain how our learned approach improves upon fixed aggregation strategies\]

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

1. **Systematic persona and metric design:**  
   * Develop principled methods for selecting representative personas that span human preference space.  
   * Design judge metrics across constrained preference dimensions (building on idea 1.a from the brainstorm).  
   * Study which characteristics best approximate human preferences and their individual contributions.  
2. **Multi-model comparison framework:**  
   * Compare GAMs, MLPs, and expanded model architectures (SVMs, random forests, decision trees)  
   * Test against baselines: naive averaging, constitutional AI methods, existing ensemble techniques (we'd need some lit review)  
   * Explore dynamic judge selection: learning which judges to use for different query types  
3. **Synthetic dataset creation and validation:**  
   * Create our own high-quality synthetic preference dataset  
   * Compare against UltraFeedback to understand quality differences  
   * Validate synthetic personas against human annotations  
4. **Robustness and generalization experiments:**  
   * Test if learned models generalize from N personas to a new persona OOD  
   * Evaluate poisoning resistance: can one "bad preference" persona break the model?  
   * Assess judge contamination: how do "bad judges" affect the system?  
   * Map Pareto frontier for optimal judges-to-personas ratios  
   * Preference decomposition: given human preferences, can we infer the underlying evaluation dimensions?

**Validation:**

* Performance metrics across model architectures  
* Interpretability analysis via partial dependencies and feature importance  
* Robustness metrics under adversarial conditions  
* Generalization accuracy on held-out personas  
* Comparison with human preference ground truth

**Key Assumptions:**

* A set of simulated diverse personas can approximate the human preference distribution  
* Judge specialization improves over monolithic evaluation  
* Interpretable models can achieve competitive performance

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

### Potential applications & practical uses

Why are your findings and/or outputs valuable **(max. 1000 characters)**?

Will your findings make models more robust or safe? What failure modes can you address? Is there a **need** that your project would resolve? If so, what is it, and how do you **know** that need exists?

Are there current use-cases which can be made better (efficiency, robustness, accessibility) with your output? Does your work open up any new possibilities? If so, justify why this is the case.

---

Our work directly improves AI safety by making multi-judge evaluation systems more robust against single-point failures and adversarial judges. Current RLHF systems using single evaluators are vulnerable to reward hacking; our approach reduces this risk through learned, interpretable aggregation. This addresses the critical need for scalable oversight as AI systems become more capable. Our work opens new possibilities: enabling personalized AI systems that can adapt to diverse user preferences while maintaining safety guarantees, creating more nuanced evaluation frameworks that capture the full spectrum of human values, and providing interpretable tools to understand which aspects of AI behavior matter most to different communities. By demonstrating that small, interpretable models can effectively orchestrate multiple judges, we validate the Expert Orchestration paradigm as a viable alternative to monolithic AI systems.

### Risks, uncertainties, monitoring, & mitigation

It can be incredibly useful to red-team your idea. Ask yourself *“What is the best reason to not support this work?”* These could include uncertainties in your assumptions, amount of resources required, or something else.

Once you’ve thought of these, address what you will do to decrease their likelihood, monitor whether or not they are becoming a more significant roadblock, and what your contingency plan is in the face of these setbacks.

---

1. **Time constraints** \- August 22 deadline is tight  
* *Mitigation:* Prioritize core experiments, prepare incremental submission  
2. 

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

### Proposed timeline/milestones

Propose a timeline for your project, including at least 3 milestones. For each milestone, use the template provided below.

---

Milestone Template: 

\------------------- 

Milestone: 

Description: 

Target Completion Date: 

Key Deliverables:

### Anything else you would like us to know about?

Feel free to leave this blank.

---

