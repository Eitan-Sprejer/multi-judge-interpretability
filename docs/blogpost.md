# Approximating Human Preferences Using a Multi-Judge Learned System

**Authors:** José Faustino, Eitan Sprejer, Fernando Avalos, Augusto Bernardi  
**Date:** July 31st, 2025  
**Reading Time:** 16 minutes  
**Source:** LessWrong, Apart Research  

---

## TL;DR

We present a conceptual discussion and loose formalism regarding Expert Orchestration, emphasizing on judges. We motivate the problem of finding the best way of combining multiple judges scores and present a solution to it: learning the function. Then, we present the architecture and low level details of the experiments we did to learn this function. We conclude by discussing scalability concerns, rooms for improvement, further work and safety considerations of our work.

---

## Epistemic Status

This post emerged from a 48-hour Apart Hackathon on Expert Orchestration where our team achieved second place. The content is roughly a translation of our original submission into blog format. 

- **Original submission:** [Link]
- **GitHub repository:** [Link] 
- **Primary author:** José Faustino (except "Further experiments" section by Fernando Avalos)

---

## Acknowledgements

We thank Apart Research and Martian for hosting the hackathon that initiated this project and for providing insightful feedback on our initial draft. Without their support, this work would not exist.

---

## Introduction

This work introduces and formalizes key concepts around Expert Orchestration, with a specific focus on judge systems for AI evaluation.

### Expert Orchestration Vision

The current AI paradigm relies on **Monolithic AIs** - large, expensive, hard-to-train single models like:
- OpenAI o3
- Anthropic's Claude 4  
- DeepMind's Gemini 2.5 Pro

**Problems with Monolithic AIs:**
- Only a few players can train these models
- Users don't understand how these models 'think'
- High computational and financial costs

**The Alternative: Expert Orchestration (EO)**

EO proposes using:

1. **Judges**: Specialized models that evaluate other models across different capabilities (domain expertise, ethics, bias, etc.)

2. **Routers**: Intelligent systems that choose the best model for specific tasks based on judge evaluations

**Potential Benefits:**
- Lower costs (using smaller models)
- Better interpretability
- Improved safety properties

---

## Core Concepts

### What is a Metric?

Consider evaluating models on "mathematics expertise." This broad capability includes:

- Knowledge of results and theorems, and how to connect them
- Explaining ideas conceptually with intuition
- Organizing arguments in a clean, beautiful manner

**Key Question:** What metric decomposition allows optimal judge system performance? Should judges evaluate:
- Single high-level metrics ("mathematics expertise")?  
- Detailed sub-metric lists?

For this work, we assume a set of metrics **E = {Me₁, Me₂, ..., Meₙ}** where each metric can be high-level ("mathematics expertise") or specific (individual sub-skills).

### What are Judges?

**Two Possible Judge Architectures:**

1. **LLM-Dependent Judges:** Receive prompt + LLM-specific answer → output score based on which LLM generated the answer

2. **LLM-Agnostic Judges:** Receive prompt + answer pair → output quality score independent of the generating LLM

**Implementation Considerations:**
- Should judges be implemented as LLMs?
- Single LLM or multiple LLMs per judge?  
- Following EO vision: use small, interpretable judges

### Judge Formalization

**Mathematical Definition:**

Let **X** = space of prompts, **A** = space of possible answers

A judge is a function: **J: X × A → ℝᵈ**

Where **d** = number of evaluation dimensions

**Examples:**
- d = 3: evaluating domain expertise, ethics, writing skills
- d = 1: evaluating single metric
- d = 2: evaluating "rigor" and "intuition" for mathematics

---

## The Judge Joining Problem

### Why Multiple Judges?

**Safety Benefits:**
- **Reduces Gaming:** Single judges can be exploited by models learning specific "hacks"
- **Bias Reduction:** Multiple independent judges help biases "cancel each other out"

### The Core Challenge

Given **t** judges: **J₁, J₂, ..., Jₜ** where **Jₖ: X × A → ℝ**

**Question:** How do we combine judge scores optimally?

**Mathematically:** Choose function **f: ℝᵗ → ℝ** that takes judge evaluations **(J₁(x,a), J₂(x,a), ..., Jₜ(x,a))** and outputs single score.

**Common Approaches:**
- **Average:** f = (J₁ + J₂ + ... + Jₜ)/t
- **Maximum:** f = max{J₁, J₂, ..., Jₜ}
- **Conditional logic:** If-then rules based on judge scores

**Problem with Maximum:** A single "contaminated" judge giving high scores to bad answers can compromise the entire system.

---

## Our Solution: Learning the Judge Aggregator

### The Approach

**Assumption:** We have ground truth function **f*: X × A → ℝ** that correctly scores each (prompt, answer) pair.

**Method:** Parameterize combination function as **fθ** and optimize parameters **θ** such that:

**fθ(J₁(x,a), J₂(x,a), ..., Jₜ(x,a)) ≈ f*(x,a)**

**Optimization Problem:**
```
min θ ∈ parameters L(fθ(J₁, J₂, …, Jₜ), f*)
```

Where **L** is a loss function (e.g., mean squared error).

### High-Level Architecture

**Implementation Steps:**

1. **Data Collection:** Gather (prompt, answer) pairs
2. **Ground Truth Generation:** Create (prompt, answer, score) triples using human annotation or LLM simulation
3. **Judge Setup:** Deploy multiple judges J₁, J₂, ..., Jₙ
4. **Score Collection:** Run all pairs through all judges
5. **Model Training:** Train fθ to predict ground truth from judge scores

**Training Data:** Judge evaluations → **{(J₁(x,a), J₂(x,a), ..., Jₙ(x,a))}**
**Target:** Ground truth scores

---

## Implementation Details

### Initial Experiments

**Dataset:** 10,000 (prompt, answer) pairs from UltraFeedback dataset

**Ground Truth Simulation:**
- Defined 8 distinct personas  
- Used Llama-3.1-405B to assign scores based on randomly assigned personas

**Judge Architecture:**
- 10 rubric-specific judges
- Each scoring different quality dimensions (conciseness, creativity, factuality)
- Judges implemented as LLMs with specific rubric prompts

**Models Tested:**
- **GAM (Generalized Additive Model):** Interpretable, combines smooth non-linear effects
- **Single-layer MLP:** 16 hidden units

### Results

| Model | MSE | MAE | R² |
|-------|-----|-----|-----|
| **NN Model (MLP)** | 3.06 | 1.35 | 0.57 |
| **GAM Model** | 3.11 | 1.36 | 0.56 |
| **Naive Mean Baseline** | 5.36 | 1.83 | 0.24 |

**Key Finding:** Both learned models significantly outperform naive averaging baseline.

### Further Analysis: Partial Dependence

**Method:** Analyzed GAM behavior using partial dependence analysis:
**PDⱼ(xⱼ) = E[f(xⱼ, x₋ⱼ)]**

**Findings:**
- **Non-safety features** (relevance, reasoning, conciseness, style): Strong positive relationship
- **Safety features** (harmlessness, privacy, bias): Weaker, inconsistent influence

**Implication:** Model prioritizes performance metrics over safety metrics in aggregation.

---

## Discussion

### Scalability Concerns

**Model Training:**
- Requires retraining for judge set modifications
- Small, interpretable models → manageable cost

**Training Data Requirements:**
- Ground truth: (prompt, answer, score) - ideally human-annotated
- Judge scores: Can be expensive with many judges
- Dataset deprecation as base models improve

### Improvement Opportunities

**Data Quality:**
- Replace LLM-simulated ground truth with human annotations
- Expand experimentation with different architectures

**Architecture Exploration:**
- Test deeper networks, different model types
- Explore ensemble methods

### Future Research Directions

**Experiment Ideas:**

1. **Personalization:** Can models learn individual human preferences? What happens with conflicting preferences?

2. **Robustness:** How does system handle bad rubrics or malicious judges?

3. **Metric-Judge Tradeoffs:** Optimal balance between number of metrics vs. judges?

4. **Architecture Optimization:** Which models perform best? Can we add control functions for safety?

### Safety Considerations

**Benefits:**
- More robust than single-judge systems
- Interpretable and cheap to train
- Enables safety research through experimentation

**Risks:**
- **Poisoning:** Bad human preferences or judge rubrics could corrupt the system
- **Information Hazard:** Economic incentives might favor non-interpretable but high-performing models

---

## Conclusion

This work presents a novel approach to multi-judge aggregation using learned functions rather than simple heuristics like averaging. Our experiments demonstrate significant performance improvements while maintaining interpretability.

The framework opens numerous research directions for optimizing judge systems and could contribute meaningfully to AI safety through better evaluation methods.

**Key Contributions:**
- Formalized the judge joining problem
- Demonstrated learned aggregation outperforms naive baselines  
- Provided interpretable insights into judge importance
- Established foundation for future safety research

---

*For detailed appendices with rubrics, personas, and additional experimental details, see the original submission linked above.*