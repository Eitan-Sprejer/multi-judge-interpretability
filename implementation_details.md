# **Multi-Judge Aggregation: Personas and Judges Configuration**

## **14 Personas for Comprehensive Preference Coverage**

### **Core Personas (Scores 8-9)**

1. **Professor** \- An academic who values intellectual rigor, proper argumentation, logical consistency, and educational value in explanations  
2. **CEO** \- A business executive who appreciates conciseness, practical solutions, strategic thinking, and clear action items that drive results  
3. **Parent** \- A caring guardian who looks for safety, age-appropriate content, clear explanations, and practical advice for everyday situations  
4. **Student** \- A learner who needs clear step-by-step explanations, examples, study tips, and help understanding difficult concepts  
5. **Data Scientist** \- A technical professional who values accuracy, statistical rigor, code quality, reproducibility, and evidence-based reasoning  
6. **Therapist** \- A mental health professional who appreciates empathy, emotional intelligence, non-judgmental language, and supportive communication  
7. **Child** \- An 8-12 year old who prefers simplicity, fun explanations, relatable examples, and encouraging language  
8. **Ethicist** \- A moral philosopher who values ethical reasoning, consideration of consequences, fairness, and philosophical grounding  
9. **Privacy Advocate** \- A security-conscious individual who prioritizes data minimization, security awareness, anonymity, and privacy protection  
10. **Skeptic** \- A critical thinker who demands evidence, identifies logical fallacies, maintains healthy doubt, and verifies claims

### **Additional Personas (Score 7\)**

11. **Engineer** \- A technical builder who values precision, implementation details, efficiency, and systematic debugging approaches  
12. **Novelist** \- A creative writer who enjoys vivid descriptions, emotional depth, narrative flow, and imaginative approaches to problem-solving  
13. **Non-native Speaker** \- A language learner who needs clear language, avoidance of idioms, cultural context, and simplified vocabulary  
14. **Lawyer** \- A legal professional who requires precision in language, edge case consideration, risk assessment, and precedent awareness

---

## **10 Judge Descriptions**

1. **truthfulness-judge** \- Evaluates factual correctness, accuracy of information, and absence of hallucinations  
2. **harmlessness-judge** \- Evaluates safety, absence of harmful content, and protection of vulnerable groups including privacy concerns  
3. **helpfulness-judge** \- Evaluates task utility, problem-solving effectiveness, actionability, and value to the user  
4. **honesty-judge** \- Evaluates appropriate expression of uncertainty, calibrated confidence, and transparency about limitations  
5. **explanatory-depth-judge** \- Evaluates thoroughness of explanation, presence of examples, step-by-step reasoning, and contextual details  
6. **instruction-following-judge** \- Evaluates adherence to explicit/implicit instructions, task focus, and constraint satisfaction  
7. **clarity-judge** \- Evaluates comprehension ease, accessibility, absence of jargon, and structural organization  
8. **conciseness-judge** \- Evaluates brevity, efficiency of communication, and elimination of redundant information  
9. **logical-consistency-judge** \- Evaluates reasoning quality, argument structure, absence of contradictions, and logical flow  
10. **creativity-judge** \- Evaluates originality, novel approaches, engaging presentation, and imaginative problem-solving

---

## **Persona-Judge Coverage Matrix**

### **Professor**

* truthfulness-judge (Critical \- needs factual accuracy)  
* explanatory-depth-judge (Critical \- wants thorough explanations)  
* logical-consistency-judge (Critical \- values rigorous reasoning)  
* honesty-judge (Important \- appreciates uncertainty acknowledgment)

### **CEO**

* conciseness-judge (Critical \- time is money)  
* helpfulness-judge (Critical \- needs actionable solutions)  
* instruction-following-judge (Critical \- wants precise task completion)  
* clarity-judge (Important \- needs quick comprehension)

### **Parent**

* harmlessness-judge (Critical \- child safety first)  
* helpfulness-judge (Critical \- practical everyday advice)  
* clarity-judge (Critical \- understandable guidance)  
* explanatory-depth-judge (Important \- teaching moments)

### **Student**

* explanatory-depth-judge (Critical \- needs learning support)  
* clarity-judge (Critical \- must understand concepts)  
* helpfulness-judge (Critical \- study effectiveness)  
* creativity-judge (Important \- engaging learning)

### **Data Scientist**

* truthfulness-judge (Critical \- statistical accuracy)  
* logical-consistency-judge (Critical \- sound methodology)  
* honesty-judge (Critical \- uncertainty quantification)  
* explanatory-depth-judge (Important \- reproducibility details)

### **Therapist**

* harmlessness-judge (Critical \- do no harm)  
* honesty-judge (Critical \- authentic communication)  
* explanatory-depth-judge (Important \- understanding context)  
* clarity-judge (Important \- clear communication)

### **Child**

* clarity-judge (Critical \- age-appropriate language)  
* harmlessness-judge (Critical \- content safety)  
* creativity-judge (Critical \- fun and engaging)  
* helpfulness-judge (Important \- actually answers questions)

### **Ethicist**

* logical-consistency-judge (Critical \- moral reasoning)  
* harmlessness-judge (Critical \- ethical implications)  
* honesty-judge (Critical \- acknowledging dilemmas)  
* explanatory-depth-judge (Important \- exploring consequences)

### **Privacy Advocate**

* harmlessness-judge (Critical \- includes privacy protection)  
* honesty-judge (Critical \- data use transparency)  
* instruction-following-judge (Important \- respecting boundaries)  
* truthfulness-judge (Important \- accurate privacy info)

### **Skeptic**

* truthfulness-judge (Critical \- evidence-based claims)  
* logical-consistency-judge (Critical \- sound reasoning)  
* honesty-judge (Critical \- admitting uncertainty)  
* explanatory-depth-judge (Important \- showing work/sources)

### **Engineer**

* instruction-following-judge (Critical \- precise specifications)  
* logical-consistency-judge (Critical \- systematic thinking)  
* conciseness-judge (Important \- efficient communication)  
* truthfulness-judge (Important \- technical accuracy)

### **Novelist**

* creativity-judge (Critical \- imaginative content)  
* explanatory-depth-judge (Critical \- rich descriptions)  
* helpfulness-judge (Important \- serves creative goals)  
* clarity-judge (Important \- readable prose)

### **Non-native Speaker**

* clarity-judge (Critical \- simple language)  
* explanatory-depth-judge (Critical \- context and examples)  
* helpfulness-judge (Important \- practical understanding)  
* harmlessness-judge (Important \- cultural sensitivity)

### **Lawyer**

* truthfulness-judge (Critical \- factual precision)  
* logical-consistency-judge (Critical \- airtight reasoning)  
* instruction-following-judge (Critical \- precise compliance)  
* explanatory-depth-judge (Important \- covering edge cases)

