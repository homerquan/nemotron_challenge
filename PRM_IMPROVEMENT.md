# PRM-based Reasoning Improvement Pipeline

Goal: Improve LLM reasoning using Reinforcement Learning + Process Reward Model (PRM)

---

# 0. Overview

Pipeline:

Raw Dataset â Step-Level Structuring â PRM Training â RL Loop â Evaluation

Key idea:
We optimize **each reasoning step**, not just final answers.

---

# 1. Datasets (Where to Get Them)

## 1.1 Math Reasoning (Core)

- GSM8K  
  https://github.com/openai/grade-school-math

- MATH  
  https://github.com/hendrycks/math

- AIME (subset of MATH or scraped contest problems)

Use case:
- Step-by-step arithmetic + symbolic reasoning
- Strong supervision for PRM

---

## 1.2 Code Reasoning (Highly Recommended)

- HumanEval  
  https://github.com/openai/human-eval

- MBPP  
  https://github.com/google-research/google-research/tree/master/mbpp

- APPS  
  https://github.com/hendrycks/apps

Use case:
- Execution-based verification (strong reward signal)

---

## 1.3 Logical Reasoning

- ProofWriter  
  https://github.com/allenai/proofwriter

- StrategyQA  
  https://github.com/eladsegal/strategyqa

- LogiQA  
  https://github.com/lgw863/LogiQA-dataset

Use case:
- Multi-hop reasoning
- Deductive chains

---

## 1.4 Synthetic Reasoning Data (You generate)

Method:
- Prompt strong model (GPT-4 / Claude / etc.)
- Generate:
  - step-by-step solutions
  - multiple solution paths
  - wrong â corrected traces

---

## 1.5 Self-Play / Exploration Data

Method:
- Let model attempt problems
- Store:
  - attempts
  - corrections
  - failures

---

## 1.6 Scientific / Technical Reasoning

Sources:
- Physics problems (e.g. PhyQA)
- StackOverflow debugging traces
- Engineering docs

---

# 2. Data Preparation for PRM

## 2.1 Convert to Step-Level Format

Original:
Q â Answer

Target format:
Q â [Step1, Step2, Step3, ..., Final Answer]

Example:

{
  "question": "...",
  "steps": [
    {"text": "Define variables", "label": 1},
    {"text": "Apply equation", "label": 1},
    {"text": "Incorrect simplification", "label": 0},
    {"text": "Fix error", "label": 1}
  ]
}

---

## 2.2 Labeling Strategy

### Option A (Best): Programmatic Verification

- Math â numeric / symbolic solver
- Code â unit tests
- Logic â rule engine / constraints

Label:
- 1 = correct step
- 0 = incorrect step

---

### Option B: LLM-as-Judge (fallback)

Prompt:
"Is this reasoning step logically valid given previous steps?"

â ï Lower quality than programmatic verification

---

## 2.3 Build Step Transitions

Convert into RL-friendly format:

(State, Action, Reward)

Where:
- State = previous steps
- Action = next step
- Reward = correctness

---

## 2.4 Optional (Advanced): Reasoning Graphs

Instead of linear steps:

- nodes = steps
- edges = dependencies

Benefits:
- better structure
- aligns with graph reasoning models

---

# 3. Train the Process Reward Model (PRM)

## 3.1 Model Objective

Input:
- (Question + Partial Reasoning + Candidate Step)

Output:
- Probability step is correct

---

## 3.2 Training Data Format

{
  "input": "Q + previous steps + candidate step",
  "label": 1 or 0
}

---

## 3.3 Model Choices

- Small transformer (1Bâ7B)
- Fine-tune base LLM or train lightweight classifier head

---

## 3.4 Loss Function

Binary cross-entropy:

L = -[y log(p) + (1-y) log(1-p)]

---

# 4. Reinforcement Learning Loop

## 4.1 Rollout

For each question:

1. Model generates reasoning steps
2. At each step:
   - PRM scores step
   - assign reward

---

## 4.2 Reward Design

Reward = PRM(step_score)

Optional:
- + final correctness bonus
- + penalty for contradictions

---

## 4.3 RL Algorithm

Recommended:
- PPO (stable)
- GRPO (if no value model)
- DPO-style variants (simpler)

---

## 4.4 Training Loop

Repeat:

1. Sample problems
2. Generate reasoning trajectories
3. Score with PRM
4. Update policy

---

## 4.5 Key Trick

Donât only reward final answer.

Reward:
- intermediate correctness
- consistency
- recovery from errors

---

# 5. Benchmarking (Before vs After)

## 5.1 Baseline Model

Evaluate:
- base LLM (no PRM, no RL)

---

## 5.2 Metrics

### Accuracy
- GSM8K accuracy
- MATH accuracy
- HumanEval pass@k

---

### Reasoning Quality (NEW)

- Step correctness rate
- Error recovery rate
- Logical consistency

---

## 5.3 Example Benchmark Table

| Model              | GSM8K | MATH | HumanEval | Step Accuracy |
|------------------|------|------|-----------|--------------|
| Base LLM         | 57%  | 23%  | 32%       | 61%          |
| + CoT Prompting  | 68%  | 30%  | 38%       | 70%          |
| + PRM            | 74%  | 38%  | 45%       | 82%          |
| + RL (PRM-based) | 82%  | 48%  | 58%       | 91%          |

---

## 5.4 Ablation Study

Compare:

- RL without PRM â
- PRM without RL
- RL + PRM â

---

# 6. Practical Tips (Important)

## 6.1 Start Simple

- GSM8K + HumanEval only
- skip logic datasets initially

---

## 6.2 Verification > Data Size

Better:
- small + clean + verifiable

Worse:
- large + noisy

---

## 6.3 Use Curriculum

Train on:
1. easy problems
2. medium
3. hard

---

## 6.4 Mix Exploration + Exploitation

- keep some randomness
- donât overfit to PRM

---

# 7. Final Mental Model

Reasoning improves when:

- model explores multiple paths
- bad steps are penalized early
- correct structure is reinforced

---

# 8. Extension (Your Advantage)

Given your background:

- add SMT solver as verifier
- use graph-based reasoning
- simulate physical constraints


---

