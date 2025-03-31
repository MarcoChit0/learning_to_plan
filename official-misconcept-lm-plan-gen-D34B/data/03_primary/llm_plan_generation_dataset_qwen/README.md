---
configs:
- config_name: t0
  data_files:
  - split: train
    path: "t0/train/*.jsonl"
  - split: test_same_domain
    path: "t0/test_same_domain/*.jsonl"
  - split: test_unseen_domain
    path: "t0/test_unseen_domain/*.jsonl"
  - split: test_longer_horizon
    path: "t0/test_longer_horizon/*.jsonl"
  - split: test_mystery_domain
    path: "t0/test_mystery_domain/*.jsonl"
- config_name: accu_t1
  data_files:
  - split: train
    path: "accu_t1/train/*.jsonl"
  - split: test_same_domain
    path: "accu_t1/test_same_domain/*.jsonl"
  - split: test_unseen_domain
    path: "accu_t1/test_unseen_domain/*.jsonl"
  - split: test_longer_horizon
    path: "accu_t1/test_longer_horizon/*.jsonl"
  - split: test_mystery_domain
    path: "accu_t1/test_mystery_domain/*.jsonl"
- config_name: accu_t2
  data_files:
  - split: train
    path: "accu_t2/train/*.jsonl"
  - split: test_same_domain
    path: "accu_t2/test_same_domain/*.jsonl"
  - split: test_unseen_domain
    path: "accu_t2/test_unseen_domain/*.jsonl"
  - split: test_longer_horizon
    path: "accu_t2/test_longer_horizon/*.jsonl"
  - split: test_mystery_domain
    path: "accu_t2/test_mystery_domain/*.jsonl"
- config_name: accu_t3
  data_files:
  - split: train
    path: "accu_t3/train/*.jsonl"
  - split: test_same_domain
    path: "accu_t3/test_same_domain/*.jsonl"
  - split: test_unseen_domain
    path: "accu_t3/test_unseen_domain/*.jsonl"
  - split: test_longer_horizon
    path: "accu_t3/test_longer_horizon/*.jsonl"
  - split: test_mystery_domain
    path: "accu_t3/test_mystery_domain/*.jsonl"
- config_name: accu_t4
  data_files:
  - split: train
    path: "accu_t4/train/*.jsonl"
  - split: test_same_domain
    path: "accu_t4/test_same_domain/*.jsonl"
  - split: test_unseen_domain
    path: "accu_t4/test_unseen_domain/*.jsonl"
  - split: test_longer_horizon
    path: "accu_t4/test_longer_horizon/*.jsonl"
  - split: test_mystery_domain
    path: "accu_t4/test_mystery_domain/*.jsonl"
- config_name: accu_t1_t4
  data_files:
  - split: train
    path: "accu_t1_t4/train/*.jsonl"
  - split: test_same_domain
    path: "accu_t1_t4/test_same_domain/*.jsonl"
  - split: test_unseen_domain
    path: "accu_t1_t4/test_unseen_domain/*.jsonl"
  - split: test_longer_horizon
    path: "accu_t1_t4/test_longer_horizon/*.jsonl"
  - split: test_mystery_domain
    path: "accu_t1_t4/test_mystery_domain/*.jsonl"
- config_name: accu_t2_t4
  data_files:
  - split: train
    path: "accu_t2_t4/train/*.jsonl"
  - split: test_same_domain
    path: "accu_t2_t4/test_same_domain/*.jsonl"
  - split: test_unseen_domain
    path: "accu_t2_t4/test_unseen_domain/*.jsonl"
  - split: test_longer_horizon
    path: "accu_t2_t4/test_longer_horizon/*.jsonl"
  - split: test_mystery_domain
    path: "accu_t2_t4/test_mystery_domain/*.jsonl"
- config_name: accu_t1_t3
  data_files:
  - split: train
    path: "accu_t1_t3/train/*.jsonl"
  - split: test_same_domain
    path: "accu_t1_t3/test_same_domain/*.jsonl"
  - split: test_unseen_domain
    path: "accu_t1_t3/test_unseen_domain/*.jsonl"
  - split: test_longer_horizon
    path: "accu_t1_t3/test_longer_horizon/*.jsonl"
  - split: test_mystery_domain
    path: "accu_t1_t3/test_mystery_domain/*.jsonl"
- config_name: accu_t1_t3_t4
  data_files:
  - split: train
    path: "accu_t1_t3_t4/train/*.jsonl"
  - split: test_same_domain
    path: "accu_t1_t3_t4/test_same_domain/*.jsonl"
  - split: test_unseen_domain
    path: "accu_t1_t3_t4/test_unseen_domain/*.jsonl"
  - split: test_longer_horizon
    path: "accu_t1_t3_t4/test_longer_horizon/*.jsonl"
  - split: test_mystery_domain
    path: "accu_t1_t3_t4/test_mystery_domain/*.jsonl" 
- config_name: t0_t5
  data_files:
  - split: train
    path: "t0_t5/train/*.jsonl"
  - split: test_same_domain
    path: "t0_t5/test_same_domain/*.jsonl"
  - split: test_unseen_domain
    path: "t0_t5/test_unseen_domain/*.jsonl"
  - split: test_longer_horizon
    path: "t0_t5/test_longer_horizon/*.jsonl"
  - split: test_mystery_domain
    path: "t0_t5/test_mystery_domain/*.jsonl" 
---

# LLM Plan Generation Dataset

## Background

we used [Rao's LLM Planning Benchmark](https://github.com/karthikv792/LLMs-Planning/tree/main/plan-bench) dataset to generate training corpus for our language model.

## Details of different type of prompts 
### Stage 0 raw text sequence

### Stage 1 training data with some reasonable data augmentation on the [DOMAIN] section

Why data augmentation:

- Data augmentation helps neural networks learn features that are invariant to certain transformations.

Reasonable Augmentation in the domain descriptions

- permute the order of mentioned actions.
- permute the order of the conditions


Purpose of this data augmentation:

- **Learning semantic invariance**: The order of mentioning actions, restrictions, or conditions doesn't change the underlying meaning or logic of the problem.

Justification for using this type of data augmentation 

- **Natural language usage pattern**: In real-world scenarios, people often express the same information in different orders. This augmentation mimics natural language variation, making the model more robust to diverse phrasings.

### Stage 2 lite CoT in [PLAN] section

- write down the following three components {(1)init state, (2) goal state, (3) how many steps left} before we provide the action
- this encourages LM to concern about the state transitions and estimate the progress
- **Building towards CoT (Implicit)**
    - **Step Count to detect whether goal is reached:**
        - If the estimated number of steps left increases or doesn't decrease as expected, it's a clear signal that something might be wrong with the plan.
        - Example: If the LM estimates 5 steps left, then takes an action and still estimates 5 steps left, it suggests the action might not have been productive.
    - **Implicit shortest CoT by repeating init state and goal state:**
        - Having init state and goal state will determine the plan sequence. Thus by repeating them, we are forcing the model to associate the actual plan sequence with the init state and goal state tokens. 
        

### Stage 3 adding dense CoT in [PLAN] section

- before giving the action name in the step, write down the precondition of this action (e.g., I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.)
- after giving the state name in the step, write down the effect of this action (e.g., Once I pick up or unstack a block, I am holding the block.)

### Stage DEPRECATED shuffle the elements in each plan step

- again, we are learning the semantic invariance

### Stage 4 mistake-correction process

Consider an example with offline planning trajectory training data:

```jsx
A -> B -> C -> D
```

To create training data for mistake correction, we intentionally introduce errors by **moving later action steps forward** in the sequence.

```jsx
C [back] -> B [back] -> A -> B -> C -> D
```

- If Stage 2 is activated (which involves predicting how many steps are left), For each step in the mistake-augmented sequence, we assign a "steps left" (SL) value, which can either be:
    - the step left of the sentence in the original true sequence, denoted as `SL-true`
    - the local step left number, which can be expressed as `Len - index of current sentence` , where `Len`  is the length of the original true sequence

### Example:

Let's go through the example with the provided sequences:

1. **Original Sequence:**
    - `A (SL-true: 3) -> B (SL-true: 2) -> C (SL-true: 1) -> D (SL-true: 0)`
    - `Len = 4`
2. **Mistake Sequence:**
    - `C [back] -> B [back] -> A -> B -> C -> D`
3. **Step Prediction for Mistake Sequence:**
    - `C (SL-aug: 50% chance of 3 (Len - 1) or 1 (SL-true))`
    - `B (SL-aug: 50% chance of 2 (Len - 2) or 2 (SL-true))`
    - `A (SL-aug: 3)`
    - `B (SL-aug: 2)`
    - `C (SL-aug: 1)`
    - `D (SL-aug: 0)`


### Error-Correction Synthesized Data Generation Details 

There are two type of mistakes we can introduce in the synthesized data:
1. state transition is illegal / not desired 
   - synthesize the state transition that appears later in the plan to appear earlier in the plan
     - the model has to generalize the state transition because negative example templates can be extremely large and we cannot cover all the possible negative examples
2. cannot detect if the goal is reached
   - create negative examples with the remaining step counter to be incorrect
