## How 
We will use two types of evaluation metric:

1. measure the longest common subsequence (LCS) between the generated plan and the reference plan. This however, is a bit problematic because there can be various valid plans for the problem and the reference plan is just one of them. 
2. using PDDL VAL to check if the generated plan is valid or not. this is a more reliable way to check the validity of the plan. But it is just a binary check, either the plan is valid or not. It thus is less informative than the LCS/LCCS metric.
## LCS / LCCS metric
Longest Common Subsequence (LCS)

- Applicability: LCS is generally more suitable for comparing PDDL plans because it captures the sequence of actions that appear in both plans without requiring them to be contiguous. This is important in planning where the order of certain actions matters, but they may be interspersed with different actions in different plans.
- Flexibility: LCS allows for interruptions in the sequence, accommodating variations in plans that might include additional steps or alternative paths that still achieve the same goals.
- Insight into Plan Quality: LCS can provide insights into the core similarities between two plans, highlighting the fundamental strategies or steps they have in common, even if there are differences in specific actions or the order of some actions.

Longest Common Contiguous Subsequence (LCCS)

- Applicability: LCCS is less suitable for PDDL plan comparison because it focuses on the longest sequence of actions that occur exactly in the same order in both plans. This can be overly restrictive since valid and high-quality plans can have variations in the sequence of actions due to the nature of planning problems.
- Specificity: LCCS might be useful in very specific contexts where you need to ensure that a particular sub-sequence of actions is strictly followed in a specific order, but this is generally not the case in typical planning scenarios.
- Limitation in Insight: Using LCCS might miss out on recognizing equivalently good plans that have the same actions in a slightly different order or interspersed with other actions.



We will use **all the three of these metrics (VAL binary check, LCS, LCCS)** to evaluate the performance of the model across different testing datasets, which are:

1. test_same_domain = "test set with the same domain"
2. test_unseen_domain = "test set with unseen domain"
3. test_longer_horizon = "test set with longer horizon"
4. test_mystery_domain = "test set with mystery domain"


## How to run the evaluation
1. Run the model on the test set using the following command:
```bash
bash ./zjob-scripts/step_3_lm_planning_evaluation.sh  <config_type> (e.g., accu_t1)
```
the script will pass the config_type as environment variable and during the evaluation, the script will use the config_type to load the corresponding model and data.


## Test data information
- Size of the test_same_domain dataset: 1600
- Size of the test_longer_horizon dataset: 1600
- Size of the test_unseen_domain dataset: 400
- Size of the test_mystery_domain dataset: 400