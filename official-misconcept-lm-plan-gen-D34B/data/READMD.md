> [!WARNING]
> The training data is extremely large (100GB) and is not included in this repository. You can download the data via Hugging Face Hub once the paper reviewing process is complete. Here we only provide the PDDL snippets and part of the test set data for you to inspect. 


## Split Name of the Dataset
The split name of the dataset is slightly different from the paper. Here is the mapping:
```python
{
    "t0": "label 1 (vanilla)",
    "accu_t1": "label 2 (+perm.)",
    "accu_t2": "label 3 (+perm. + Goal CoT)",
    "accu_t1_t3": "label 4 (+perm. + State CoT)",
    "accu_t3": "label 5 (+perm. + Goal CoT + State CoT)",
    "accu_t1_t4": "label 6 (+perm. + self-correction)",
    "accu_t2_t4": "label 7 (+perm. + Goal CoT + self-correction)",
    "accu_t1_t3_t4": "label 8 (+perm. + State CoT + self-correction)",
    "accu_t4": "label 9 (+perm. + Goal CoT + State CoT + self-correction)"
    "t0_t5": "A special case of self-correction mentioned in the appendix, where the retry data is not obtained by just selecting irrelevant actions from the action space.
}
```