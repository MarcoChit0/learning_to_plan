## Testing models 
Only the models that are trained using scratchpad data are tested as they are capable of generating "\<BACK\>". 

## Purpose
We are going to see if the model is able to identify the invalid plans more effectively when CoT content is provided. 

## How to run the probing
1. Run the model on the test set using the following command:
```bash
bash ./zjob-scripts/step_4_lm_planning_probing.sh  <config_type> (e.g., accu_t1)
```
the script will pass the config_type as environment variable and during the probing, the script will use the config_type to load the corresponding model and data.

## Results Structure 
The `results` dictionary contains four key metrics used to evaluate the model's performance in identifying mistakes:

1. `token_probability_contrast_ratio_mistake_case`: 
   - A list storing the contrast ratios between the probability of having the \<BACK\> token versus having newline token, for cases where a mistake is actually present.
   - Higher values indicate better performance in identifying actual mistakes.

2. `token_probability_contrast_ratio_valid_case`:
   - A list storing the contrast ratios between the probability of having the \<BACK\> token versus having newline token, for cases where no mistake is present.
   - Lower values are desirable, indicating the model correctly identifies valid cases.

3. `mistake_precision_count`:
   - A list tracking the precision of mistake identification. When `p(\<BACK\>) > p(newline)`, we consider it as a positive identification of a mistake
   - In the end we will calculate (number of true mistakes identified) / (total number of identified mistakes)
   - Higher values indicate better precision in mistake identification.

4. `mistake_recall_count`:
   - A list tracking the recall of mistake identification. When `p(\<BACK\>) > p(newline)`, we consider it as a positive identification of a mistake.
   - In the end we will calculate (number of true mistakes identified) / (total number of actual mistakes)
   - Higher values indicate better recall in finding all actual mistakes.

> [!NOTE]
> Normally we think that recall is more important than precision in this case, because we want to make sure that we don't miss any mistakes.


## Documentation for `output_dict`

The `output_dict` is a dictionary that stores the results of the language model token probing. This dictionary is serialized and saved as a pickle file. Below is a detailed explanation of each key in the dictionary and its corresponding value:

1. **mistake_identification_precision_val**
   - **Type:** `float`
   - **Description:** This value represents the precision of mistake identification during the language model probing. Precision is the ratio of correctly identified mistakes to the total number of identified mistakes.

2. **mistake_identification_recall_val**
   - **Type:** `float`
   - **Description:** This value represents the recall of mistake identification during the language model probing. Recall is the ratio of correctly identified mistakes to the total number of actual mistakes.

3. **token_probability_contrast_ratio_mistake_case_lst**
   - **Type:** `list`
   - **Description:** This list contains the contrast ratios of token probabilities for cases where mistakes were identified. Each element in the list represents the contrast ratio for a specific mistake case.

4. **token_probability_contrast_ratio_valid_case_lst**
   - **Type:** `list`
   - **Description:** This list contains the contrast ratios of token probabilities for valid cases (i.e., cases without mistakes). Each element in the list represents the contrast ratio for a specific valid case.

### Example Usage

To read the pickle file and access the pickled dict, you can use the following code:

```python
import pickle
import os

# Define the path to the pickle file
pkl_save_path = os.path.join(os.environ["WORKING_DIR"], "data/08_reporting/language_model_token_probing/{model_config_type}_probing_results.pkl")

# Load the dictionary from the pickle file
with open(pkl_save_path, "rb") as f:
    output_dict = pickle.load(f)

# Accessing the values
precision = output_dict["mistake_identification_precision_val"]
recall = output_dict["mistake_identification_recall_val"]
mistake_cases = output_dict["token_probability_contrast_ratio_mistake_case_lst"] # use this to generate boxplot
valid_cases = output_dict["token_probability_contrast_ratio_valid_case_lst"] # use this to generate boxplot

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Mistake Cases: {mistake_cases}")
print(f"Valid Cases: {valid_cases}")
```

