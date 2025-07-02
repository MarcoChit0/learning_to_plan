import os
from typing import Any, Callable, Optional
from learning_to_plan import config, models
from learning_to_plan import task
import datetime
import subprocess
import csv
from tqdm import tqdm
logger = config.get_logger(__name__)
import numpy as np
# # domin_path = "data/raw/blocksworld/generated_domain.pddl"
# # problem_path = "data/raw/blocksworld/generated_basic/instance-0.pddl"
# # plan ="""(unstack a c)
# # (put-down a)
# # (pick-up b)
# # (stack b c)
# # (pick-up a)
# # (stack a b)"""

# # mapping = {
# #     "a": "red",
# #     "b": "blue",
# #     "c": "yellow"
# # }

# # nl_plan = """unstack the red block from the yellow block
# # put down the red block
# # pick up the blue block
# # stack the blue block on top of the yellow block
# # pick up the red block
# # stack the red block on top of the blue block"""


# # with open("plan_blocksworld_instance-0.txt", "w") as f:
# #     f.write(plan)


# # val_command = f"../VAL/bin/Validate -v -t 0.001 {domin_path} {problem_path} plan_blocksworld_instance-0.txt"
# # os.system(val_command)

def validate_plans(model:models.Model, **kwargs):
    logger.info(f"Starting plan validation at {datetime.datetime.now()}.")
    generated_plans = model.get_generated_plans()
    for content in tqdm(generated_plans, desc="Validating Plans", unit="plan"):
        if content.was_vaidated():
            continue
        
        is_plan_valid = False
        if content._status != models.Model.Content.STATUS.ERROR:
            try:        
                temp_plan_file = os.path.join(
                model._model_dir_path,
                f".temp_plan_{model._model_name}_{content._task._domain_file_path}_{content._task._instance_file_path}_{content._prompt_type}_{content._id}.txt".replace(" ", "_").replace("/", "_")
                )
                with open(temp_plan_file, "w") as f:
                    f.write(content._pddl_plan)
                logger.debug(f"Created temporary plan file: {temp_plan_file}")
                cmd_list = [
                    "utils/VAL/build/bin/Validate",
                    "-v",
                    "-t", "0.001",
                    content._task._domain_file_path,
                    content._task._instance_file_path,
                    temp_plan_file
                ]
                result = subprocess.run(cmd_list, capture_output=True, text=True, check=False)

                for line in result.stdout.splitlines():
                    if "Plan valid" in line:
                        is_plan_valid = True
                        break
                os.remove(temp_plan_file)
            except Exception as e:
                logger.error(f"Error validating plan for task {content._task._id} - Model '{model._model_name}' - Prompt Type '{content._prompt_type}': {e}")
                raise e
        content.validate(is_plan_valid)
    model.save_generated_plans()
    logger.info(f"Plan validation completed at {datetime.datetime.now()}.")

import pandas as pd

def compute_metrics(
    model:models.Model,
    **kwargs,
):
    logger.info(f"Computing metrics for model {model._model_name} at {datetime.datetime.now()}.")
    data = {}
    for content in model.get_generated_plans():
        if not content.was_vaidated():
            print(f"skip -- {content._validity}")
            continue 

        if content._prompt_type not in data:
            data[content._prompt_type] = {}

        # convert dict into string to use as a key
        prompt_metadata_key = str(content._prompt_metadata)
        if prompt_metadata_key not in data[content._prompt_type]:
            data[content._prompt_type][prompt_metadata_key] = {}
        
        # convert dict into string to use as a key
        model_metadata_key = str(content._model_metadata)
        if model_metadata_key not in data[content._prompt_type][prompt_metadata_key]:
            data[content._prompt_type][prompt_metadata_key][model_metadata_key] = {}
        
        if content._task._domain not in data[content._prompt_type][prompt_metadata_key][model_metadata_key]:
            data[content._prompt_type][prompt_metadata_key][model_metadata_key][content._task._domain] = {}
        
        plan_size = "longer_plan" if content._task._is_longer_plan else "basic_plan"

        if plan_size not in data[content._prompt_type][prompt_metadata_key][model_metadata_key][content._task._domain]:
            data[content._prompt_type][prompt_metadata_key][model_metadata_key][content._task._domain][plan_size] = {}

        if content._task._instance_file_path not in data[content._prompt_type][prompt_metadata_key][model_metadata_key][content._task._domain][plan_size]:
            data[content._prompt_type][prompt_metadata_key][model_metadata_key][content._task._domain][plan_size][content._task._instance_file_path] = {
                'num_samples': 0,
                'num_valid_samples': 0,
            }
        
        data[content._prompt_type][prompt_metadata_key][model_metadata_key][content._task._domain][plan_size][content._task._instance_file_path]['num_samples'] += 1
        if content._validity == models.Model.Content.VALIDITY.VALID:
            data[content._prompt_type][prompt_metadata_key][model_metadata_key][content._task._domain][plan_size][content._task._instance_file_path]['num_valid_samples'] += 1
    
    processed_data = {}
    for prompt_type in data:
        for prompt_metadata in data[prompt_type]:
            for model_metadata in data[prompt_type][prompt_metadata]:
                for domain in data[prompt_type][prompt_metadata][model_metadata]:
                    for plan_size in data[prompt_type][prompt_metadata][model_metadata][domain]:
                        for intance_file_path, metrics in data[prompt_type][prompt_metadata][model_metadata][domain][plan_size].items():
                            num_samples = metrics['num_samples']
                            num_valid_samples = metrics['num_valid_samples']
                            
                            if prompt_type not in processed_data:
                                processed_data[prompt_type] = {}
                            if prompt_metadata not in processed_data[prompt_type]:
                                processed_data[prompt_type][prompt_metadata] = {}
                            if model_metadata not in processed_data[prompt_type][prompt_metadata]:
                                processed_data[prompt_type][prompt_metadata][model_metadata] = {}
                            if domain not in processed_data[prompt_type][prompt_metadata][model_metadata]:
                                processed_data[prompt_type][prompt_metadata][model_metadata][domain] = {}
                            if plan_size not in processed_data[prompt_type][prompt_metadata][model_metadata][domain]:
                                processed_data[prompt_type][prompt_metadata][model_metadata][domain][plan_size] = {}
                            
                            if num_samples not in processed_data[prompt_type][prompt_metadata][model_metadata][domain][plan_size]:
                                processed_data[prompt_type][prompt_metadata][model_metadata][domain][plan_size][num_samples] = {
                                    'list_num_valid_samples': []
                                }
                            
                            processed_data[prompt_type][prompt_metadata][model_metadata][domain][plan_size][num_samples]['list_num_valid_samples'].append(num_valid_samples)

    def pass_at_k(total_num_samples, num_correct_samples, k):
        # compute 1 - comb(n - c, k) / comb(n, k)
        if k > total_num_samples - num_correct_samples:
            return 1.0
        return 1 - np.prod(1 - k / np.arange(total_num_samples - num_correct_samples + 1, total_num_samples + 1))
                    
    MAX_K = 10

    results_list = []
    for prompt_type, prompt_data in processed_data.items():
        for prompt_metadata, models_data in prompt_data.items():
            for model_metadata, domains in models_data.items():
                for domain, plan_types in domains.items():
                    for plan_type, num_samples_data in plan_types.items():
                        for num_samples_val, metrics in num_samples_data.items():
                            k = 0
                            accuracy_all_valid = 0.0
                            accuracy_any_valid = 0.0
                            std_validity_ratio = 0.0
                            avg_validity_ratio = 0.0
                            list_num_valid_samples = metrics['list_num_valid_samples']
                            number_of_instances = len(list_num_valid_samples)
                            
                            if number_of_instances > 0:
                                k = min(MAX_K, num_samples_val)

                                all_valid = sum(1 for x in list_num_valid_samples if x == num_samples_val)
                                accuracy_all_valid = all_valid / number_of_instances
                                
                                any_valid = sum(1 for x in list_num_valid_samples if x > 0) 
                                accuracy_any_valid = any_valid / number_of_instances

                                avg_validity_ratio = sum(list_num_valid_samples) / (number_of_instances * num_samples_val)
                                std_validity_ratio = (sum((x - avg_validity_ratio) ** 2 for x in list_num_valid_samples) / number_of_instances) ** 0.5

                                
                                pass_at_k_values_by_k = {k : [] for k in range(1, k + 1)}
                                for instance in range(number_of_instances):
                                    for k in range(1, k + 1):
                                        pass_at_k_values_by_k[k].append(
                                            pass_at_k(num_samples_val, list_num_valid_samples[instance], k)
                                        )
                                
                                k_values = {i: np.mean(pass_at_k_values_by_k[i]) if i <= k else np.nan for i in range(1, MAX_K + 1)}

                            
                            results_list.append({
                                'prompt_type': prompt_type,
                                'prompt_metadata': prompt_metadata,
                                'model_metadata': model_metadata,
                                'domain': domain,
                                'plan_type': plan_type,
                                'num_samples': num_samples_val,
                                'accuracy_all_valid': accuracy_all_valid,
                                'all_valid': all_valid,
                                'accuracy_any_valid': accuracy_any_valid,
                                'any_valid': any_valid,
                                'avg_validity_ratio': avg_validity_ratio,
                                'std_validity_ratio': std_validity_ratio,
                                'number_of_instances': number_of_instances,
                                **{f'pass_at_k_{i}': k_values[i] for i in range(1, MAX_K + 1)}
                            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list, columns=[
        'prompt_type', 'prompt_metadata', 'model_metadata' 'domain', 'plan_type', 'num_samples',
        'accuracy_all_valid', 'all_valid', 'accuracy_any_valid',
        'any_valid', 'avg_validity_ratio', 'std_validity_ratio',
        'number_of_instances',
        *[f'pass_at_k_{k}' for k in range(1, MAX_K + 1)]
    ])
    results_df = results_df.sort_values(by=['prompt_type', 'prompt_metadata', 'model_metadata' 'domain', 'plan_type', 'num_samples'])
    results_df.reset_index(drop=True, inplace=True)
    # Save results to CSV
    csv_file_path = os.path.join(model._model_dir_path, f"metrics_{model._model_name}.csv".replace(" ", "_").replace("/", "_"))
    results_df.to_csv(csv_file_path, index=False)
    logger.info(f"Metrics for model {model._model_name} saved to {csv_file_path} at {datetime.datetime.now()}.")