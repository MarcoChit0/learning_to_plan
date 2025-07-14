import os
from learning_to_plan import config
import datetime
import subprocess
from tqdm import tqdm

from learning_to_plan.data import generated_plan, task, base
from learning_to_plan import database
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

def validate_plans(**kwargs):
    logger.info(f"Starting plan validation at {datetime.datetime.now()}.")
    generated_plans : set[generated_plan.GeneratedPlan] = database.generated_plan_database.get(filter_by_validity=generated_plan.GeneratedPlan.VALIDITY.UNCHECKED)
    for gen_plan in tqdm(generated_plans, desc="Validating Plans", unit="plan"):
        is_plan_valid = False
        try:        
            temp_plan_file = os.path.join(
            f".temp_plan_{gen_plan.id}.txt"
            )
            with open(temp_plan_file, "w") as f:
                f.write(gen_plan.pddl_plan)

            t : task.Task = database.task_database.get_by_id(gen_plan.task_id)

            logger.debug(f"Created temporary plan file: {temp_plan_file}")
            cmd_list = [
                "utils/VAL/build/bin/Validate",
                "-v",
                "-t", "0.001",
                t.domain_file_path,
                t.instance_file_path,
                temp_plan_file
            ]
            result = subprocess.run(cmd_list, capture_output=True, text=True, check=False)
            for line in result.stdout.splitlines():
                if "Plan valid" in line:
                    is_plan_valid = True
                    break
            os.remove(temp_plan_file)
        except Exception as e:
            logger.error(f"Error validating generated plan {gen_plan.id} : {e}")
            raise e
        gen_plan.validate(is_valid=is_plan_valid)
    database.generated_plan_database.update(generated_plans)
    logger.info(f"Plan validation completed at {datetime.datetime.now()}.")

import pandas as pd

def compute_metrics(
    **kwargs
):
    logger.info(f"Computing metrics at {datetime.datetime.now()}.")
    data = {}
    generated_plans : set[generated_plan.GeneratedPlan] = database.generated_plan_database.get()
    if any({gp for gp in generated_plans if gp.validity == generated_plan.GeneratedPlan.VALIDITY.UNCHECKED}):
        raise ValueError("Must validate all generated plans before computing metrics.")
    
    for gen_plan in generated_plans:
        if gen_plan.prompt_metadata_id not in data:
            data[gen_plan.prompt_metadata_id] = {}
        
        if gen_plan.model_metadata_id not in data[gen_plan.prompt_metadata_id]:
            data[gen_plan.prompt_metadata_id][gen_plan.model_metadata_id] = {}
        
        try:
            t : task.Task = database.task_database.get_by_id(gen_plan.task_id)
        except:
            raise ValueError(f"Could not recover task {gen_plan.task_id} from the task database.")

        if t.domain not in data[gen_plan.prompt_metadata_id][gen_plan.model_metadata_id]:
            data[gen_plan.prompt_metadata_id][gen_plan.model_metadata_id][t.domain] = {}

        if t.type not in data[gen_plan.prompt_metadata_id][gen_plan.model_metadata_id][t.domain]:
            data[gen_plan.prompt_metadata_id][gen_plan.model_metadata_id][t.domain][t.type] = {}

        if t.id not in data[gen_plan.prompt_metadata_id][gen_plan.model_metadata_id][t.domain][t.type]:
            data[gen_plan.prompt_metadata_id][gen_plan.model_metadata_id][t.domain][t.type][t.id] = {
                'num_samples': 0,
                'num_valid_samples': 0
            }
        
        data[gen_plan.prompt_metadata_id][gen_plan.model_metadata_id][t.domain][t.type][t.id]['num_samples'] += 1

        if gen_plan.validity == generated_plan.GeneratedPlan.VALIDITY.VALID:
            data[gen_plan.prompt_metadata_id][gen_plan.model_metadata_id][t.domain][t.type][t.id]['num_valid_samples'] += 1
    
    processed_data = {}
    for prompt_metadata_id, prompt_data in data.items():
        for model_metadata_id, models_data in prompt_data.items():
            for domain, task_types in models_data.items():
                for task_type, tasks in task_types.items():
                    for task_id, metrics in tasks.items():
                        num_samples = metrics['num_samples']
                        num_valid_samples = metrics['num_valid_samples']
                        
                        if prompt_metadata_id not in processed_data:
                            processed_data[prompt_metadata_id] = {}
                        if model_metadata_id not in processed_data[prompt_metadata_id]:
                            processed_data[prompt_metadata_id][model_metadata_id] = {}
                        if domain not in processed_data[prompt_metadata_id][model_metadata_id]:
                            processed_data[prompt_metadata_id][model_metadata_id][domain] = {}
                        if task_type not in processed_data[prompt_metadata_id][model_metadata_id][domain]:
                            processed_data[prompt_metadata_id][model_metadata_id][domain][task_type] = {}

                        if num_samples not in processed_data[prompt_metadata_id][model_metadata_id][domain][task_type]:
                            processed_data[prompt_metadata_id][model_metadata_id][domain][task_type][num_samples] = {
                                'list_num_valid_samples': []
                            }
                        
                        processed_data[prompt_metadata_id][model_metadata_id][domain][task_type][num_samples]['list_num_valid_samples'].append(num_valid_samples)
                    
    def pass_at_k(total_num_samples, num_correct_samples, k):
        # compute 1 - comb(n - c, k) / comb(n, k)
        if k > total_num_samples - num_correct_samples:
            return 1.0
        return 1 - np.prod(1 - k / np.arange(total_num_samples - num_correct_samples + 1, total_num_samples + 1))
                    
    MAX_K = 10

    results_list = []
    for prompt_metadata_id, prompt_data in processed_data.items():
        try:
            prompt_metadata = database.metadata_database.get_by_id(prompt_metadata_id)
        except Exception as e:
            raise ValueError(f"Could not recover prompt metadata {prompt_metadata_id} from the database: {e}")
        for model_metadata_id, models_data in prompt_data.items():

            try:
                model_metadata = database.metadata_database.get_by_id(model_metadata_id)
            except Exception as e:
                raise ValueError(f"Could not recover model metadata {model_metadata_id} from the database: {e}")
            for domain, domains_data in models_data.items():
                for task_type, tasks_type_data in domains_data.items():
                    for num_samples, metrics in tasks_type_data.items():
                        k = 0
                        accuracy_all_valid = 0.0
                        accuracy_any_valid = 0.0
                        std_validity_ratio = 0.0
                        avg_validity_ratio = 0.0
                        list_num_valid_samples = metrics['list_num_valid_samples']
                        number_of_instances = len(list_num_valid_samples)
                        
                        if number_of_instances > 0:
                            k = min(MAX_K, num_samples)
                            all_valid = sum(1 for x in list_num_valid_samples if x == num_samples)
                            accuracy_all_valid = all_valid / number_of_instances
                            
                            any_valid = sum(1 for x in list_num_valid_samples if x > 0) 
                            accuracy_any_valid = any_valid / number_of_instances
                            avg_validity_ratio = sum(list_num_valid_samples) / (number_of_instances * num_samples)
                            std_validity_ratio = (sum((x - avg_validity_ratio) ** 2 for x in list_num_valid_samples) / number_of_instances) ** 0.5
                            
                            pass_at_k_values_by_k = {k : [] for k in range(1, k + 1)}
                            for instance in range(number_of_instances):
                                for k in range(1, k + 1):
                                    pass_at_k_values_by_k[k].append(
                                        pass_at_k(num_samples, list_num_valid_samples[instance], k)
                                    )
                            
                            k_values = {i: np.mean(pass_at_k_values_by_k[i]) if i <= k else np.nan for i in range(1, MAX_K + 1)}
                        
                        results_list.append({
                            'prompt_metadata': prompt_metadata,
                            'model_metadata': model_metadata,
                            'domain': domain,
                            'task_type': task_type,
                            'num_samples': num_samples,
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
        'prompt_metadata', 
        'model_metadata', 
        'domain', 
        'task_type', 
        'num_samples',
        'accuracy_all_valid', 
        'all_valid', 
        'accuracy_any_valid',
        'any_valid', 
        'avg_validity_ratio', 
        'std_validity_ratio',
        'number_of_instances',
        *[f'pass_at_k_{k}' for k in range(1, MAX_K + 1)]
    ])
    results_df = results_df.sort_values(by=['prompt_metadata', 'model_metadata', 'domain', 'task_type', 'num_samples'])
    results_df.reset_index(drop=True, inplace=True)
    results_df.to_csv(config.METRICS_FILE_PATH, index=False)
    logger.info(f"Metrics computed and saved to {config.METRICS_FILE_PATH} at {datetime.datetime.now()}.")