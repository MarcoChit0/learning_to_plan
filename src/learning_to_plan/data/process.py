import os
from learning_to_plan import config
import datetime
from tqdm import tqdm

from learning_to_plan.data import generated_plan, task
from learning_to_plan import database, utils
logger = config.get_logger(__name__)
import numpy as np

def validate_plans(**kwargs):
    logger.info(f"Starting plan validation at {datetime.datetime.now()}.")
    generated_plans : set[generated_plan.GeneratedPlan] = database.generated_plan_database.get(filter_by_validity=generated_plan.GeneratedPlan.VALIDITY.UNCHECKED)
    for gen_plan in tqdm(generated_plans, desc="Validating Plans", unit="plan"):
        is_plan_valid = False
        try:        
            t : task.Task = database.task_database.get_by_id(gen_plan.task_id)
            if t is None:
                raise ValueError(f"Task with ID {gen_plan.task_id} not found in the task database.")
            try:
                result = utils.call_val(t, gen_plan.pddl_plan)
            except Exception as e:
                logger.error(f"Error calling validation for task {t.id} with plan {gen_plan.pddl_plan}: {e}")
                raise e

            for line in result.splitlines():
                if "Plan valid" in line:
                    is_plan_valid = True
                    break
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
    
                    
    def pass_at_k(total_num_samples, num_correct_samples, k):
        if k > total_num_samples:
            return np.nan
        # compute 1 - comb(n - c, k) / comb(n, k)
        if k > total_num_samples - num_correct_samples:
            return 1.0
        return 1 - np.prod(1 - k / np.arange(total_num_samples - num_correct_samples + 1, total_num_samples + 1))
                    
    MAX_K = 10
    _max_k = 0

    rows = []
    for pm_id, pm_data in data.items():
        for mm_id, mm_data in pm_data.items():
            for domain, domain_data in mm_data.items():
                for task_type, task_type_data in domain_data.items():
                    for task_id, task_data in task_type_data.items():
                        prompt_metadata = database.metadata_database.get_by_id(pm_id)
                        model_metadata = database.metadata_database.get_by_id(mm_id)
                        rows.append({
                            'prompt_metadata': prompt_metadata,
                            'model_metadata': model_metadata,
                            'domain': domain,
                            'task_type': task_type,
                            'task_id': task_id,
                            'num_samples': task_data['num_samples'],
                            'valid_samples': task_data['num_valid_samples']
                        })
    if not rows:
        logger.warning("No data to process for metrics computation.")
        return
    
    _max_k = max(row['num_samples'] for row in rows)
    _max_k = min(_max_k, MAX_K)
        
    df = pd.DataFrame(rows)

    # Calculate per-task metrics before grouping
    df['all_valid'] = (df['num_samples'] == df['valid_samples']).astype(int)
    df['any_valid'] = (df['valid_samples'] > 0).astype(int)

    for k in range(1, _max_k + 1):
        df[f'pass_at_{k}'] = df.apply(
            lambda row: pass_at_k(row['num_samples'], row['valid_samples'], k),
            axis=1
        )

    # Group by the specified columns
    group_cols = ['prompt_metadata', 'model_metadata', 'domain', 'task_type', 'num_samples']
    grouped = df.groupby(group_cols)

    # Define aggregations
    agg_dict = {
        'task_id': 'count',  # This will be our number_of_instances
        'all_valid': 'sum',
        'any_valid': 'sum',
        **{f'pass_at_{k}': 'mean' for k in range(1, _max_k + 1)}
    }

    # Apply aggregations
    results_df = grouped.agg(agg_dict).reset_index()

    # Rename columns and calculate accuracy metrics
    results_df.rename(columns={'task_id': 'number_of_instances'}, inplace=True)
    
    # Calculate accuracies, handling division by zero
    results_df['all_valid_accuracy'] = (results_df['all_valid'] / results_df['number_of_instances']).where(results_df['number_of_instances'] > 0, 0)
    results_df['any_valid_accuracy'] = (results_df['any_valid'] / results_df['number_of_instances']).where(results_df['number_of_instances'] > 0, 0)

    # Reorder columns to match the desired output structure
    final_cols = group_cols + [
        'number_of_instances',
        'all_valid',
        'any_valid',
        'all_valid_accuracy',
        'any_valid_accuracy'
    ] + [f'pass_at_{k}' for k in range(1, _max_k + 1)]

    results_df = results_df[final_cols]
    results_df = results_df.sort_values(by=['prompt_metadata', 'model_metadata', 'domain', 'task_type', 'num_samples'])
    results_df.reset_index(drop=True, inplace=True)
    results_df.to_csv(config.METRICS_FILE_PATH, index=False)
    logger.info(f"Metrics computed and saved to {config.METRICS_FILE_PATH} at {datetime.datetime.now()}.")