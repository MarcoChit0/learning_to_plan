import os
from typing import Any, Callable, Optional
from learning_to_plan import config, models
from learning_to_plan import task
import datetime
import subprocess
import csv
from tqdm import tqdm
logger = config.get_logger(__name__)
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
    # Calculate total number of plans to validate
    total_plans = sum(
        len(model._generated_plans[task][prompt_type]['pddl'])
        for task in model._generated_plans
        for prompt_type in model._generated_plans[task]
    )
    progress_bar = tqdm(total=total_plans, desc="Validated Tasks", unit="plan")

    for task in model._generated_plans:
        for prompt_type in model._generated_plans[task]:
            pddl_plans = model._generated_plans[task][prompt_type]['pddl']
            for i, pddl_plan in enumerate(pddl_plans):
                temp_plan_file = os.path.join(
                    model._model_dir_path,
                    f".temp_plan_{model._model_name}_{task._domain_file_path}_{task._instance_file_path}_{prompt_type}_{i}.txt".replace(" ", "_").replace("/", "_")
                )
                try:
                    with open(temp_plan_file, "w") as f:
                        f.write(pddl_plan)
                    logger.debug(f"Created temporary plan file: {temp_plan_file}")
                    cmd_list = [
                        "utils/VAL/build/bin/Validate",
                        "-v",
                        "-t", "0.001",
                        task._domain_file_path,
                        task._instance_file_path,
                        temp_plan_file
                    ]
                    result = subprocess.run(cmd_list, capture_output=True, text=True, check=False)
                    is_plan_valid = False
                    for line in result.stdout.splitlines():
                        if "Plan valid" in line:
                            is_plan_valid = True
                            break
                    model.validate_generated_plan(task=task, prompt_type=prompt_type, plan_idx=i, is_valid=is_plan_valid)
                    os.remove(temp_plan_file)
                except Exception as e:
                    logger.error(f"Error validating plan for task {task._id} - Model '{model._model_name}' - Prompt Type '{prompt_type}': {e}")
                    raise e
                progress_bar.update(1)
    progress_bar.close()
    model.save_generated_plans()
    logger.info(f"Plan validation completed at {datetime.datetime.now()}.")

def compute_metrics(
    model:models.Model,
    **kwargs,
):
    data = {}
    for task in model._generated_plans:
        for prompt_type in model._generated_plans[task]:
            valid_plans = model._generated_plans[task][prompt_type]['is_valid']
            if task._domain not in data:
                data[task._domain] = {}
            
            task_size = "longer" if task._is_longer_plan else "basic"
            if task_size not in data[task._domain]:
                data[task._domain][task_size] = {}
            
            if prompt_type not in data[task._domain][task_size]:
                data[task._domain][task_size][prompt_type] = {
                    'any_valid' : 0,
                    'all_valid' : 0,
                    'num_instances' : 0,
                    'valid_ratio' : [], # list of valid ratios
                }
            
            data[task._domain][task_size][prompt_type]['num_instances'] += 1
            data[task._domain][task_size][prompt_type]['valid_ratio'].append(sum(valid_plans) / len(valid_plans))
            if any(valid_plans):
                data[task._domain][task_size][prompt_type]['any_valid'] += 1
            if all(valid_plans):
                data[task._domain][task_size][prompt_type]['all_valid'] += 1

    # --- Calculate final metrics ---
    for domain in data:
        for task_size in data[domain]:
            for prompt_type in data[domain][task_size]:
                num_instances = data[domain][task_size][prompt_type]['num_instances']
                if num_instances > 0:
                    data[domain][task_size][prompt_type]['valid_ratio'] = sum(data[domain][task_size][prompt_type]['valid_ratio']) / len(data[domain][task_size][prompt_type]['valid_ratio'])
                    data[domain][task_size][prompt_type]['any_valid'] = data[domain][task_size][prompt_type]['any_valid'] / num_instances
                    data[domain][task_size][prompt_type]['all_valid'] = data[domain][task_size][prompt_type]['all_valid'] / num_instances
                else:
                    # Handle cases with no instances to avoid division by zero
                    data[domain][task_size][prompt_type]['valid_ratio'] = 0.0
                    data[domain][task_size][prompt_type]['any_valid'] = 0.0
                    data[domain][task_size][prompt_type]['all_valid'] = 0.0

    logger.info(f"Metrics computed at {datetime.datetime.now()}.")

    # --- Prepare data for CSV and logging ---
    header = ['Model', 'Domain', 'Task Size', 'Prompt Type', 'Any Valid (%)', 'All Valid (%)', 'Avg Valid Ratio (%)']
    table_data = [header]
    for domain in data:
        for task_size in data[domain]:
            for prompt_type in data[domain][task_size]:
                metrics = data[domain][task_size][prompt_type]
                any_valid_pct = metrics['any_valid'] * 100
                all_valid_pct = metrics['all_valid'] * 100
                valid_ratio_pct = metrics['valid_ratio'] * 100
                table_data.append([
                    model._model_name,
                    domain,
                    task_size,
                    prompt_type,
                    f"{any_valid_pct:.2f}",
                    f"{all_valid_pct:.2f}",
                    f"{valid_ratio_pct:.2f}"
                ])

    # --- Save metrics to CSV ---
    csv_file_path = os.path.join(model._model_dir_path, f"metrics.csv")
    try:
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(table_data)
        logger.info(f"Metrics table saved to {csv_file_path}")
    except Exception as e:
        logger.error(f"Error saving metrics to CSV: {e}")

    # --- Print metrics in a formatted table to logger ---
    if table_data:
        # Determine column widths for pretty printing
        col_widths = [max(len(str(item)) for item in col) for col in zip(*table_data)]
        
        # Create a format string
        header_format_string = " | ".join([f"{{:<{width}}}" for width in col_widths])
        separator_line = "-+-".join(['-' * width for width in col_widths])

        logger.info("Plan Validation Metrics:")
        logger.info(header_format_string.format(*table_data[0])) # Header
        logger.info(separator_line) # Separator

        for row in table_data[1:]:
            logger.info(header_format_string.format(*row))
    else:
        logger.info("No metrics data to display.")