import os
from typing import Optional
from learning_to_plan import config, models
from learning_to_plan import task
import datetime
import subprocess
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

def validate_plans(model_name: str, **validation_kwargs):
    logger.info(f"Starting plan validation at {datetime.datetime.now()}.")

    validation_kwargs['is_trainable'] = False
    model = models.get_model(model_name=model_name, **validation_kwargs)
    
    for task in model._generated_plans:
        for prompt_type in model._generated_plans[task]:
            pddl_plans = model._generated_plans[task][prompt_type]['pddl_plans']
            for i, pddl_plan in enumerate(pddl_plans):
                temp_plan_file = os.path.join(model._model_dir_path, f".temp_plan_{model._model_name}_{task._domain_file_path}_{task._instance_file_path}_{prompt_type}_{i}.txt".replace(" ", "_").replace("/", "_"))
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
                    model.validate_generated_plan(task=task, prompt_type=prompt_type, plan=pddl_plan, is_valid=is_plan_valid)
                    # TODO: remove this later
                    with open(temp_plan_file, mode="a") as f:
                        f.write("\n")
                        for line in result.stdout.splitlines():
                            f.write(line + "\n")
                        if is_plan_valid:
                            f.write("--> Plan is valid.\n")
                        else:
                            f.write("--> Plan is invalid.\n")
                    # TODO: when this method is ok, remove the temp file
                except Exception as e:
                    logger.error(f"Error validating plan for task {task._id} - Model '{model._model_name}' - Prompt Type '{prompt_type}': {e}")
                    raise e
    model.save_generated_plans()
    logger.info(f"Plan validation completed at {datetime.datetime.now()}.")


# from learning_to_plan import config
# from learning_to_plan import task
# import datetime
# import os

# logger = config.get_logger(__name__)

# def compute_metrics(data_file_path: str):
#     logger.info(f"Starting metrics computation at {datetime.datetime.now()}.")
#     assert os.path.exists(data_file_path), f"Data file {data_file_path} does not exist."

#     tasks:set[task.Task] = task.get_tasks_from_jsonl(data_file_path)
#     assert len(tasks) > 0, f"Data file {data_file_path} is empty."
#     logger.info(f"Loaded {len(tasks)} tasks from {data_file_path}.")

#     test_tasks = {t for t in tasks if t._type == task.Task.Type.TEST}
#     assert len(test_tasks) > 0, f"No test tasks found in {data_file_path}."
#     logger.info(f"Found {len(test_tasks)} test tasks.")

#     # -1. For each model, For each prompt type, For each task: if a plan is valid, add 1 to the number of valid plans. Divide this number by the number of tasks. Aggregate the results. MODEL_NAME, PROMPT_TYPE, VALIDITY
#     data = {}
#     for t in test_tasks:
#         for plan_manager in t._plan_managers:

#             if plan_manager._model_name not in data:
#                 data[plan_manager._model_name] = {}
            
#             for prompt_type, plan in plan_manager._prompt_to_plan_mapping.items():

#                 if prompt_type not in data[plan_manager._model_name]:
#                     data[plan_manager._model_name][prompt_type] = {
#                         "valid_plans": 0,
#                         "total_plans": 0
#                     }
                
#                 # Check if the plan is valid
#                 if plan._is_valid:
#                     data[plan_manager._model_name][prompt_type]["valid_plans"] += 1
#                 data[plan_manager._model_name][prompt_type]["total_plans"] += 1
    
#     for model_name, prompt_data in data.items():
#         for prompt_type, metrics in prompt_data.items():
#             if metrics["total_plans"] > 0:
#                 metrics["validity"] = metrics["valid_plans"] / metrics["total_plans"]
#             else:
#                 metrics["validity"] = 0.0
#             logger.info(f"Model: {model_name}, Prompt Type: {prompt_type}, Validity: {metrics['valid_plans']}/{metrics['total_plans']} = {metrics['validity']:.2f}%")