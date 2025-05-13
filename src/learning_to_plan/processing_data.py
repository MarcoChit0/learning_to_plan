# import os
# from learning_to_plan import config
# from learning_to_plan import task
# import datetime
# import subprocess
# logger = config.get_logger(__name__)
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

def validate_plans(model_name: str, domain: str, ):
    logger.info(f"Starting plan validation at {datetime.datetime.now()}.")
    assert os.path.exists(data_file_path), f"Data file {data_file_path} does not exist."

    tasks:set[task.Task] = task.get_tasks_from_jsonl(data_file_path)
    assert len(tasks) > 0, f"Data file {data_file_path} is empty."

    test_tasks = {t for t in tasks if t._type == task.Task.Type.TEST}
    assert len(test_tasks) > 0, f"No test tasks found in {data_file_path}."

    logger.info(f"Loaded {len(tasks)} tasks from {data_file_path}, of which {len(test_tasks)} are test tasks.")

    for t in test_tasks:
        logger.debug(f"Validating generated plans for task {t._id}...")
        for plan_manager in t._plan_managers:
            for prompt_type, plan in plan_manager._prompt_to_plan_mapping.items():                    
                temp_plan_file = f".temp_plan_{t._id}_{plan_manager._model_name}_{prompt_type}.txt".replace(" ", "_").replace("/", "_")
                try:
                    pddl_plan = t.convert_natural_language_plan_to_pddl(plan._content)
                except Exception as e:
                    pddl_plan = ""
                try:

                    with open(temp_plan_file, "w") as f:
                        f.write(pddl_plan)
                    logger.debug(f"Created temporary plan file: {temp_plan_file}")
                    cmd_list = [
                        "utils/VAL/build/bin/Validate",
                        "-v",
                        "-t", "0.001",
                        t._domain_file_path,
                        t._instance_file_path,
                        temp_plan_file
                    ]
                    result = subprocess.run(cmd_list, capture_output=True, text=True, check=False)
                    if not "instance-26.pddl" in t._instance_file_path:
                        os.remove(temp_plan_file)
                    is_plan_valid = False
                    for line in result.stdout.splitlines():
                        if "Plan valid" in line:
                            is_plan_valid = True
                            break
                    plan_manager.validate(prompt_type=prompt_type, is_valid=is_plan_valid)
                    if is_plan_valid:
                        logger.info(f"Task {t._id} - Model '{plan_manager._model_name}' - Prompt Type '{prompt_type}': Plan is valid.")
                    else:
                        logger.info(f"Task {t._id} - Model '{plan_manager._model_name}' - Prompt Type '{prompt_type}': Plan is invalid.")
                except Exception as e:
                    logger.error(f"Error validating plan for task {t._id} - Model '{plan_manager._model_name}' - Prompt Type '{prompt_type}': {e}")
                    raise e

    try:
        task.save_tasks_to_jsonl(tasks, data_file_path)
        logger.info(f"Finished writing to {data_file_path}.")
    except Exception as e:
        logger.error(f"Error saving tasks to JSONL: {e}")
        raise e
    
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