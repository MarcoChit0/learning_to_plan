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