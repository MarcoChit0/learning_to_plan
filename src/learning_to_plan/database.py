from learning_to_plan import config
from learning_to_plan.task import Task
import json
import os
from typing import Optional
from learning_to_plan import content
logger = config.get_logger(__name__)


# TODO: UPDATE TO USE SQL INSTEAD OF JSONL
# TODO: CREATE A TABLE FOR CONTENT AS WELL
# TODO: MAKE MOST OF METHODS REUSABLE FOR CONTENT AS WELL
# TODO: WHEN CONTENT USES THE DATABASE, ITS PROBLEM WITH IDS WOULD BE SOLVED
TASK_DATABASE: set[Task] = set()
def get_task_database() -> set[Task]:
    global TASK_DATABASE
    if not TASK_DATABASE:
        raise ValueError("Dataset is empty. Please load the dataset first.")
    return TASK_DATABASE

def get_tasks(filter_by_domain: Optional[str] = None,  filter_by_task_type: Optional[Task.TYPE] = None, is_longer_plan:Optional[bool] = None, number_of_instances: Optional[int] = None) -> set[Task]:
    global TASK_DATABASE
    if not TASK_DATABASE:
        raise ValueError("Dataset is empty. Please load the dataset first.")
    tasks = TASK_DATABASE
    # --- Filter by domain and type ---
    if filter_by_domain:
        tasks = {t for t in tasks if t._domain == filter_by_domain}
        if len(tasks) == 0:
            raise ValueError(f"No tasks found for domain '{filter_by_domain}'.")
    if filter_by_task_type:
        tasks = {t for t in tasks if t._type == filter_by_task_type}
        if len(tasks) == 0:
            raise ValueError(f"No tasks found for type '{filter_by_task_type}'.")
    # --- Filter by basic or long tasks ---
    if is_longer_plan is not None:
        tasks = {t for t in tasks if t._is_longer_plan == is_longer_plan}
        if len(tasks) == 0:
            raise ValueError(f"No tasks found with is_longer_plan={is_longer_plan}.")
    # --- Limit number of instances ---
    if number_of_instances is not None and isinstance(number_of_instances, int):
        tasks = set(sorted(tasks)[:min(number_of_instances, len(tasks))])
        if len(tasks) == 0:
            raise ValueError(f"No tasks found after filtering.")
    return tasks

def get_task(domain_file_path: str, instance_file_path: str) -> Task:
    global TASK_DATABASE
    if not TASK_DATABASE:
        raise ValueError("Dataset is empty. Please load the dataset first.")
    for task in TASK_DATABASE:
        if task._domain_file_path == domain_file_path and task._instance_file_path == instance_file_path:
            return task
    raise ValueError(f"Task with domain file path '{domain_file_path}' and instance file path '{instance_file_path}' not found.")

def load_tasks() -> None:
    jsonl_file_path = config.TASKS_DATASET_FILE_PATH
    global TASK_DATABASE
    if not os.path.exists(jsonl_file_path):
        raise ValueError(f"JSONL file not found: {jsonl_file_path}")
    tasks = set()
    logger.info(f"Loading tasks from {jsonl_file_path}.")
    with open(jsonl_file_path, "r", encoding='utf-8') as f:
        for line in f:
            try:
                json_obj = json.loads(line)
                domain = json_obj.get("domain", None)
                instance_file_path = json_obj.get("instance_file_path", None)
                domain_file_path = json_obj.get("domain_file_path", None)
                assert domain, "Domain is not specified in the JSON object."
                assert instance_file_path, "Instance file path is not specified in the JSON object."
                assert domain_file_path, "Domain file path is not specified in the JSON object."
                task = Task(
                    domain,
                    domain_file_path,
                    instance_file_path
                )
                task.from_json(json_obj)
                tasks.add(task)
            except Exception as e:
                m = f"Error processing task from file {jsonl_file_path}: {e}"
                # Changed config.log to logger.error
                logger.error(m)
                raise e
    TASK_DATABASE = tasks
    logger.info(f"Loaded {len(TASK_DATABASE)} tasks from {jsonl_file_path}.")

def save_tasks()-> None:
    jsonl_file_path = config.TASKS_DATASET_FILE_PATH
    global TASK_DATABASE
    if not TASK_DATABASE:
        raise ValueError("Dataset is empty. Please load the dataset first.")
    logger.info(f"Saving {len(TASK_DATABASE)} tasks to {jsonl_file_path}.")
    with open(jsonl_file_path, "w", encoding='utf-8') as f:
        for task in sorted(TASK_DATABASE):
            try:
                json_str = task.to_json() # Get the JSON string representation
                f.write(json_str + "\n") # Write the JSON string followed by a newline
            except Exception as e:
                m = f"Error saving task to file {jsonl_file_path}: {e}"
                # Changed config.log to logger.error
                logger.error(m)
                raise e
    logger.info(f"Saved {len(TASK_DATABASE)} tasks to {jsonl_file_path}.")


