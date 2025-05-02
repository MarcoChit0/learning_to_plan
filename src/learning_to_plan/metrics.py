from learning_to_plan import config
from learning_to_plan import task
import datetime
import os

logger = config.get_logger(__name__)

def compute_metrics(data_file_path: str):
    logger.info(f"Starting metrics computation at {datetime.datetime.now()}.")
    assert os.path.exists(data_file_path), f"Data file {data_file_path} does not exist."

    tasks:set[task.Task] = task.get_tasks_from_jsonl(data_file_path)
    assert len(tasks) > 0, f"Data file {data_file_path} is empty."
    logger.info(f"Loaded {len(tasks)} tasks from {data_file_path}.")

    test_tasks = {t for t in tasks if t.type == task.Task.TaskType.TEST}
    assert len(test_tasks) > 0, f"No test tasks found in {data_file_path}."
    logger.info(f"Found {len(test_tasks)} test tasks.")

    # -1. for each task, if a plan is valid, add 1 to the number of valid plans. Divide this number by the number of tasks
    metric_1 = {}
    for t in test_tasks:
        for model in t._plan_managers.keys():
            if model not in metric_1:
                metric_1[model] = {"valid_tasks": 0, "total_tasks": 0}
            metric_1[model]["total_tasks"] += 1
            for i, nlp_class in enumerate(t._plan_managers[model]):
                if nlp_class._is_valid:
                    metric_1[model]["valid_tasks"] += 1
                    break
    for model in metric_1.keys():
        metric_1[model]["ratio"] = metric_1[model]["valid_tasks"] / metric_1[model]["total_tasks"]

    logger.info(f"Metric 1: {metric_1}")

    # -2. for each task, take the ratio of valid plans over the number of plans. Compute the mean and std of this ratio.
    metric_2 = {}
    for t in test_tasks:
        for model in t._plan_managers.keys():
            if model not in metric_2:
                metric_2[model] = {"ratio_per_task": [], "total_tasks": 0}
            metric_2[model]["total_tasks"] += 1
            valid_plans = 0
            for i, nlp_class in enumerate(t._plan_managers[model]):
                if nlp_class._is_valid:
                    valid_plans += 1
            metric_2[model]["ratio_per_task"].append(valid_plans/len(t._plan_managers[model]))

    for model in metric_2.keys():
        metric_2[model]["mean"] = sum(metric_2[model]["ratio_per_task"]) / len(metric_2[model]["ratio_per_task"])
        metric_2[model]["std"] = (sum([(x - metric_2[model]["mean"]) ** 2 for x in metric_2[model]["ratio_per_task"]]) / len(metric_2[model]["ratio_per_task"])) ** 0.5

    logger.info(f"Metric 2: {metric_2}")