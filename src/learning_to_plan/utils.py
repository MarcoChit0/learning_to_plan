from tqdm import tqdm
from learning_to_plan.data import task
from learning_to_plan import config
import asyncio
import aiohttp
import os
import datetime
from sklearn.model_selection import train_test_split
logger = config.get_logger(__name__)
import re
import json

def process_paas_response(t: task.Task, response: dict) -> None:
    plan = ""
    status = response.get("status", "error")
    if status == "ok":
        plan = response["result"]["output"]["sas_plan"]

    t.paas_status = config.STATUS(status) if status in [e.value for e in config.STATUS] else None
    t.pddl_plan = plan

# Removed unused import 'Dataset'
# from datasets import Dataset
def get_tasks_from_domain_directory(domain: str) -> dict:
    structure_json = json.load(open(config.RAW_DIR_STRUCTURE_FILE_PATH, 'r'))
    if domain not in structure_json:
        raise ValueError(f"Domain '{domain}' not found in raw directory structure file {config.RAW_DIR_STRUCTURE_FILE_PATH}.")
    
    domain_info = structure_json[domain]
    if "domain" not in domain_info or "path" not in domain_info["domain"]:
        raise ValueError(f"Domain information for '{domain}' is incomplete in raw directory structure file {config.RAW_DIR_STRUCTURE_FILE_PATH}. 'domain' or 'path' key is missing.")
    
    domain_file_path = os.path.join(config.RAW_DIR, domain_info["domain"]["path"])
    if not os.path.exists(domain_file_path):
        raise ValueError(f"Domain file not found: {domain_file_path}. Please ensure the domain file exists in the raw directory.")
    
    domain_tasks = set()
    for key in domain_info:
        if key == "domain":
            continue

        assert isinstance(key, str), f"In the structure of the raw directory each element except 'domain' is a type, whose type should be a string. Found type {type(key)} for key '{key}' in domain '{domain}'."

        if key.upper() not in task.Task.TYPE.__members__:
            raise ValueError(f"Invalid task type '{key}' in domain '{domain}'. Must be one of {list(task.Task.TYPE.__members__.keys())}.")
        
        _type = task.Task.TYPE[key.upper()]

        p = os.path.join(config.RAW_DIR, domain_info[key]["path"])
        pattern = domain_info[key].get("regex_pattern", "instance-[0-9]+\\.pddl")
        tasks = set()
        for file in os.listdir(p):
            if re.match(pattern, file):
                instance_file_path = os.path.join(p, file)
                if not os.path.exists(instance_file_path):
                    raise ValueError(f"Instance file not found: {instance_file_path}. Please ensure the instance file exists in the raw directory.")
                
                tasks.add(task.Task(
                    domain=domain,
                    domain_file_path=domain_file_path,
                    instance_file_path=instance_file_path,
                    type=_type
                ))
        
        number_of_expected_instances = domain_info[key].get("number_of_instances", 0)
        if len(tasks) == 0:
            raise ValueError(
                f"No instances found for type '{key}' in domain '{domain}'. "
                f"Please check the raw directory structure and ensure instances match the regex pattern '{pattern}'."
            )
        if len(tasks) != number_of_expected_instances:
            raise ValueError(
                f"Expected {number_of_expected_instances} instances for type '{key}' in domain '{domain}', "
                f"but found {len(tasks)} instances. Please check the raw directory structure."
            )
        expected_training_instances = domain_info[key].get("number_of_instances_for_training", 0)
        expected_validation_instances = domain_info[key].get("number_of_instances_for_validation", 0)
        expected_testing_instances = domain_info[key].get("number_of_instances_for_testing", 0)
        if expected_training_instances + expected_validation_instances + expected_testing_instances != number_of_expected_instances:
            raise ValueError(
                f"Expected {expected_training_instances} training, {expected_validation_instances} validation, and {expected_testing_instances} testing instances for type '{key}' in domain '{domain}', "
                f"but the total does not match the number of expected instances ({number_of_expected_instances})."
            )
        logger.info(f"Found {len(tasks)} tasks for domain '{domain}' with type '{key}'.")

        tasks = sorted(tasks)
        if expected_training_instances != 0:
            train_tasks, remaining_tasks = train_test_split(
                tasks,
                train_size=expected_training_instances,
                random_state=config.RANDOM_SEED
            )
        else:
            train_tasks = set()
            remaining_tasks = tasks
        if expected_validation_instances != 0:
            validation_tasks, test_tasks = train_test_split(
                remaining_tasks,
                test_size=expected_validation_instances,
                random_state=config.RANDOM_SEED
            )
        else:
            validation_tasks = set()
            test_tasks = remaining_tasks

        for t in train_tasks:
            t.pourpose = task.Task.POURPOSE.TRAIN
        for t in validation_tasks:
            t.pourpose = task.Task.POURPOSE.VALIDATION
        for t in test_tasks:
            t.pourpose = task.Task.POURPOSE.TEST
        
        assert len(train_tasks) + len(validation_tasks) + len(test_tasks) == number_of_expected_instances, \
            f"Expected {number_of_expected_instances} instances for type '{key}' in domain '{domain}', " \
            f"but found {len(train_tasks) + len(validation_tasks) + len(test_tasks)} instances. " \
            f"Please check the raw directory structure and ensure instances match the regex pattern '{pattern}'."
        logger.info(f"Split {len(tasks)} tasks for domain '{domain}' with type '{key}' into {len(train_tasks)} training, {len(validation_tasks)} validation, and {len(test_tasks)} testing tasks.")

        domain_tasks.update(train_tasks)
        domain_tasks.update(validation_tasks)   
        domain_tasks.update(test_tasks)
    try:
        task.task_database.add(obj=domain_tasks)
    except Exception as e:
        raise ValueError(f"Error adding tasks to the task database: {e}")
    return domain_tasks


async def get_plan_from_paas(t:task.Task, solver_url="http://localhost:5001/package/lama-first/solve", max_retries=2):
    domain_content = t.read_domain()
    instance_content = t.read_instance()

    req_body = {"domain": domain_content, "problem": instance_content}
    async with aiohttp.ClientSession() as session:
        for attempt in range(1, max_retries + 2): # 1 to max_retries + 1
            try:
                async with session.post(solver_url, json=req_body) as resp:
                    solve_response = await resp.json()
                result_url = "http://localhost:5001" + solve_response["result"]
                while True:
                    async with session.post(result_url) as result_resp:
                        result_data = await result_resp.json()
                    if result_data.get("status") == "PENDING":
                        await asyncio.sleep(0.3)
                        continue
                    if result_data.get("status") == "ok":
                        output = result_data.get("result", {}).get("output", {})
                        stderr = result_data.get("result", {}).get("stderr", "")
                        if output.get("sas_plan") and stderr.strip() == "":
                            logger.info(f"Instance {t} -- Attempt {attempt} -- Success!")
                            return result_data
                    break
            except Exception as e:
                logger.warning(f"Instance {t} -- Attempt {attempt} -- Error during planning request: {e}")
            logger.info(f"Instance {t} -- Attempt {attempt} -- Retrying...")
            await asyncio.sleep(2)
        logger.warning(f"Instance {t} -- Attempt {attempt} -- Exceeded max retries.")
        return {"status": "error", "error": "Max retries exceeded or no valid plan returned."}

async def call_paas(
    domain: str,
    max_retries=2,
    num_workers=4
):
    logger.info(f"Starting call to planning as a service at {datetime.datetime.now()}.")
    async def process_instance(t: task.Task):
        async with semaphore:
            try:
                response = await get_plan_from_paas(
                    t=t,
                    max_retries=max_retries
                )
                process_paas_response(t, response)
            except Exception as e:
                raise e

    try:    
        tasks_to_process = task.task_database.get(
            filter_by_domain=domain,
        )
        if not tasks_to_process:
            logger.info(f"No tasks found for domain {domain}. Creating new tasks.")
            tasks_to_process = get_tasks_from_domain_directory(domain=domain)
        else:
            logger.info(f"Domain {domain} already exists. Loading tasks from dataset.")
            tasks_to_process = {t for t in tasks_to_process if t.paas_status == config.STATUS.ERROR}

    except Exception as e:
        logger.error(f"Error loading tasks from dataset: {e}", exc_info=True)
        logger.info(f"Creating new tasks for domain: {domain}.")
        tasks_to_process = get_tasks_from_domain_directory(domain=domain)
    if len(tasks_to_process) == 0:
        logger.info(f"No tasks to process for domain {domain}. Exiting.")
        return

    logger.info(f"Must process {len(tasks_to_process)} tasks.")
    semaphore = asyncio.Semaphore(num_workers)
    await asyncio.gather(*[process_instance(t) for t in tasks_to_process])

    try:
        task.task_database.update(tasks_to_process)
        logger.info(f"Added/ Updated {len(tasks_to_process)} tasks.")
    except Exception as e:
        logger.error(f"Error adding tasks to singleton_task_database: {e}", exc_info=True)
    logger.info(f"Finished call to planning as a service at {datetime.datetime.now()}.")

import subprocess
def get_landmark_graph() -> None:
    def call_downward(t: task.Task) -> str:
        # command = ./utils/downward/fast-downward.py ./data/raw/blocksworld/generated_domain.pddl ./data/raw/blocksworld/generated_basic_longer_plan_len/instance-20.pddl --search "lazy_greedy([landmark_sum(lm_zg(verbosity=debug))])"
        cmd_list = [
            './utils/downward/fast-downward.py',
            t.domain_file_path,
            t.instance_file_path,
            '--search',
            'lazy_greedy([landmark_sum(lm_zg(verbosity=debug))])'
        ]
        try:
            result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error calling Downward: {e.stderr.strip()}")

        reading_graph = False
        graph_lines = []
        for line in result.stdout.splitlines():
            if reading_graph:
                if line.startswith("[t="):
                    reading_graph = False
                    continue
                graph_lines.append(line)
            if "Dumping landmark graph:" in line:
                reading_graph = True
                continue
        
        if not graph_lines:
            raise ValueError("No landmark graph found in the output of Downward.")

        graph = "\n".join(graph_lines)
        # check on the start of the graph "digraph G {"
        if not graph.startswith("digraph G {"):
            raise ValueError("The landmark graph does not start with 'digraph G {'.")
        # check on the end of the graph "}"
        if not graph.endswith("}"):
            raise ValueError("The landmark graph does not end with '}'.")
        return graph

    try:
        tasks:set[task.Task] = task.task_database.get(filter_by_landmark_graph_status=config.STATUS.ERROR)
        assert len(tasks) > 0, f"No tasks found in file {config.TASKS_DATASET_FILE_PATH}."
    except Exception as e:
        logger.error(f"No tasks found in file {config.TASKS_DATASET_FILE_PATH}.", exc_info=True)
        raise e
    
    for t in tqdm(tasks, total=len(tasks), desc="Generating landmark graphs"):
        try:
            graph = call_downward(t)
            t.landmark_graph = graph
            t.landmark_graph_status = config.STATUS.OK
            task.task_database.update(t)
            logger.info(f"Generated landmark graph for task {t.id}.")
        except Exception as e:
            logger.error(f"Error generating landmark graph for task {t.id}: {e}", exc_info=True)
    