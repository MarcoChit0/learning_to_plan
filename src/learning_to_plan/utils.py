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
def get_tasks_from_raw_data() -> None:
    structure_json = json.load(open(config.RAW_DIR_STRUCTURE_FILE_PATH, 'r'))

    d_ord = structure_json["structure"]["order"]["domain"] 
    i_ord = structure_json["structure"]["order"]["instance"]
    domain_organization = structure_json["domain"]
    exp_instances_on_db = 0
    for d in d_ord:
        domain_info = domain_organization[d]
        if "domain" not in domain_info or "path" not in domain_info["domain"]:
            raise ValueError(f"Domain information for '{d}' is incomplete in raw directory structure file {config.RAW_DIR_STRUCTURE_FILE_PATH}. 'domain' or 'path' key is missing.")
        
        domain_file_path = os.path.join(config.RAW_DIR, domain_info["domain"]["path"])
        if not os.path.exists(domain_file_path):
            raise ValueError(f"Domain file not found: {domain_file_path}. Please ensure the domain file exists in the raw directory.")
        
        for instance_type in i_ord:
            if instance_type not in domain_info:
                continue

            assert isinstance(instance_type, str), f"In the structure of the raw directory each element except 'domain' is a type, whose type should be a string. Found type {type(instance_type)} for key '{instance_type}' in domain '{d}'."

            if instance_type.upper() not in task.Task.TYPE.__members__:
                raise ValueError(f"Invalid task type '{instance_type}' in domain '{d}'. Must be one of {list(task.Task.TYPE.__members__.keys())}.")
            
            _type = task.Task.TYPE[instance_type.upper()]

            p = os.path.join(config.RAW_DIR, domain_info[instance_type]["path"])
            pattern = domain_info[instance_type].get("regex_pattern", "instance-([0-9])+\\.pddl")
            number_of_expected_instances = domain_info[instance_type].get("number_of_instances")
            exp_instances_on_db += number_of_expected_instances
            if number_of_expected_instances is None:
                raise ValueError(f"Number of expected instances for type '{instance_type}' in domain '{d}' is not defined in the raw directory structure file {config.RAW_DIR_STRUCTURE_FILE_PATH}. Please check the structure and ensure it is defined.")
            
            already_loaded_tasks:set[task.Task] = task.task_database.get(
                filter_by_domain=d,
                filter_by_type=_type,
            )
            if len(already_loaded_tasks) > 0:
                if len(already_loaded_tasks) == number_of_expected_instances:
                    logger.info(f"Skipping domain '{d}' with type '{instance_type}' as it already has {len(already_loaded_tasks)}/{number_of_expected_instances} tasks loaded.")
                    continue
                elif len(already_loaded_tasks) > number_of_expected_instances:
                    raise ValueError(
                        f"Domain '{d}' with type '{instance_type}' has {len(already_loaded_tasks)} tasks already loaded, "
                        f"but expected {number_of_expected_instances} tasks. Please check the raw directory structure file {config.RAW_DIR_STRUCTURE_FILE_PATH}."
                    )
                else:
                    logger.warning(
                        f"Domain '{d}' with type '{instance_type}' has {len(already_loaded_tasks)}/{number_of_expected_instances} tasks already loaded. Processing remaining {number_of_expected_instances - len(already_loaded_tasks)} tasks."
                    )
            
            tasks:set[task.Task] = set()
            def sort_files(file: str, pattern : str) -> int:
                match = re.search(pattern, file)
                return int(match.group(1)) if match else float('inf')

            for file in sorted(os.listdir(p), key=lambda f: sort_files(f, pattern)):
                if re.match(pattern, file):
                    instance_file_path = os.path.join(p, file)
                    if not os.path.exists(instance_file_path):
                        raise ValueError(f"Instance file not found: {instance_file_path}. Please ensure the instance file exists in the raw directory.")
                    
                    new_task:task.Task = task.Task(
                        domain=d,
                        domain_file_path=domain_file_path,
                        instance_file_path=instance_file_path,
                        type=_type
                    )
                    found = False
                    for t in already_loaded_tasks:
                        if t.domain == new_task.domain and t.instance_file_path == new_task.instance_file_path:
                            found = True
                            break
                    if not found:
                        tasks.add(new_task)
                    else:
                        del new_task
            
            tasks = tasks.union(already_loaded_tasks)
            
            if len(tasks) == 0:
                raise ValueError(
                    f"No instances found for type '{instance_type}' in domain '{d}'. "
                    f"Please check the raw directory structure and ensure instances match the regex pattern '{pattern}'."
                )
            if len(tasks) != number_of_expected_instances:
                raise ValueError(
                    f"Expected {number_of_expected_instances} instances for type '{instance_type}' in domain '{d}', "
                    f"but found {len(tasks)} instances. Please check the raw directory structure."
                )
            expected_training_instances = domain_info[instance_type].get("number_of_instances_for_training", 0)
            expected_validation_instances = domain_info[instance_type].get("number_of_instances_for_validation", 0)
            expected_testing_instances = domain_info[instance_type].get("number_of_instances_for_testing", 0)
            if expected_training_instances + expected_validation_instances + expected_testing_instances != number_of_expected_instances:
                raise ValueError(
                    f"Expected {expected_training_instances} training, {expected_validation_instances} validation, and {expected_testing_instances} testing instances for type '{instance_type}' in domain '{d}', "
                    f"but the total does not match the number of expected instances ({number_of_expected_instances})."
                )
            logger.info(f"Found {len(tasks)} tasks for domain '{d}' with type '{instance_type}'.")

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
                f"Expected {number_of_expected_instances} instances for type '{instance_type}' in domain '{d}', " \
                f"but found {len(train_tasks) + len(validation_tasks) + len(test_tasks)} instances. " \
                f"Please check the raw directory structure and ensure instances match the regex pattern '{pattern}'."
            logger.info(f"Split {len(tasks)} tasks for domain '{d}' with type '{instance_type}' into {len(train_tasks)} training, {len(validation_tasks)} validation, and {len(test_tasks)} testing tasks.")

            try:
                task.task_database.add(set(train_tasks + validation_tasks + test_tasks))
                logger.info(f"Added {len(train_tasks)} training, {len(validation_tasks)} validation, and {len(test_tasks)} testing tasks to the task database for domain '{d}' with type '{instance_type}'.")
            except Exception as e:
                raise ValueError(f"Error adding tasks to the task database: {e}")
    if exp_instances_on_db == 0:
        raise ValueError(f"No instances found in the raw directory structure file {config.RAW_DIR_STRUCTURE_FILE_PATH}. Please check the structure and ensure instances are defined correctly.")
    tasks_on_db = task.task_database.get()
    if len(tasks_on_db) != exp_instances_on_db:
        raise ValueError(f"Expected {exp_instances_on_db} instances in the task database, but found {len(tasks_on_db)}. Please check the task database and ensure all tasks are added correctly.")


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
            filter_by_paas_status=config.STATUS.ERROR
        )

    except Exception as e:
        raise ValueError(f"Error retrieving tasks for domain '{domain}': {e}") from e

    if len(tasks_to_process) == 0:
        logger.info(f"No tasks to process for domain '{domain}'.")
        return

    logger.info(f"Must process {len(tasks_to_process)} tasks.")
    semaphore = asyncio.Semaphore(num_workers)
    await asyncio.gather(*[process_instance(t) for t in tasks_to_process])

    try:
        task.task_database.update(tasks_to_process)
        logger.info(f"Updated {len(tasks_to_process)} tasks.")
    except Exception as e:
        logger.error(f"Error adding tasks to singleton_task_database: {e}", exc_info=True)
    logger.info(f"Finished call to planning as a service at {datetime.datetime.now()}.")

import subprocess
import resource

def get_landmark_graph(t : task.Task, memory_limit: int = 24, landmark_factory:str = "lm_zg") -> None:
    # command = ./utils/downward/fast-downward.py ./data/raw/blocksworld/generated_domain.pddl ./data/raw/blocksworld/generated_basic_longer_plan_len/instance-20.pddl --search "lazy_greedy([landmark_sum(lm_zg(verbosity=debug))])"
    cmd_list = [
        './utils/downward/fast-downward.py',
        t.domain_file_path,
        t.instance_file_path,
        '--search',
        f"lazy_greedy([landmark_sum({landmark_factory}(verbosity=debug))])"
    ]
    
    def set_memory_limit(memory_limit : int):
        # Set memory limit to 2GB (adjust as needed)
        mem_in_gbyte = memory_limit * 1024 * 1024 * 1024  # 24GB in bytes
        resource.setrlimit(resource.RLIMIT_AS, (mem_in_gbyte, mem_in_gbyte))
    
    try:
        result = subprocess.run(
            cmd_list, 
            capture_output=True, 
            text=True, 
            check=True,
            preexec_fn=set_memory_limit
        )
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
