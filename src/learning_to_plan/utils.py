from typing import Callable

from tqdm import tqdm
from learning_to_plan import task
from learning_to_plan import config
from learning_to_plan import models
import asyncio
import aiohttp
import os
import datetime
from sklearn.model_selection import train_test_split
logger = config.get_logger(__name__)
from learning_to_plan import database

def process_paas_response(t: task.Task, response: dict) -> None:
    plan = ""
    status = response.get("status", "error")
    if status == "ok":
        plan = response["result"]["output"]["sas_plan"]
    
    t._paas_status = task.Task.PAAS_STATUS(status) if status in [e.value for e in task.Task.PAAS_STATUS] else None
    t._pddl_plan = plan

# Removed unused import 'Dataset'
# from datasets import Dataset
def get_tasks_from_domain_directory(domain: str) -> set[task.Task]:
    # Assuming config.RAW_DIR, config.DOMAIN_FILE_NAME, config.BASIC_INSTANCES, config.LONG_INSTANCES are defined
    domain_file_path = os.path.join(config.RAW_DIR, domain, config.DOMAIN_FILE_NAME)

    # Determine which instance directories to include
    instance_dirs = [
        os.path.join(config.RAW_DIR, domain, config.BASIC_INSTANCES), # training, validation, test instances
        os.path.join(config.RAW_DIR, domain, config.LONG_INSTANCES) # out of distribution instances
    ]
    # Collect tasks
    tasks:set[task.Task] = set()
    for instance_dir in instance_dirs:
        if not os.path.exists(instance_dir):
            raise ValueError(f"Instance directory not found: {instance_dir}")

        for file_name in os.listdir(instance_dir):
            if task.INSTANCE_PATTERN.search(file_name):
                tasks.add(
                    task.Task(
                        domain=domain,
                        domain_file_path=domain_file_path,
                        instance_file_path=os.path.join(instance_dir, file_name)
                    )
                )

    return tasks

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
        logger.info(f"Domain {domain} already exists. Loading tasks from dataset.")
        tasks_to_process = {t for t in database.get_tasks(filter_by_domain=domain) if t._paas_status == task.Task.PAAS_STATUS.ERROR}
    except Exception as e:
        logger.error(f"Error loading tasks from dataset: {e}", exc_info=True)
        logger.info(f"Creating new tasks for domain: {domain}.")
        tasks_to_process = get_tasks_from_domain_directory(domain=domain)
    
    logger.info(f"Must process {len(tasks_to_process)} tasks.")
    semaphore = asyncio.Semaphore(num_workers)
    await asyncio.gather(*[process_instance(t) for t in tasks_to_process])

    database.save()
    logger.info(f"Finished call to planning as a service at {datetime.datetime.now()}.")


def split_dataset(
):
    logger.info(f"Starting to build finetuning dataset at {datetime.datetime.now()}.")

    try:
        tasks = database.get_dataset()
        assert len(tasks) > 0, f"No tasks found in file {config.TASKS_DATASET_FILE_PATH}."
    except Exception as e:
        logger.error(f"No tasks found in file {config.TASKS_DATASET_FILE_PATH}.", exc_info=True)
        raise e

    valid_tasks:set[task.Task] = {t for t in tasks if t._paas_status == task.Task.PAAS_STATUS.OK}
    domains:set[str] = {t._domain for t in valid_tasks}
    tasks_per_domain = {d: [t for t in valid_tasks if t._domain == d] for d in domains}

    for d in domains:
        logger.info(f"Splitting tasks in domain: {d}")
        assert len(tasks_per_domain[d]) == 4400, f"Data file must contain at least 4400 valid instances, but only {len(valid_tasks)} instances were found."
        
        # Extract longer plans and basic plans
        longer_tasks = [t for t in tasks_per_domain[d] if t._is_longer_plan]
        basic_tasks = [t for t in tasks_per_domain[d] if not t._is_longer_plan]

        assert len(longer_tasks) == 200, f"Expected 200 instances with 'is_longer_plan' as True, but found {len(longer_tasks)} instances."
        assert len(basic_tasks) == 4200, f"Expected 4200 instances with 'is_longer_plan' as False, but found {len(basic_tasks)} instances."

        sorted_basic_tasks = sorted(basic_tasks)
    
        train_tasks, temp_tasks = train_test_split(
            sorted_basic_tasks, test_size=1000, random_state=config.RANDOM_SEED
        )
        validation_tasks, basic_test_tasks = train_test_split(
            temp_tasks, test_size=200, random_state=config.RANDOM_SEED
        )
        test_tasks = longer_tasks + basic_test_tasks
        for t in train_tasks:
            t._type = task.Task.TYPE.TRAIN
        for t in validation_tasks:
            t._type = task.Task.TYPE.VALIDATION
        for t in test_tasks:
            t._type = task.Task.TYPE.TEST
        
        # Save the final dataset
        database.save()
    logger.info(f"Finished building finetuning dataset at {datetime.datetime.now()}.")

import subprocess
def get_landmark_graph() -> None:
    def call_downward(t: task.Task) -> str:
        # command = ./utils/downward/fast-downward.py ./data/raw/blocksworld/generated_domain.pddl ./data/raw/blocksworld/generated_basic_longer_plan_len/instance-20.pddl --search "lazy_greedy([landmark_sum(lm_zg(verbosity=debug))])"
        cmd_list = [
            './utils/downward/fast-downward.py',
            t._domain_file_path,
            t._instance_file_path,
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
        tasks = database.get_dataset()
        assert len(tasks) > 0, f"No tasks found in file {config.TASKS_DATASET_FILE_PATH}."
    except Exception as e:
        logger.error(f"No tasks found in file {config.TASKS_DATASET_FILE_PATH}.", exc_info=True)
        raise e
    domains:set[str] = {t._domain for t in tasks}
    tasks_per_domain = {d: [t for t in tasks if t._domain == d] for d in domains}

    for d in domains:
        if not tasks_per_domain[d]:
            logger.warning(f"No valid tasks found for domain {d}. Skipping landmark graph generation for this domain.")
            continue
        logger.info(f"Generating landmark graphs for domain: {d}")

        tasks_to_process = {t for t in tasks_per_domain[d] if t._landmark_graph_status != task.Task.LANDMARK_GRAPH_STATUS.OK}
        if not tasks_to_process:
            logger.info("All tasks already have a landmark graph. No processing needed.")
            continue

        if len(tasks_to_process) != len(tasks_per_domain[d]):
            logger.info(f"{len(tasks_per_domain[d]) - len(tasks_to_process)} tasks already have a landmark graph. Processing {len(tasks_to_process)} tasks.")
        else:
            logger.info(f"No tasks have a landmark graph. Processing all {len(tasks_per_domain[d])} tasks.")

        for t in tqdm(tasks_to_process, total=len(tasks_to_process), desc="Generating landmark graphs"):
            try:
                graph = call_downward(t)
                t._landmark_graph = graph
                t._landmark_graph_status = task.Task.LANDMARK_GRAPH_STATUS.OK
            except Exception as e:
                t._landmark_graph_status = task.Task.LANDMARK_GRAPH_STATUS.ERROR
                logger.error(f"Error generating landmark graph for task {t._id}: {e}", exc_info=True)
        
        database.save()

def get_model_names_from_models_dir():
    """
    Go through all models names and compute the metrics for each one of them.
    A model names is in the following format:
        config.MODELS_DIR/model_name/config.GENERATED_PLANS_FILE_NAME,

    where model_name can be separated by '/' as well.
    """
    if not os.path.exists(config.MODELS_DIR):
        raise FileNotFoundError(f"Models directory '{config.MODELS_DIR}' not found.")
    
    model_names = []
    for root, dirs, files in os.walk(config.MODELS_DIR):
        for file in files:
            if file == config.GENERATED_PLANS_FILE_NAME:
                model_path = os.path.join(root, file)
                model_name = os.path.relpath(model_path, config.MODELS_DIR)
                model_name = os.path.dirname(model_name)
                model_names.append(model_name)
    
    logger.info(f"Found {len(model_names)} models in '{config.MODELS_DIR}' directory.")
    return model_names

def apply_function_to_all_models(
    function: Callable[..., None],
    **fn_kwargs
):
    model_names = get_model_names_from_models_dir()
    if not model_names:
        logger.warning(f"No models found in the {config.MODELS_DIR} directory.")
        return
    logger.info(f"Applying function '{function.__name__}' to all models...")
    for model_name in model_names:
        try:
            model = models.get_model(model_name=model_name)
            if not model:
                logger.warning(f"Model '{model_name}' not found or could not be loaded.")
                continue
        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {e}", exc_info=True)
            continue
        try:
            function(model=model, **fn_kwargs)
        except Exception as e:
            raise ValueError(f"Error applying function '{function.__name__}' to model '{model_name}': {e}") from e