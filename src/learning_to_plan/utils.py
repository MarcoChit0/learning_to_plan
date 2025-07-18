from tqdm import tqdm
from learning_to_plan.data import task
from learning_to_plan import config
import asyncio
import aiohttp
import os
import datetime
from sklearn.model_selection import train_test_split
logger = config.get_logger(__name__)
from learning_to_plan import database
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
    acc = 0
    tasks_to_add_on_db = set()
    
    for d in d_ord:
        domain_info = domain_organization[d]
        if "domain" not in domain_info or "path" not in domain_info["domain"]:
            raise ValueError(f"Domain information for '{d}' is incomplete in raw directory structure file {config.RAW_DIR_STRUCTURE_FILE_PATH}. 'domain' or 'path' key is missing.")
        domain_file_path = os.path.join(config.RAW_DIR, domain_info["domain"]["path"])
        if not os.path.exists(domain_file_path):
            raise ValueError(f"Domain file not found: {domain_file_path}. Please ensure the domain file exists in the raw directory.")
        
        for task_type in i_ord:
            if task_type not in domain_info:
                continue

            assert isinstance(task_type, str), f"In the structure of the raw directory each element except 'domain' is a type, whose type should be a string. Found type {type(task_type)} for key '{task_type}' in domain '{d}'."
            if task_type.upper() not in task.Task.TYPE.__members__:
                raise ValueError(f"Invalid task type '{task_type}' in domain '{d}'. Must be one of {list(task.Task.TYPE.__members__.keys())}.")
            _type = task.Task.TYPE[task_type.upper()]

            p = os.path.join(config.RAW_DIR, domain_info[task_type]["path"])
            pattern = domain_info[task_type].get("regex_pattern", "instance-([0-9])+\\.pddl")

            number_of_instances = domain_info[task_type].get("number_of_instances", None)
            if number_of_instances is None:
                raise ValueError(f"Number of expected instances for type '{task_type}' in domain '{d}' is not defined in the raw directory structure file {config.RAW_DIR_STRUCTURE_FILE_PATH}. Please check the structure and ensure it is defined.")
            num_tasks = number_of_instances.get("total", 0)
            num_train_tasks = number_of_instances.get("train", 0)
            num_validation_tasks = number_of_instances.get("validation", 0)
            num_test_tasks = number_of_instances.get("test", 0)
            if num_tasks == 0:
                raise ValueError(
                    f"Number of expected instances for type '{task_type}' in domain '{d}' is set to 0 in the raw directory structure file {config.RAW_DIR_STRUCTURE_FILE_PATH}. "
                    f"Please check the structure and ensure it is defined correctly."
                )
            if num_train_tasks + num_validation_tasks + num_test_tasks != num_tasks:
                raise ValueError(
                    f"Expected {num_train_tasks} training, {num_validation_tasks} validation, and {num_test_tasks} testing instances for type '{task_type}' in domain '{d}', "
                    f"but the total does not match the number of expected instances ({num_tasks}). Please check the raw directory structure."
                )
            acc += num_tasks

            tasks_on_db:set[task.Task] = database.task_database.get(
                filter_by_domain=d,
                filter_by_type=_type,
            )
            if len(tasks_on_db) > 0:
                if len(tasks_on_db) == num_tasks:
                    logger.info(f"Skipping domain '{d}' with type '{task_type}' as it already has {len(tasks_on_db)}/{num_tasks} tasks loaded.")
                    continue
                elif len(tasks_on_db) > num_tasks:
                    raise ValueError(
                        f"Domain '{d}' with type '{task_type}' has {len(tasks_on_db)} tasks already loaded, "
                        f"but expected {num_tasks} tasks. Please check the raw directory structure file {config.RAW_DIR_STRUCTURE_FILE_PATH}."
                    )
                else:
                    logger.warning(
                        f"Domain '{d}' with type '{task_type}' has {len(tasks_on_db)}/{num_tasks} tasks already loaded. Processing remaining {num_tasks - len(tasks_on_db)} tasks."
                    )
            
            new_tasks:set[task.Task] = set()
            def sort_files(file: str, pattern : str) -> int:
                match = re.search(pattern, file)
                return int(match.group(1)) if match else float('inf')

            for file in sorted(os.listdir(p), key=lambda f: sort_files(f, pattern)):
                if re.match(pattern, file):
                    instance_file_path = os.path.join(p, file)
                    if not os.path.exists(instance_file_path):
                        raise ValueError(f"Instance file not found: {instance_file_path}. Please ensure the instance file exists in the raw directory.")
                    
                    found = False
                    for t in tasks_on_db:
                        if t.instance_file_path == instance_file_path:
                            found = True
                            break

                    if not found:
                        new_tasks.add(task.Task(
                            domain=d,
                            domain_file_path=domain_file_path,
                            instance_file_path=instance_file_path,
                            type=_type
                        ))
            
            new_tasks = new_tasks.union(tasks_on_db)
            if len(new_tasks) == 0 or len(new_tasks) != num_tasks:
                raise ValueError(
                    f"Expected {num_tasks} instances for type '{task_type}' in domain '{d}', "
                    f"but found {len(new_tasks)} instances. Please check the raw directory structure and ensure instances match the regex pattern '{pattern}'."
                )
            logger.info(f"Found {len(new_tasks)} tasks for domain '{d}' with type '{task_type}'.")

            new_tasks = sorted(new_tasks)
            train_tasks, validation_tasks, test_tasks = [], [], []
            if num_train_tasks != 0:
                if num_validation_tasks != 0:
                    train_tasks, temp_tasks = train_test_split(
                        new_tasks, test_size=num_validation_tasks + num_test_tasks, random_state=config.RANDOM_SEED
                    )
                    validation_tasks, test_tasks = train_test_split(
                        temp_tasks, test_size=num_test_tasks, random_state=config.RANDOM_SEED
                    )
                else:
                    train_tasks, test_tasks = train_test_split(
                        new_tasks, test_size=num_test_tasks, random_state=config.RANDOM_SEED
                    )
            elif num_validation_tasks != 0:
                validation_tasks, test_tasks = train_test_split(
                    new_tasks, test_size=num_test_tasks, random_state=config.RANDOM_SEED
                )
            elif num_test_tasks != 0:
                test_tasks = list(new_tasks)

            
            assert len(train_tasks) + len(validation_tasks) + len(test_tasks) == num_tasks, \
                f"Expected {num_tasks} instances for type '{task_type}' in domain '{d}', " \
                f"but found {len(train_tasks) + len(validation_tasks) + len(test_tasks)} instances. " \
                f"Please check the raw directory structure and ensure instances match the regex pattern '{pattern}'."
            logger.info(f"Split {len(new_tasks)} tasks for domain '{d}' with type '{task_type}' into {len(train_tasks)} training, {len(validation_tasks)} validation, and {len(test_tasks)} testing tasks.")

            for t in train_tasks:
                t.purpose = task.Task.purpose.TRAIN
            for t in validation_tasks:
                t.purpose = task.Task.purpose.VALIDATION
            for t in test_tasks:
                t.purpose = task.Task.purpose.TEST

            tasks_to_add_on_db.update(set(train_tasks + validation_tasks + test_tasks))
            if len(tasks_to_add_on_db) != acc:
                raise ValueError(
                    f"Expected {acc} tasks to be added to the database for domain '{d}' with type '{task_type}', "
                    f"but found {len(tasks_to_add_on_db)} tasks. Please check the raw directory structure and ensure all tasks are defined correctly."
                )
            
            logger.info(f"Added {len(train_tasks)} training, {len(validation_tasks)} validation, and {len(test_tasks)} testing tasks to the task database for domain '{d}' with type '{task_type}'.")

    if acc == 0 or len(tasks_to_add_on_db) == 0:
        raise ValueError(f"No instances found in the raw directory structure file {config.RAW_DIR_STRUCTURE_FILE_PATH}. Please check the structure and ensure instances are defined correctly.")
    try:
        database.task_database.add(tasks_to_add_on_db)
        logger.info(f"Added/ Updated {len(tasks_to_add_on_db)} tasks to the task database.")
    except Exception as e:
        raise ValueError(f"Error adding tasks to the task database: {e}")
    tasks_on_db: set[task.Task] = database.task_database.get()
    if len(tasks_on_db) != acc:
        raise ValueError(f"Expected {acc} instances in the task database, but found {len(tasks_on_db)}. Please check the task database and ensure all tasks are added correctly.")


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
        tasks_to_process = database.task_database.get(
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
        database.task_database.update(tasks_to_process)
        logger.info(f"Updated {len(tasks_to_process)} tasks.")
    except Exception as e:
        logger.error(f"Error adding tasks to singleton_task_database: {e}", exc_info=True)
    logger.info(f"Finished call to planning as a service at {datetime.datetime.now()}.")

def get_selected_domains(domains : str) -> set[str]:
    tasks = database.task_database.get()
    available_domains = {t.domain for t in tasks}
    logger.info(f"Available domains: {', '.join(available_domains)}")

    if domains.lower() == "all":
        logger.info("Processing all available domains.")
        return available_domains
    else:
        selected = set(s.strip() for s in domains.split(","))
        assert selected.issubset(available_domains), f"Selected domains {selected} are not in available domains {available_domains}."
        selected = selected.intersection(available_domains)
        assert len(selected) > 0, f"No valid domains selected from {domains}."
        logger.info(f"Processing selected domains: {', '.join(selected)}")
        return selected

def run_on_domains(
    domains : set[str],
    fn: callable,
    raise_on_error: bool = True,
    **kwargs
):
    for domain in domains:
        logger.info(f"Applying function to domain: {domain} at {datetime.datetime.now()}.")
        try:
            fn(domain=domain, **kwargs)
        except Exception as e:
            logger.error(f"Error occurred while processing domain {domain}: {e}", exc_info=True)
            if raise_on_error:
                raise ValueError(f"Error occurred while processing domain {domain}: {e}") from e
        logger.info(f"Finished processing domain: {domain} at {datetime.datetime.now()}.")