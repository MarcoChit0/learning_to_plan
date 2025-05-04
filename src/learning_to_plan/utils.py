from learning_to_plan import task
from learning_to_plan import config
import asyncio
import aiohttp
import os
import datetime
from sklearn.model_selection import train_test_split
logger = config.get_logger(__name__)

async def get_plan_from_paas(task:task.Task, solver_url="http://localhost:5001/package/lama-first/solve", max_retries=2):
    domain_content = task.read_domain()
    instance_content = task.read_instance()

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
                            logger.info(f"Instance {task} -- Attempt {attempt} -- Success!")
                            return result_data
                    break
            except Exception as e:
                logger.warning(f"Instance {task} -- Attempt {attempt} -- Error during planning request: {e}")
            logger.info(f"Instance {task} -- Attempt {attempt} -- Retrying...")
            await asyncio.sleep(2)
        logger.warning(f"Instance {task} -- Attempt {attempt} -- Exceeded max retries.")
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
                    task=t,
                    max_retries=max_retries
                )
                t.process_paas_response(response)
            except Exception as e:
                raise e

    try:
        logger.info(f"Domain {domain} already exists. Loading tasks from dataset.")
        tasks_to_process = {t for t in task.get_tasks(filter_by_domain=domain) if t._status == task.Task.PlanningAsAServiceStatus.ERROR}
    except Exception as e:
        logger.error(f"Error loading tasks from dataset: {e}", exc_info=True)
        logger.info(f"Creating new tasks for domain: {domain}.")
        tasks_to_process = task.get_tasks_from_domain_directory(domain=domain)
    
    logger.info(f"Must process {len(tasks_to_process)} tasks.")
    semaphore = asyncio.Semaphore(num_workers)
    await asyncio.gather(*[process_instance(t) for t in tasks_to_process])

    task.save()
    logger.info(f"Finished call to planning as a service at {datetime.datetime.now()}.")


def split_dataset(
    random_seed=42
):
    logger.info(f"Starting to build finetuning dataset at {datetime.datetime.now()}.")

    try:
        tasks = task.get_dataset()
        assert len(tasks) > 0, f"No tasks found in file {config.TASKS_DATASET_FILE_PATH}."
    except Exception as e:
        logger.error(f"No tasks found in file {config.TASKS_DATASET_FILE_PATH}.", exc_info=True)
        raise e

    valid_tasks:set[task.Task] = {t for t in tasks if t._status == task.Task.PlanningAsAServiceStatus.OK}
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
            sorted_basic_tasks, test_size=1000, random_state=random_seed
        )
        validation_tasks, basic_test_tasks = train_test_split(
            temp_tasks, test_size=200, random_state=random_seed
        )
        test_tasks = longer_tasks + basic_test_tasks
        for t in train_tasks:
            t._type = task.Task.Type.TRAIN
        for t in validation_tasks:
            t._type = task.Task.Type.VALIDATION
        for t in test_tasks:
            t._type = task.Task.Type.TEST
        
        # Save the final dataset
        logger.info(f"Writing {len(tasks)} tasks to {data_file_path}.")
        task.save(data_file_path)
        logger.info(f"Finished writing {len(tasks)} tasks to {data_file_path}.")
        logger.info(f"Finished splitting tasks in domain: {d}.")
    logger.info(f"Finished building finetuning dataset at {datetime.datetime.now()}.")