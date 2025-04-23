import learning_to_plan.task as task
import asyncio
import aiohttp
import os
import learning_to_plan.config as config
import datetime

async def get_plan_from_paas(domain_content, instance_content, instance_name, solver_url="http://localhost:5001/package/lama-first/solve", max_retries=3):
    req_body = {"domain": domain_content, "problem": instance_content}
    async with aiohttp.ClientSession() as session:
        for attempt in range(1, max_retries + 1):
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
                            config.logging.info(f"Instance {instance_name} -- Attempt {attempt} -- Success!")
                            return result_data
                    break
            except Exception as e:
                config.logging.warning(f"Instance {instance_name} -- Attempt {attempt} -- Error during planning request: {e}")
            config.logging.info(f"Instance {instance_name} -- Attempt {attempt} -- Retrying...")
            await asyncio.sleep(2)
        config.logging.warning(f"Instance {instance_name} -- Attempt {attempt} -- Exceeded max retries.")
        return {"status": "error", "error": "Max retries exceeded or no valid plan returned."}

async def call_paas(
    tasks: set[task.Task],
    data_file_path: str,
    max_retries=3,
    num_workers=4,
    overwrite=False
):
    config.logging.info(f"Starting call to planning as a service at {datetime.datetime.now()}.")
    async def process_instance(task):
        async with semaphore:
            try:
                response = await get_plan_from_paas(
                    domain_content=task.read_domain(),
                    instance_content=task.read_instance(),
                    instance_name=task._instance,
                    max_retries=max_retries
                )
                task.update_status(response)
            except Exception as e:
                raise e

    processed_tasks = set()
    if os.path.exists(data_file_path):
        if overwrite:
            config.logging.warning(f"Overwriting existing dataset at {data_file_path}.")
        else:
            config.logging.info(f"Loading existing dataset at {data_file_path}. Skipping recalculation for already processed tasks.")
            processed_tasks = task.get_tasks_from_jsonl(data_file_path)
            # filter out for the tasks that have status "ok"
            processed_tasks = {t for t in processed_tasks if t.status == "ok"}
            tasks = tasks - processed_tasks
    semaphore = asyncio.Semaphore(num_workers)
    await asyncio.gather(*[process_instance(task) for task in tasks])

    processed_tasks.update(tasks)
    config.create_necessary_dirs(data_file_path)
    with open(data_file_path, "w") as json_file:
        for task in sorted(processed_tasks):
            json_file.write(task.to_jsonl() + "\n")
    config.logging.info(f"Finished writing to {data_file_path}.")

    config.logging.info(f"Finished call to planning as a service at {datetime.datetime.now()}.")

from sklearn.model_selection import train_test_split
import pandas as pd

def split_dataset(
    data_file_path,
    random_seed=42
):
    config.log(f"Starting to build finetuning dataset at {datetime.datetime.now()}.", level=config.logging.INFO)
    if not os.path.exists(data_file_path):
        e = f"Data file not found: {data_file_path}"
        config.log(e, level=config.logging.ERROR)
        raise ValueError(e)

    df = pd.read_json(data_file_path, lines=True)
    # Filter valid rows and separate into longer and basic plans
    valid_mask = (df["status"] == "ok") & (df["plan"].notna()) & (df["plan"].str.strip() != "")
    df_valid = df[valid_mask]

    if not len(df_valid) == 4400:
        e = f"Data file must contain at least 4400 valid instances, but only {len(df_valid)} instances were found."
        config.log(e, level=config.logging.ERROR)
        raise ValueError(e)

    # Extract longer plans and basic plans
    longer_df = df_valid[df_valid["is_longer_plan"] == True]
    basic_df = df_valid[df_valid["is_longer_plan"] == False]
    
    # Check if exactly 200 rows have is_longer_plan as True
    if len(longer_df) != 200:
        e = f"Expected 200 instances with 'is_longer_plan' as True, but found {len(longer_df)} instances."
        config.log(e, level=config.logging.ERROR)
        raise ValueError(e)
    
    # Only split the basic dataset
    train_df, temp_df = train_test_split(
        basic_df, test_size=800, random_state=random_seed
    )
    validation_df, basic_test_df = train_test_split(
        temp_df, test_size=200, random_state=random_seed
    )
    test_df = pd.concat([longer_df, basic_test_df], ignore_index=True)
    train_df["type"] = "train"
    validation_df["type"] = "validation"
    test_df["type"] = "test"
    # Concatenate all dataframes
    all_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)
    # Save the final dataset
    with open(data_file_path, "w") as f:
        all_df.to_json(f, orient="records", lines=True)
    
    config.log(f"Finished building finetuning dataset. Ending at {datetime.datetime.now()}", level=config.logging.INFO)