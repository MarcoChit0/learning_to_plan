from learning_to_plan.task import Task, get_task_from_csv
import asyncio
import aiohttp
import os
import csv
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
    tasks: set[Task],
    output_file_path: str,
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
    if os.path.exists(output_file_path):
        if overwrite:
            config.logging.warning(f"Overwriting existing dataset at {output_file_path}.")
        else:
            config.logging.info(f"Loading existing dataset at {output_file_path}. Skipping recalculation for already processed tasks.")
            with open(output_file_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    task = get_task_from_csv(row)
                    if task in tasks and task._status == "ok":
                        config.logging.info(f"Skipping {task._instance}.")
                        processed_tasks.add(task)
                        tasks.remove(task)

    semaphore = asyncio.Semaphore(num_workers)
    await asyncio.gather(*[process_instance(task) for task in tasks])

    processed_tasks.update(tasks)
    config.create_necessary_dirs(output_file_path)
    with open(output_file_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["domain_file_path", "instance_file_path", "domain", "instance", "status", "plan", "error", "is_longer_plan"])
        writer.writeheader()

        for t in sorted(processed_tasks):
            writer.writerow(t.write_to_dict())
            config.logging.info(f"Processed {t}.")

    config.logging.info(f"Finished call to planning as a service at {datetime.datetime.now()}.")