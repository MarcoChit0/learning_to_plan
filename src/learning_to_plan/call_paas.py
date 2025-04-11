from learning_to_plan.task import Task, get_task_from_csv
import asyncio
import aiohttp
import os
import csv
from learning_to_plan.config import create_necessary_dirs

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
                            print(f"Instance {instance_name} -- Attempt {attempt} -- Success!")
                            return result_data
                    break
            except Exception as e:
                print(f"Instance {instance_name} -- Attempt {attempt} -- Error during planning request: {e}")
            print(f"Instance {instance_name} -- Attempt {attempt} -- Retrying...")
            await asyncio.sleep(2)
        print(f"Instance {instance_name} -- Attempt {attempt} -- Exceeded max retries.")
        return {"status": "error", "error": "Max retries exceeded or no valid plan returned."}

async def call_paas(
    tasks:dict[str, Task],
    output_file_path:str,
    max_retries = 3,
    num_workers = 4,
    overwrite = False
):
    async def process_instance(task):
        async with semaphore:
            try:
                response = await get_plan_from_paas(
                    domain_content= task.read_domain(),
                    instance_content= task.read_instance(),
                    instance_name=task._instance,
                    max_retries=max_retries
                )
                task.update_status(response)

            except Exception as e:
                raise e

    if os.path.exists(output_file_path):
        if overwrite:
            print(f"Overwriting existing dataset at {output_file_path}.")
        else:
            print(f"Loading existing dataset at {output_file_path}. Skipping recalculation for already processed tasks.")
            with open(output_file_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    task = get_task_from_csv(row)
                    if task._instance in tasks.keys():
                        tasks[task._instance] = task
                        if task._status == "ok": print(f"Task {task._instance} already processed.")

    tasks_to_process = [task for task in tasks.values() if task._status != "ok"]
    semaphore = asyncio.Semaphore(num_workers)
    await asyncio.gather(*[process_instance(task) for task in tasks_to_process])

    create_necessary_dirs(output_file_path)
    with open(output_file_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["domain_file_path","instance_file_path", "domain", "instance", "status", "plan", "error"])
        writer.writeheader()

        for t in sorted(tasks.values()):
            writer.writerow(t.write_to_dict())
            print(f"Processed {t._instance}: {t._status}.")

    print(f"Dataset written to {output_file_path}.")