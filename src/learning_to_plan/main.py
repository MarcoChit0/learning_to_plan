import os
import asyncio
import aiohttp
import time
import re
import csv
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def convert_instance_to_natural_language(pddl_text: str) -> str:
    obj_match = re.search(r"\(:objects\s+(.*?)\)", pddl_text, re.DOTALL)
    objects = obj_match.group(1).split() if obj_match else []
    objects_str = "blocks: " + ", ".join(objects) + "."
    init_match = re.search(r"\(:init\s+(.*?)\)", pddl_text, re.DOTALL)
    init_lines = init_match.group(1).split("\n") if init_match else []
    init_facts = []
    for line in init_lines:
        line = line.strip()
        if not line: continue
        fact = line.strip("()")
        tokens = fact.split()
        if tokens[0] == "handempty":
            init_facts.append("your hand is empty.")
        elif tokens[0] == "holding":
            init_facts.append(f"you are holding {tokens[1]}.")
        elif tokens[0] == "clear":
            init_facts.append(f"{tokens[1]} is clear.")
        elif tokens[0] == "ontable":
            init_facts.append(f"{tokens[1]} is on the table.")
        elif tokens[0] == "on":
            init_facts.append(f"{tokens[1]} is on {tokens[2]}.")
        else:
            init_facts.append(fact + ".")
    init_text = "initial state:\n" + "\n".join(init_facts)
    goal_match = re.search(r"\(:goal\s+\(and\s+(.*?)\)\s*\)", pddl_text, re.DOTALL)
    goal_lines = goal_match.group(1).split("\n") if goal_match else []
    goal_facts = []
    for line in goal_lines:
        line = line.strip()
        if not line: continue
        fact = line.strip("()")
        tokens = fact.split()
        if tokens[0] == "handempty":
            goal_facts.append("your hand is empty.")
        elif tokens[0] == "holding":
            goal_facts.append(f"you are holding {tokens[1]}.")
        elif tokens[0] == "clear":
            goal_facts.append(f"{tokens[1]} is clear.")
        elif tokens[0] == "ontable":
            goal_facts.append(f"{tokens[1]} is on the table.")
        elif tokens[0] == "on":
            goal_facts.append(f"{tokens[1]} is on {tokens[2]}.")
        else:
            goal_facts.append(fact + ".")
    goal_text = "goal state:\n" + "\n".join(goal_facts)
    return f"{objects_str}\n\n{init_text}\n\n{goal_text}"

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

def map_action_plan_to_plain_text(plan: str) -> str:
    actions = plan.split(";")[0].strip().split("\n")
    nl_plan = ""
    for action in actions:
        action = action.strip()
        if action.startswith("(") and action.endswith(")"):
            action = action[1:-1].strip()
            parts = action.split()
            if parts[0] == "unstack":
                nl_plan += f"unstack {parts[1]} from {parts[2]};\n"
            elif parts[0] == "pick-up":
                nl_plan += f"pick up {parts[1]};\n"
            elif parts[0] == "stack":
                nl_plan += f"stack {parts[1]} on {parts[2]};\n"
            elif parts[0] == "put-down":
                nl_plan += f"put down {parts[1]};\n"
            else:
                raise ValueError(f"Unknown action: {action}")
    return nl_plan

import asyncio
import aiofiles

async def build_blocksworld_dataset(
    output_file_path = "blocksworld_dataset.csv",
    domain_file_path = "./data/raw/instances/blocksworld/generated_domain.pddl",
    instances_dir_path = "./data/raw/instances/blocksworld/generated_basic/",
    max_instances = 50,
    max_retries = 3,
    num_workers = 4,
    overwrite = False
):
    async def process_instance(instance, domain_content):
        async with semaphore:
            try:
                async with aiofiles.open(os.path.join(instances_dir_path, instance), "r") as pf:
                    instance_content = await pf.read()
                description = convert_instance_to_natural_language(instance_content)
                response = await get_plan_from_paas(
                    domain_content=domain_content,
                    instance_content=instance_content,
                    instance_name=instance,
                    max_retries=max_retries
                )
                status = response.get("status", "error")
                if status == "ok":
                    plan_text = response["result"]["output"]["sas_plan"]
                    plain_plan = map_action_plan_to_plain_text(plan_text) if plan_text else ""
                    err_msg = ""
                else:
                    plain_plan = ""
                    err_msg = response.get("error", "Missing plan details or planning failed.")

                data[instance] = {
                    "instance": instance,
                    "description": description,
                    "status": status,
                    "plan": plain_plan,
                    "error": err_msg
                }
            except Exception as e:
                print(f"Error processing {instance}: {e}")
                data[instance] = {
                    "instance": instance,
                    "description": "",
                    "status": "error",
                    "plan": "",
                    "error": str(e)
                }


    data = {}
    if os.path.exists(output_file_path):
        if overwrite:
            print(f"Overwriting existing dataset at {output_file_path}.")
        else:
            print(f"Loading existing dataset at {output_file_path}. Will skip recalculating those instances.")
            with open(output_file_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data[row["instance"]] = row

    pattern = re.compile(r"instance-(\d+)\.pddl$")
    instances = sorted([f for f in os.listdir(instances_dir_path) if re.match(pattern, f)], key=lambda x: int(pattern.search(x).group(1)))[:max_instances]

    # remove instances that are not to be computed
    for instance in list(data.keys()):
        if instance not in instances:
            print(f"Removing {instance} from dataset.")
            del data[instance]
    
    # remove instances that were already computed with status 'ok'
    if not overwrite:
        for instance in list(data.keys()):
            if instance in instances and data[instance].get("status") == "ok":
                print(f"Removing {instance} from instances to be computed.")
                instances.remove(instance)

    semaphore = asyncio.Semaphore(num_workers)

    async with aiofiles.open(domain_file_path, "r") as df:
        domain_content = await df.read()

    tasks = [process_instance(instance, domain_content) for instance in instances]

    await asyncio.gather(*tasks)

    with open(output_file_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["instance", "description", "status", "plan", "error"])
        writer.writeheader()

        for key in sorted(data.keys(), key=lambda x: int(pattern.search(x).group(1))):
            d = data[key]
            writer.writerow(d)
            print(f"Processed {key}: {d['status']}.")

    print(f"Dataset written to {output_file_path}.")


def create_finetuning_dataset(
    csv_path,
    train_output="train.jsonl",
    test_output="test.jsonl",
    test_size=0.2,
    random_seed=42,
    overwrite=False
):
    if os.path.exists(train_output) and os.path.exists(test_output) and not overwrite:
        print(f"Finetuning dataset files already exist: {train_output}, {test_output}")
        return

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df_valid = df[(df["status"] == "ok") & (df["plan"].notna()) & (df["plan"].str.strip() != "")]

    if len(df_valid) == 0:
        print("No valid rows found in dataset.")
        return

    train_df, test_df = train_test_split(
        df_valid, test_size=test_size, random_state=random_seed
    )

    def write_dataset(df, output_path):
        def format_row(row):
            data = {
                "prompt": (
                    "# Goal.\n\n"
                    "Use the available actions to transform the initial state into the goal state.\n\n"
                    "# Output Format.\n\n"
                    "Return a sequence of actions, one per line, in the order they should be applied.\n\n"
                    "# Warnings.\n\n"
                    "An action can only be applied if all its preconditions are true in the current state.\n"
                    "When an action is applied, its effects update the current state by adding and removing facts.\n"
                    "The goal is reached when all facts in the goal state are present in the current state.\n"
                    "A valid plan must transform the initial state into the goal state using only applicable actions.\n"
                    "Starting from the initial state, choose an applicable action, apply it, and repeat this process until the goal is reached.\n"
                    "If no sequence of actions can reach the goal, return nothing.\n\n"
                    "# Context.\n\n"
                    "## Available actions.\n\n"
                    "### Action: pick up block.\n"
                    "preconditions:\n"
                    "block is on the table.\n"
                    "block is clear.\n"
                    "your hand is empty.\n\n"
                    "effects:\n"
                    "you are holding block.\n"
                    "your hand is not empty.\n"
                    "block is not on the table.\n"
                    "block is not clear.\n\n"
                    "### Action: put down block.\n"
                    "preconditions:\n"
                    "you are holding block.\n\n"
                    "effects:\n"
                    "block is on the table.\n"
                    "block is clear.\n"
                    "your hand is empty.\n"
                    "you are not holding block.\n\n"
                    "### Action: stack block1 on block2.\n"
                    "preconditions:\n"
                    "you are holding block1.\n"
                    "block2 is clear.\n\n"
                    "effects:\n"
                    "your hand is empty.\n"
                    "block1 is clear.\n"
                    "block2 is not clear.\n"
                    "you are not holding block1.\n"
                    "block1 is on block2.\n\n"
                    "### Action: unstack block1 from block2.\n"
                    "preconditions:\n"
                    "block1 is clear.\n"
                    "block1 is on block2.\n"
                    "your hand is empty.\n\n"
                    "effects:\n"
                    "you are holding block1.\n"
                    "your hand is not empty.\n"
                    "block2 is clear.\n"
                    "block1 is not clear.\n"
                    "block1 is not on block2.\n\n"
                    "## Instance.\n\n"
                    + row['description'] + "\n\n"
                    "## Plan.\n\n"
                    + row['plan']
                )
            }
            return json.dumps(data, ensure_ascii=False)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(format_row(row) + "\n")
    
    write_dataset(train_df, train_output)
    write_dataset(test_df, test_output)


def run_training_procedure(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_length=2048,
    batch_size=1,
    grad_accum_steps=1,
    learning_rate=2e-5,
    epochs=20,
    train_file="train.jsonl",
    validation_file="test.jsonl",
    output_dir="./qwen-finetuned"
):
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import load_dataset
    import torch

    # Load dataset from JSONL files
    dataset = load_dataset("json", data_files={"train": train_file, "validation": validation_file})

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)

    # Tokenize function
    def tokenize_fn(example):
        return tokenizer(
            example["prompt"],
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["prompt"])

    # Data collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        save_steps=500,
        logging_steps=20,
        fp16=False,
        save_total_limit=2,
        report_to="none"
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator
    )

    trainer.train()


if __name__ == "__main__":
    import sys
    import asyncio

    if len(sys.argv) != 2:
        print("Usage: python main.py [dataset-raw|dataset-processed|train]")
    else:
        mode = sys.argv[1]
        if mode == "dataset-raw":
            asyncio.run(build_blocksworld_dataset(output_file_path="blocksworld_dataset.csv", max_instances=4200, overwrite=False, max_retries=3))
        elif mode == "dataset-processed":
            create_finetuning_dataset(csv_path="blocksworld_dataset.csv")
        elif mode == "train":
            run_training_procedure()
        else:
            print("Invalid argument. Options: dataset-raw, dataset-processed, train")