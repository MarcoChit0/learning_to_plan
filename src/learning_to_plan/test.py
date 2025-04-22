import os
import datetime
import json
from typing import List, Dict, Any

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv

import learning_to_plan.config as config


def run_evaluation_procedure(
    model_dir: str,
    test_file: str,
    *,
    max_instances: int | None = None
):
    """
    Evaluate a model checkpoint on a JSONL test set.

    Parameters
    ----------
    model_dir : str
        Path to the HF checkpoint directory.
    test_file : str
        JSONL file whose "prompt" field contains
        "<task description> ## Plan. <ground‑truth>".
    max_instances : int | None, default 200
        Evaluate at most this many test tasks; None = all.
    """
    config.logging.info(
        "Starting evaluation – checkpoint: %s – time: %s",
        model_dir,
        datetime.datetime.now(),
    )

    # ── Load model & tokenizer ──────────────────────────────────────────
    load_dotenv()

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, token=config.HUGGINGFACE_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
        if config.MODEL_TRAINING_CONFIG["bf16"]
        else torch.float16,
        token=config.HUGGINGFACE_TOKEN,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # ── Load dataset ────────────────────────────────────────────────────
    dataset = load_dataset("json", data_files={"test": test_file})["test"]
    if len(dataset) == 0:
        raise ValueError("Test dataset is empty.")

    if max_instances:
        dataset = dataset.select(range(min(max_instances, len(dataset))))
    total_instances = len(dataset)

    # ── Evaluation loop ────────────────────────────────────────────────
    data: List[Dict[str, Any]] = []
    loader = DataLoader(dataset, batch_size=config.MODEL_TRAINING_CONFIG["eval_batch_size"])

    for batch in tqdm(loader, total=len(loader), desc="evaluating", unit="batch"):
        prompts: List[str] = []
        ground_truths: List[str] = []
        raw_prompts: List[str] = []

        # Prepare inputs
        for full_prompt in batch["prompt"]:
            if "## Plan." not in full_prompt:
                # Skip malformed samples (keeps behaviour of old script)
                continue
            prompt_part, _, plan_part = full_prompt.partition("## Plan.")
            prompts.append(prompt_part.strip() + "\n## Plan.\n\n")
            ground_truths.append(plan_part.strip())
            raw_prompts.append(full_prompt)

        if not prompts:
            continue

        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16,
        ):
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.MODEL_TRAINING_CONFIG["max_seq_length"],
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=config.MODEL_TRAINING_CONFIG["max_new_tokens"],
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode & metric calculation
        for idx, out_seq in enumerate(outputs):
            decoded = tokenizer.decode(out_seq, skip_special_tokens=True)
            # Keep only what comes after the marker
            if "## Plan." in decoded:
                generated_text = decoded.split("## Plan.", 1)[1].strip()
            else:
                generated_text = decoded

            generated_plan = [
                a.strip() for a in generated_text.split("\n") if a.strip()
            ]
            ground_truth_plan = [
                a.strip() for a in ground_truths[idx].split("\n") if a.strip()
            ]

            # Longest Common Sub‑sequence length (dynamic programming)
            C = np.zeros(
                (len(generated_plan) + 1, len(ground_truth_plan) + 1), dtype=int
            )
            for i in range(1, len(generated_plan) + 1):
                for j in range(1, len(ground_truth_plan) + 1):
                    if generated_plan[i - 1] == ground_truth_plan[j - 1]:
                        C[i, j] = C[i - 1, j - 1] + 1
                    else:
                        C[i, j] = max(C[i - 1, j], C[i, j - 1])

            lcs_len = int(C[-1, -1])
            data.append(
                {
                    "instance": raw_prompts[idx],
                    "correct": 1 if lcs_len == len(ground_truth_plan) else 0,
                    "lcs": lcs_len,
                    "lcs_ratio": lcs_len / len(ground_truth_plan)
                    if ground_truth_plan
                    else 0.0,
                    "generated_plan_length": len(generated_plan),
                    "ground_truth_plan_length": len(ground_truth_plan),
                }
            )

    # ── Persist detailed data ──────────────────────────────────────────
    metrics_file_path = os.path.join(model_dir, config.TEST_METRICS_FILE_NAME)
    if os.path.exists(metrics_file_path):
        df_metrics = pd.read_csv(metrics_file_path)
    else:
        df_metrics = pd.DataFrame(
            columns=[
                "timestamp",
                "trained_epochs",
                "accuracy",
                "mean_lcs",
                "std_lcs",
                "mean_lcs_ratio",
                "std_lcs_ratio",
                "mean_generated_plan_length",
                "std_generated_plan_length",
            ]
        )

    data_index = len(df_metrics) + 1
    data_file_path = os.path.join(
        model_dir, config.TEST_DATA_FILE_NAME.format(index=data_index)
    )
    pd.DataFrame(data).to_csv(data_file_path, index=False)

    # ── Aggregate metrics ──────────────────────────────────────────────
    accuracy = 100.0 * sum(d["correct"] for d in data) / total_instances
    mean_lcs = np.mean([d["lcs"] for d in data])
    std_lcs = np.std([d["lcs"] for d in data])
    mean_lcs_ratio = np.mean([d["lcs_ratio"] for d in data])
    std_lcs_ratio = np.std([d["lcs_ratio"] for d in data])
    mean_gen_len = np.mean([d["generated_plan_length"] for d in data])
    std_gen_len = np.std([d["generated_plan_length"] for d in data])

    new_row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trained_epochs": config.MODEL_TRAINING_CONFIG["num_train_epochs"],
        "accuracy": accuracy,
        "mean_lcs": mean_lcs,
        "std_lcs": std_lcs,
        "mean_lcs_ratio": mean_lcs_ratio,
        "std_lcs_ratio": std_lcs_ratio,
        "mean_generated_plan_length": mean_gen_len,
        "std_generated_plan_length": std_gen_len,
    }

    df_metrics = pd.concat([df_metrics, pd.DataFrame([new_row])], ignore_index=True)
    df_metrics.to_csv(metrics_file_path, index=False)

    config.logging.info(
        "Evaluation finished – accuracy: %.2f%% – saved to %s",
        accuracy,
        metrics_file_path,
    )
