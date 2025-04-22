import os
import datetime
import json
from typing import List, Dict, Any

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig # Added AutoConfig import
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
    load_dotenv() # Loads variables from .env if present (useful locally)
    # Ensure the environment variable is set, preferably via Kaggle Secrets
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        config.logging.warning("HUGGINGFACE_TOKEN environment variable not set.")
        # Decide how to handle missing token: raise error, proceed without, etc.
        # For now, we'll allow proceeding but log a warning.
        # raise ValueError("HUGGINGFACE_TOKEN must be set in the environment or Kaggle Secrets.")


    # --- Correction: Use hf_token variable ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
        if config.MODEL_TRAINING_CONFIG["bf16"]
        else torch.float16,
        token=hf_token, # --- Correction: Use hf_token variable ---
    )
    # --- End Correction ---


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
    # Corrected: Use config helper function for eval_batch_size
    loader = DataLoader(dataset, batch_size=config.model_params("eval_batch_size", 1)) # Use cfg helper

    for batch in tqdm(loader, total=len(loader), desc="evaluating", unit="batch"):
        prompts: List[str] = []
        ground_truths: List[str] = []
        raw_prompts: List[str] = []

        # Prepare inputs
        for prompt_data in batch["prompt"]: # Assuming batch["prompt"] contains the full JSON string or dict
             # If prompt_data is a string, parse it; if dict, use directly
            if isinstance(prompt_data, str):
                try:
                    loaded_data = json.loads(prompt_data)
                    full_prompt = loaded_data.get("prompt", "") # Extract the actual prompt string
                except json.JSONDecodeError:
                    config.logging.warning(f"Skipping malformed JSON string: {prompt_data[:100]}...")
                    continue # Skip this item if JSON is invalid
            elif isinstance(prompt_data, dict):
                 full_prompt = prompt_data.get("prompt", "")
            else:
                 config.logging.warning(f"Skipping unexpected data type in batch: {type(prompt_data)}")
                 continue


            if "## Plan." not in full_prompt:
                config.logging.warning(f"Skipping sample without '## Plan.' separator: {full_prompt[:100]}...")
                continue # Skip malformed samples

            prompt_part, _, plan_part = full_prompt.partition("## Plan.")
            prompts.append(prompt_part.strip() + "\n## Plan.\n\n")
            ground_truths.append(plan_part.strip())
            # Store the original structured data if needed, otherwise the full prompt string
            raw_prompts.append(prompt_data if isinstance(prompt_data, (str, dict)) else full_prompt)


        if not prompts:
            continue

        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16,
        ):
            # Corrected: Use config helper function
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.model_params("max_seq_length", 2048), # Use cfg helper
            ).to(device)

            outputs = model.generate(
                **inputs,
                # Corrected: Use config helper function
                max_new_tokens=config.model_params("max_new_tokens", 2048), # Use cfg helper
                do_sample=True, # Keep True if you want sampling
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id, # Usually set eos_token_id as pad_token_id for generation
            )

        # Decode & metric calculation
        for idx, out_seq in enumerate(outputs):
            # Decode the generated sequence, skipping special tokens
            # And only decode the generated part (after input_ids length)
            input_len = inputs['input_ids'].shape[1]
            generated_ids = out_seq[input_len:]
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)


            # --- Refined plan extraction ---
            # The decoded part already starts *after* "## Plan.\n\n" conceptually
            # We just need to clean it up.
            generated_text = decoded.strip()
            # --- End Refined plan extraction ---


            generated_plan = [
                a.strip() for a in generated_text.split('\n') if a.strip()
            ]
            ground_truth_plan = [
                a.strip() for a in ground_truths[idx].split('\n') if a.strip()
            ]

            # Longest Common Sub‑sequence length (dynamic programming)
            len_gen = len(generated_plan)
            len_gt = len(ground_truth_plan)
            C = np.zeros((len_gen + 1, len_gt + 1), dtype=int)

            for i in range(1, len_gen + 1):
                for j in range(1, len_gt + 1):
                    if generated_plan[i - 1] == ground_truth_plan[j - 1]:
                        C[i, j] = C[i - 1, j - 1] + 1
                    else:
                        C[i, j] = max(C[i - 1, j], C[i, j - 1])

            lcs_len = int(C[len_gen, len_gt]) # Correct indexing

            data.append(
                {
                    "instance": json.dumps(raw_prompts[idx]) if isinstance(raw_prompts[idx], dict) else raw_prompts[idx], # Store original data
                    "correct": 1 if lcs_len == len_gt and len_gen == len_gt else 0, # Stricter correctness: LCS must match GT length, and generated length must also match GT length
                    "lcs": lcs_len,
                    "lcs_ratio": lcs_len / len_gt if len_gt > 0 else 0.0, # Use len_gt
                    "generated_plan_length": len_gen, # Use len_gen
                    "ground_truth_plan_length": len_gt, # Use len_gt
                    # Optional: Add generated text for inspection
                    "generated_text": generated_text,
                }
            )

    # ── Persist detailed data ──────────────────────────────────────────
    metrics_file_path = os.path.join(model_dir, config.TEST_METRICS_FILE_NAME)
    try: # Add error handling for file reading
        if os.path.exists(metrics_file_path):
            df_metrics = pd.read_csv(metrics_file_path)
        else:
            df_metrics = pd.DataFrame(
                columns=[
                    "timestamp",
                    "trained_epochs", # This might be inaccurate if loading a pre-trained model not trained in this framework
                    "accuracy",
                    "mean_lcs",
                    "std_lcs",
                    "mean_lcs_ratio",
                    "std_lcs_ratio",
                    "mean_generated_plan_length",
                    "std_generated_plan_length",
                ]
            )
    except pd.errors.EmptyDataError:
         config.logging.warning(f"Metrics file {metrics_file_path} is empty. Creating a new DataFrame.")
         df_metrics = pd.DataFrame(columns=[...]) # Reinitialize as above
    except Exception as e:
        config.logging.error(f"Error reading metrics file {metrics_file_path}: {e}")
        # Decide recovery strategy, e.g., create new df or raise error
        raise e # Or reinitialize df


    # Ensure data_index is correctly calculated even if file didn't exist or was empty
    data_index = len(df_metrics) + 1 if 'df_metrics' in locals() and not df_metrics.empty else 1


    # Corrected: Use config helper function
    data_file_path = os.path.join(
        model_dir, config.TEST_DATA_FILE_NAME.format(index=data_index)
    )
    # Add error handling for file writing
    try:
        pd.DataFrame(data).to_csv(data_file_path, index=False)
        config.logging.info(f"Detailed evaluation results saved to {data_file_path}")
    except Exception as e:
        config.logging.error(f"Error saving detailed evaluation data to {data_file_path}: {e}")


    # ── Aggregate metrics ──────────────────────────────────────────────
    if not data: # Handle case where no data was generated (e.g., all samples skipped)
         config.logging.warning("No data generated during evaluation loop. Skipping metrics calculation.")
         return # Or handle as appropriate

    accuracy = 100.0 * sum(d["correct"] for d in data) / total_instances
    mean_lcs = np.mean([d["lcs"] for d in data])
    std_lcs = np.std([d["lcs"] for d in data])
    mean_lcs_ratio = np.mean([d["lcs_ratio"] for d in data])
    std_lcs_ratio = np.std([d["lcs_ratio"] for d in data])
    mean_gen_len = np.mean([d["generated_plan_length"] for d in data])
    std_gen_len = np.std([d["generated_plan_length"] for d in data])

    # --- Determine trained_epochs ---
    # Attempt to read from training arguments if available, otherwise use config
    # This requires the training_args used for the specific checkpoint, which isn't
    # directly available here. Using the config value might be misleading if the
    # checkpoint wasn't trained for that many epochs.
    # A better approach might be to store epoch info with the checkpoint.
    # For now, using the config value as a fallback.
    trained_epochs_value = config.model_params("num_train_epochs", "unknown") # Use cfg helper


    new_row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trained_epochs": trained_epochs_value,
        "accuracy": accuracy,
        "mean_lcs": mean_lcs,
        "std_lcs": std_lcs,
        "mean_lcs_ratio": mean_lcs_ratio,
        "std_lcs_ratio": std_lcs_ratio,
        "mean_generated_plan_length": mean_gen_len,
        "std_generated_plan_length": std_gen_len,
    }

    # Use concat instead of append (deprecated)
    df_metrics = pd.concat([df_metrics, pd.DataFrame([new_row])], ignore_index=True)

    # Add error handling for file writing
    try:
        df_metrics.to_csv(metrics_file_path, index=False)
        config.logging.info(f"Aggregated metrics appended to {metrics_file_path}")
    except Exception as e:
        config.logging.error(f"Error saving aggregated metrics to {metrics_file_path}: {e}")


    config.logging.info(
        "Evaluation finished – accuracy: %.2f%% – saved to %s",
        accuracy,
        metrics_file_path,
    )