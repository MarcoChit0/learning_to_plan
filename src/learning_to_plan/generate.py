# generate.py (Modified)

import os
import datetime
import json
import logging # Import standard logging for level constants
from typing import Optional, List, Dict, Any # Added List, Dict, Any

import torch
# Removed direct AutoTokenizer/AutoModelForCausalLM imports as they are now in config.py
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

# Import the refactored config module
import learning_to_plan.config as config

# --- Single Prompt Generation (Modified) ---
def generate_single(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_text: str,
) -> List[str]: # Return type changed to List[str]
    """
    Generates one or more plan candidates for a single prompt.

    Parameters:
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
        prompt_text: The input prompt text (including the '## Plan.\n\n' marker).

    Returns:
        A list of generated plan texts.
    """
    model.eval() # Ensure model is in eval mode

    # Determine the device the model is actually on
    # If CUDA is available, the model loading should have placed it there.
    # Otherwise, it will be on CPU.
    effective_device = next(model.parameters()).device
    device_type = next(model.parameters()).device.type

    # Determine dtype from model if possible, fallback to config
    dtype = model.dtype if hasattr(model, 'dtype') else (torch.bfloat16 if config.get_config("bf16", False) else torch.float16)

    with torch.no_grad(), torch.autocast(
        device_type=device_type,
        dtype=dtype,
        enabled=(device_type == 'cuda') # Only enable autocast on CUDA
    ):
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=config.get_config("max_seq_length", 2048),
        ).to(effective_device) # Send inputs to the model's device

        outputs = model.generate(
            **inputs,
            max_new_tokens=config.get_config("max_new_tokens", 2048),
            do_sample=config.get_config("do_sample", True),
            temperature=config.get_config("temperature", 0.7),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id, # Use EOS token for padding
            num_return_sequences=config.get_config("num_return_sequences", 1),
        )

        # Decode all the sequences, removing the prompt part
        generated_texts = []
        input_length = inputs.input_ids.shape[1]
        for output_sequence in outputs:
            # Slice the output sequence to get only the generated tokens
            generated_tokens = output_sequence[input_length:]
            # Decode the generated tokens
            decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(decoded_text.strip()) # Add stripped text

        return generated_texts

# --- Batch Generation from File (Modified) ---
from typing import Union 
from learning_to_plan import task

def generate_batch(
    checkpoint_model_dir: str, # Renamed from model_dir for clarity
    data_file_path: str,
    number_of_problems_per_domain: Union[int, str] = None,
):
    """
    Loads model, generates plans for instances in a test file, saves results
    including original instance data and multiple generated plans.

    Parameters:
        checkpoint_model_dir: Path to the HF checkpoint directory for the specific model/domain.
        test_file: Path to the input JSONL file (expected to have a 'prompt' key).
        output_jsonl_path: Path to save the output JSONL file.
        max_instances: Max instances to process (None for all).
    """
    start_time = datetime.datetime.now()
    config.log(
        f"Starting generation – checkpoint: {checkpoint_model_dir}, data: {data_file_path} – time: {start_time}",
        level=logging.INFO
    )

    try:
        model, tokenizer = config.load_model_and_tokenizer(checkpoint_dir=checkpoint_model_dir)
    except Exception as e:
        config.log(f"Fatal error loading model/tokenizer from {checkpoint_model_dir}: {e}", level=logging.ERROR, exc_info=True)
        raise e # Stop execution

    model.eval()

    # --- Load Dataset ---
    config.log(f"Loading tasks from {data_file_path}")
    try:
        all_tasks:set[task.Task] = task.get_tasks_from_jsonl(data_file_path)
        test_tasks = [t for t in all_tasks if t._type == task.Task.TaskType.TEST]
        config.log(f"Found {len(test_tasks)} test tasks.")
        instances = [] 

        if number_of_problems_per_domain is None or (isinstance(number_of_problems_per_domain, str) and number_of_problems_per_domain.lower() == "all"):
            instances = sorted(test_tasks)
            config.log(f"Selecting all {len(instances)} test tasks.")
        elif isinstance(number_of_problems_per_domain, str):
            mode = number_of_problems_per_domain.lower()
            if mode == "basic":
                instances = sorted([t for t in test_tasks if not t._is_longer_plan])
                config.log(f"Selecting {len(instances)} basic test tasks.")
            elif mode == "long":
                instances = sorted([t for t in test_tasks if t._is_longer_plan])
                config.log(f"Selecting {len(instances)} long test tasks.")
            else:
                raise ValueError(f"Invalid string value for selection: '{number_of_problems_per_domain}'. Expected 'all', 'basic', or 'long'.")
        elif isinstance(number_of_problems_per_domain, int):
            if number_of_problems_per_domain > 0:
                sorted_test_tasks = sorted(test_tasks)
                num_to_take = min(number_of_problems_per_domain, len(sorted_test_tasks))
                instances = sorted_test_tasks[:num_to_take]
                config.log(f"Selecting the first {len(instances)} sorted test tasks (requested {number_of_problems_per_domain}).")
            else:
                raise ValueError(f"Number of problems must be a positive integer, got: {number_of_problems_per_domain}.")
        else:
            raise TypeError(f"Unsupported type for selection criteria: {type(number_of_problems_per_domain)}. Expected int, str, or None.")
            config.log(f"Selected {len(instances)} final instances for generation.")
    except Exception as e: # Catch any error during loading or selection
        config.log(f"Error loading or selecting tasks from {data_file_path}: {e}", level=logging.ERROR, exc_info=True)
        raise e
    # -- Generate Plans ---
    config.log("Starting plan generation...")    
    for current_task in tqdm(instances, total=len(instances), desc="Generating plans"):
        try:
            # Check if the task already has generated plans (for potential updates, though typically we generate fresh)
            is_update = hasattr(current_task, 'generated_plans') and current_task.generated_plans
            if is_update:
                config.log(f"Task {current_task.task_id} already has plans, will overwrite.") # Log if overwriting

            config.log(f"Generating plans for task {current_task}")
            # Generate plans using the task's prompt
            generated_plans = generate_single(
                model=model,
                tokenizer=tokenizer,
                prompt_text=current_task.add_separator(current_task.build_prompt())
            )
            print(generated_plans)
            # Store the generated plans directly in the Task object
            current_task.generated_plans = generated_plans
            config.log(f"Generated {len(generated_plans)} plans for task {current_task}.")
            all_tasks.remove(current_task)
            all_tasks.add(current_task) # Update the set with the modified task
        except Exception as e:
            config.log(f"Error generating plan for task {current_task}: {e}", level=logging.ERROR, exc_info=True)
            current_task.generated_plans = ["Error: " + str(e)]
            continue 
    config.log(f"Plan generation completed. {len(instances)} instances ready for saving.")
    # --- Save Results ---
    try:
        config.log(f"Saving generated plans to {data_file_path}...")
        task.save_tasks_to_jsonl(all_tasks, data_file_path)
        config.log(f"Results saved to {data_file_path}.")
    except Exception as e:
        config.log(f"Error saving tasks to {data_file_path}: {e}", level=logging.ERROR, exc_info=True)
        raise e
