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
def generate_batch(
    checkpoint_model_dir: str, # Renamed from model_dir for clarity
    data_file_path: str,
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
    try:
        config.log(f"Loading test data from {data_file_path}")
        dataset = load_dataset("json", data_files=data_file_path)
        
        # Filter into train and validation sets based on the "type" field
        test_dataset = dataset["test"].filter(lambda example: example["type"] == "train")
        
        instances = list(test_dataset)    
        # Log the size
        config.log(f"Loaded test dataset with {len(instances)} instances")
    except Exception as e:
        config.log(f"Error loading test data: {e}", level=logging.ERROR, exc_info=True)
        raise e

    # --- Generate Plans ---
    config.log("Starting plan generation...")    
    for index, instance in tqdm(enumerate(instances), total=len(instances), desc="Generating plans"):
        try:
            # Check if we're updating existing plans
            is_update = 'generated_plans' in instance and instance['generated_plans']
            if is_update:
                config.log(f"Updating existing plans for instance {index}")
                
            # Generate plans
            generated_plans = generate_single(
                model=model,
                tokenizer=tokenizer,
                prompt_text=instance['prompt']
            )
            
            # Update the instance in-place
            # This ensures changes are directly reflected in the testing data
            instance['generated_plans'] = generated_plans
        except Exception as e:
            config.log(f"Error generating plan for instance {index}: {e}", level=logging.ERROR, exc_info=True)
            instance['generated_plans'] = ["Error: " + str(e)]
            continue
    
    config.log(f"Plan generation completed. {len(instances)} instances ready for saving.")
    # --- Save Results ---
    try:
        config.log(f"Saving results to {data_file_path}")
        with open(data_file_path, "w") as json_file:
            for instance in instances:
                json_file.write(json.dumps(instance) + "\n")
        config.log(f"Results saved successfully to {data_file_path}")
    except Exception as e:
        config.log(f"Error saving results: {e}", level=logging.ERROR, exc_info=True)
        raise e