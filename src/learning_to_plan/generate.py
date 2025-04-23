# generate.py

import os
import datetime
import json
import logging # Import standard logging for level constants
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the refactored config module
import learning_to_plan.config as config

# --- Single Prompt Generation ---
def generate_single(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_text: str,
    device: torch.device,
) -> str:
    """
    Generates a plan for a single prompt using the provided model and tokenizer.

    Parameters:
        model: The loaded Hugging Face model (already on device).
        tokenizer: The loaded Hugging Face tokenizer.
        prompt_text: The input prompt text (including the '## Plan.\n\n' marker).
        device: The device (CPU or CUDA) to run inference on.

    Returns:
        The generated plan text.
    """
    model.eval() # Ensure model is in eval mode

    # Get generation parameters from config
    # Use config.get_config instead of the non-existent config.eval_params
    max_len = config.get_config("max_seq_length", 2048)
    new_tokens = config.get_config("max_new_tokens", 2048)
    do_sampling = config.get_config("do_sample", True) # Default to True as before
    temperature = config.get_config("temperature", 0.7) # Default from previous snippet, adjust if needed

    # Determine dtype from model if possible, fallback to config
    dtype = model.dtype if hasattr(model, 'dtype') else (torch.bfloat16 if config.get_config("bf16", False) else torch.float16)

    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=dtype, # Use determined dtype
    ):
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_len, # Use variable from get_config
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=new_tokens,    # Use variable
            do_sample=do_sampling,        # Use variable
            temperature=temperature,      # Use variable
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id, # Use eos token for padding
        )

        # Decode only the newly generated part
        input_len = inputs['input_ids'].shape[1]
        # Ensure output is on CPU for decoding if needed, and handle potential batches (though batch=1 here)
        generated_ids = outputs[0, input_len:].cpu()
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_text = decoded.strip()

        return generated_text

# --- Batch Generation from File ---
def generate_batch(
    checkpoint_model_dir: str,
    test_file: str,
    output_jsonl_path: str,
    *,
    max_instances: Optional[int] = None
):
    """
    Loads model, generates plans for instances in a test file, saves results.

    Parameters:
        model_dir: Path to the HF checkpoint directory.
        test_file: Path to the input JSONL file.
        output_jsonl_path: Path to save the output JSONL file.
        max_instances: Max instances to process (None for all).
        batch_size: Processing batch size (currently only 1 supported effectively).
    """
    start_time = datetime.datetime.now()
    # Use config.log function
    config.log(f"Starting generation – checkpoint: {checkpoint_model_dir}, input: {test_file}, output: {output_jsonl_path} – time: {start_time}", level=logging.INFO)

    model, tokenizer = config.load_model_and_tokenizer(checkpoint_dir=checkpoint_model_dir)

    device = "auto" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval() 
    config.log(f"Model and tokenizer loaded successfully onto device: {device}", level=logging.INFO)

    batch_size = config.get_config("generate_batch_size", 1)

    # --- Load Dataset ---
    try:
        dataset = load_dataset("json", data_files={"test": test_file})["test"]
        if len(dataset) == 0:
            raise ValueError("Test dataset is empty.")
    except Exception as e:
         config.log(f"Fatal error loading dataset from {test_file}: {e}", level=logging.ERROR, exc_info=True)
         raise e

    if max_instances is not None and max_instances > 0:
        dataset = dataset.select(range(min(max_instances, len(dataset))))
    total_instances = len(dataset)
    config.log(f"Loaded {total_instances} instances for generation.")

    # --- Generation Loop ---
    results_to_save = []
    loader = DataLoader(dataset, batch_size=batch_size) # batch_size=1 default

    for batch_data in tqdm(loader, total=len(loader), desc="Generating plans", unit="batch"):
        items_to_process = batch_data["prompt"]
        for item_str in items_to_process:
            instance_data_json = None
            full_prompt = None
            ground_truth_plan = ""

            try:
                instance_data = json.loads(item_str)
                full_prompt = instance_data.get("prompt", "")
                instance_data_json = item_str

                if "## Plan." not in full_prompt:
                    config.log(f"Skipping sample without '## Plan.' separator: {full_prompt[:100]}...", level=logging.WARNING)
                    continue

                prompt_part, _, ground_truth_plan_raw = full_prompt.partition("## Plan.")
                generation_prompt = prompt_part.strip() + "\n## Plan.\n\n"
                ground_truth_plan = ground_truth_plan_raw.strip()

            except json.JSONDecodeError:
                 config.log(f"Skipping malformed JSON line: {item_str[:100]}...", level=logging.WARNING)
                 continue
            except Exception as e:
                config.log(f"Error processing instance data: {item_str[:100]}... Error: {e}", level=logging.ERROR, exc_info=True)
                continue

            # --- Call Single Generation ---
            try:
                generated_plan = generate_single(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=generation_prompt,
                    device=device
                )

                results_to_save.append({
                    "instance_data": instance_data_json,
                    "ground_truth_plan": ground_truth_plan,
                    "generated_plan": generated_plan,
                })

            except Exception as e:
                 config.log(f"Error during generation for prompt: {generation_prompt[:100]}... Error: {e}", level=logging.ERROR, exc_info=True)
                 results_to_save.append({
                    "instance_data": instance_data_json,
                    "ground_truth_plan": ground_truth_plan,
                    "generated_plan": f"GENERATION_ERROR: {e}",
                 })

    # --- Save Results ---
    try:
        # Use config helper to ensure directory exists
        config.create_necessary_dirs(output_jsonl_path)

        with open(output_jsonl_path, "w", encoding="utf-8") as f:
            for result in results_to_save:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        config.log(f"Saved {len(results_to_save)} generated results to {output_jsonl_path}")

    except Exception as e:
        config.log(f"Error saving results to {output_jsonl_path}: {e}", level=logging.ERROR, exc_info=True)

    end_time = datetime.datetime.now()
    config.log(
        f"Generation finished – Total time: {str(end_time - start_time)}",
        level=logging.INFO
    )