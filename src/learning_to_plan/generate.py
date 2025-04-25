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
import google.generativeai as genai # Import Gemini API library

# Import the refactored config module
import learning_to_plan.config as config


# --- Single Prompt Generation (Hugging Face) ---
def generate_single_hf(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_text: str,
) -> List[str]: # Return type changed to List[str]
    """
    Generates one or more plan candidates for a single prompt using a Hugging Face model.

    Parameters:
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
        prompt_text: The input prompt text (including the '## Plan.\n\n' marker).

    Returns:
        A list of generated plan texts.
    """
    config.log("Generating with Hugging Face model.", level=logging.INFO, do_print=False)
    model.eval() # Ensure model is in eval mode

    # Determine the device the model is actually on
    # If CUDA is available, the model loading should have placed it there.
    # Otherwise, it will be on CPU.
    effective_device = next(model.parameters()).device
    device_type = effective_device.type # Correctly get device type

    # Determine dtype from model if possible, fallback to config
    dtype = model.dtype if hasattr(model, 'dtype') else (torch.bfloat16 if config.get_config("bf16", False) else torch.float16)

    with torch.no_grad():
        # Use autocast only if on CUDA
        with torch.autocast(
            device_type='cuda', # Specify 'cuda' explicitly
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
            # Ensure slicing is within bounds
            generated_tokens = output_sequence[input_length:] if output_sequence.shape[0] > input_length else torch.tensor([], dtype=torch.long, device=effective_device)
            # Decode the generated tokens
            decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(decoded_text.strip()) # Add stripped text

        return generated_texts

# --- Single Prompt Generation (Gemini) ---
def generate_single_gemini(
    prompt_text: str,
    model_name: str # Pass model name to potentially use different Gemini models
) -> List[str]:
    """
    Generates one or more plan candidates for a single prompt using the Gemini API.

    Parameters:
        prompt_text: The input prompt text.
        model_name: The specific Gemini model name to use (e.g., "gemini-pro").

    Returns:
        A list containing a single generated plan text (Gemini typically returns one response).
        Returns an empty list if generation fails.
    """
    config.log(f"Generating with Gemini model: {model_name}", level=logging.INFO, do_print=False)

    if not google_api_key:
        config.log("GOOGLE_API_KEY is not set. Cannot use Gemini API.", level=logging.ERROR)
        return []

    try:
        # Use configuration from config.py, with defaults if not set
        generation_config = {
            "temperature": config.get_config("temperature", 0.0),
            "top_p": config.get_config("top_p", 0.1),
            # Gemini typically returns text/plain by default, but can specify
            # "response_mime_type": "text/plain",
        }

        # Use the specified model name
        model = genai.GenerativeModel(model_name, generation_config=generation_config)

        # Gemini's generate_content handles batching and sampling internally
        # We expect one response per prompt for this use case
        response = model.generate_content(prompt_text)

        # Extract the text from the response
        # Check if the response has parts and text
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
             # Assuming the first part contains the text
             generated_text = response.candidates[0].content.parts[0].text
             config.log(f"Gemini generation successful. Metadata: {response.usage_metadata}", level=logging.INFO, do_print=False)
             return [generated_text.strip()] # Return as a list for consistency
        else:
             config.log(f"Gemini generation failed or returned empty response for prompt: {prompt_text[:100]}...", level=logging.WARNING)
             # Log potential safety ratings or finish reasons
             if response and response.candidates:
                 for i, candidate in enumerate(response.candidates):
                     config.log(f"Candidate {i} finish reason: {candidate.finish_reason}", level=logging.WARNING, do_print=False)
                     config.log(f"Candidate {i} safety ratings: {candidate.safety_ratings}", level=logging.WARNING, do_print=False)
             return [] # Return empty list on failure

    except Exception as e:
        config.log(f"Error during Gemini API call for model {model_name}: {e}", level=logging.ERROR, exc_info=True)
        return [] # Return empty list on error


# --- Batch Generation from File (Modified) ---
from typing import Union, Optional
from learning_to_plan import task
# TODO: VERIFY WHETHER THIS IS BEING SAVED - This comment seems to be a leftover, saving is handled below.
def generate_batch(
    data_file_path: str,
    number_of_problems_per_domain: Union[int, str] = None,
    checkpoint_model_dir: Optional[str] = None
):
    """
    Loads model (HF) or configures API (Gemini), generates plans for instances
    in a test file, saves results including original instance data and multiple
    generated plans.

    Parameters:
        data_file_path: Path to the input JSONL file (expected to have a 'prompt' key).
        number_of_problems_per_domain: Criteria for selecting instances ("all", "basic", "long", or int).
        checkpoint_model_dir: Path to the HF checkpoint directory or None for Gemini models.
    """
    start_time = datetime.datetime.now()
    model_name = config.get_config("model_name") # Get the configured model name

    config.log(
        f"Starting generation with model '{model_name}' – data: {data_file_path} – time: {start_time}",
        level=logging.INFO
    )

    # --- Load Model or Configure API based on model_name ---
    is_gemini_model = model_name and model_name.lower().startswith("gemini")
    hf_model, hf_tokenizer = None, None


    if is_gemini_model:
        config.log(f"Using Gemini model: {model_name}. No local model loading required.", level=logging.INFO)
        try:
            if config.GOOGLE_API_KEY:
                genai.configure(api_key=config.GOOGLE_API_KEY)
                config.log("Gemini API configured successfully.", level=logging.INFO, do_print=False)
            else:
                m = "GOOGLE_API_KEY environment variable not set. Gemini API not configured."
                config.log(m, level=logging.error, do_print=False)
                raise ValueError(m)
        except Exception as e:
            config.log(f"Error configuring Gemini API: {e}", level=logging.ERROR, exc_info=True, do_print=False)
    else:
        config.log(f"Using Hugging Face model: {model_name}. Loading model and tokenizer.", level=logging.INFO)
        try:
            assert checkpoint_model_dir, "Checkpoint model directory must be provided for Hugging Face models."
            hf_model, hf_tokenizer = config.load_model_and_tokenizer(checkpoint_dir=checkpoint_model_dir)
            hf_model.eval() 
        except Exception as e:
            config.log(f"Fatal error loading Hugging Face model/tokenizer from {checkpoint_model_dir}: {e}", level=logging.ERROR, exc_info=True)
            raise e

    # --- Load Dataset ---
    config.log(f"Loading tasks from {data_file_path}")
    try:
        tasks:set[task.Task] = task.get_tasks_from_jsonl(data_file_path)
        test_tasks = [t for t in tasks if t._type == task.Task.TaskType.TEST]
        config.log(f"Found {len(test_tasks)} test tasks.")
        tasks_to_process : set[task.Task] = set()

        if number_of_problems_per_domain is None or (isinstance(number_of_problems_per_domain, str) and number_of_problems_per_domain.lower() == "all"):
            tasks_to_process = set(test_tasks) # Convert list to set
            config.log(f"Selecting all {len(tasks_to_process)} test tasks.")
        elif isinstance(number_of_problems_per_domain, str):
            mode = number_of_problems_per_domain.lower()
            if mode == "basic":
                tasks_to_process = {t for t in test_tasks if not t._is_longer_plan}
                config.log(f"Selecting {len(tasks_to_process)} basic test tasks.")
            elif mode == "long":
                tasks_to_process = {t for t in test_tasks if t._is_longer_plan}
                config.log(f"Selecting {len(tasks_to_process)} long test tasks.")
            else:
                raise ValueError(f"Invalid string value for selection: '{number_of_problems_per_domain}'. Expected 'all', 'basic', or 'long'.")
        elif isinstance(number_of_problems_per_domain, int):
            if number_of_problems_per_domain > 0:
                # Sort test_tasks before slicing to get a consistent subset
                sorted_test_tasks = sorted(test_tasks)
                tasks_to_process = set(sorted_test_tasks[:min(number_of_problems_per_domain, len(sorted_test_tasks))])
                config.log(f"Selecting the first {len(tasks_to_process)} sorted test tasks (requested {number_of_problems_per_domain}).")
            else:
                raise ValueError(f"Number of problems must be a positive integer, got: {number_of_problems_per_domain}.")
        else:
            raise TypeError(f"Unsupported type for selection criteria: {type(number_of_problems_per_domain)}. Expected int, str, or None.")

        config.log(f"Selected {len(tasks_to_process)} final instances for generation.")
    except Exception as e: # Catch any error during loading or selection
        config.log(f"Error loading or selecting tasks from {data_file_path}: {e}", level=logging.ERROR, exc_info=True)
        raise e

    tasks = tasks - tasks_to_process


    # --- Generate Plans ---
    config.log("Starting plan generation...")
    for t in tqdm(tasks_to_process, total=len(tasks_to_process), desc="Generating plans"):
        try:
            config.log(f"Generating plans for task {t._id} with model {model_name}")
            prompt_text = t.add_separator(t.build_prompt())

            if is_gemini_model:
                generated_plans = generate_single_gemini(
                    prompt_text=prompt_text,
                    model_name=model_name # Pass the specific Gemini model name
                )
            else: # Assume Hugging Face model
                assert hf_model, "Hugging Face model is not loaded. Check the model loading process."
                assert hf_tokenizer, "Hugging Face tokenizer is not loaded. Check the model loading process."
                # Generate plans using the Hugging Face model
                generated_plans = generate_single_hf(
                    model=hf_model,
                    tokenizer=hf_tokenizer,
                    prompt_text=prompt_text
                )

            # Add generated plans to the task object under the model's name
            t.add_generated_plans(model_name, generated_plans)
            config.log(f"Generated {len(generated_plans)} plans for task {t._id} using {model_name}.")
            tasks.add(t) # Add the task back to the set of tasks

        except Exception as e:
            config.log(f"Error generating plan for task {t._id} with model {model_name}: {e}", level=logging.ERROR, exc_info=True)
            # Add an error message as a generated plan
            t.add_generated_plans(model_name, [f"Error: {e}"])
            continue # Continue to the next task even if one fails
    config.log(f"Plan generation completed for {len(tasks_to_process)} instances.")

    # --- Save Results ---
    try:
        config.log(f"Saving generated plans to {data_file_path}...")
        # Save all tasks back, including those not processed but loaded initially
        task.save_tasks_to_jsonl(tasks, data_file_path)
        config.log(f"Results saved to {data_file_path}.")
    except Exception as e:
        config.log(f"Error saving tasks to {data_file_path}: {e}", level=logging.ERROR, exc_info=True)
        raise e

    end_time = datetime.datetime.now()
    config.log(f"Generation finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}. Total time: {end_time - start_time}", level=logging.INFO)