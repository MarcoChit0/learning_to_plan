# generate.py (Modified)

import datetime
from typing import Optional, List, Union # Added Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
import google.generativeai as genai

# Import project modules
import learning_to_plan.config as config
from learning_to_plan import task # Import task module


# --- Single Prompt Generation (Hugging Face) ---
def generate_single_hf(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_text: str,
) -> List[str]:
    """
    Generates one or more plan candidates for a single prompt using a Hugging Face model.

    Parameters:
        model: The loaded Hugging Face model (already on the correct device).
        tokenizer: The loaded Hugging Face tokenizer.
        prompt_text: The input prompt text (including the '## Plan.\n\n' marker).

    Returns:
        A list of generated plan texts.
    """
    config.log("Generating with Hugging Face model.", level=config.logging.DEBUG, do_print=False) # DEBUG level
    # model.eval() # Model should already be in eval mode from generate_batch

    # Model is assumed to be on the correct device already
    effective_device = next(model.parameters()).device
    device_type = effective_device.type

    # Determine dtype from model
    dtype = model.dtype if hasattr(model, 'dtype') else (torch.bfloat16 if config.get_config("bf16", False) else torch.float16)

    with torch.no_grad():
        # Use autocast only if on CUDA and model is not 4/8 bit
        # Quantized models handle types internally via bitsandbytes
        is_quantized = hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit
        use_autocast = (device_type == 'cuda') and not is_quantized

        with torch.autocast(
            device_type='cuda',
            dtype=dtype,
            enabled=use_autocast
        ):
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=False, # Do not pad here, generate handles it
                truncation=True,
                max_length=config.get_config("max_seq_length", 2048),
            ).to(effective_device) # Send inputs to the model's device

            outputs = model.generate(
                **inputs,
                max_new_tokens=config.get_config("max_new_tokens", 2048), 
                do_sample=config.get_config("do_sample", True),
                temperature=config.get_config("temperature", 0.7),
                top_p=config.get_config("top_p", 0.93), # Added top_p
                top_k=config.get_config("top_k", 50),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=config.get_config("num_return_sequences", 1), # On the paper 1, 3 and 5
            )

        # Decode all the sequences, removing the prompt part
        generated_texts = []
        input_length = inputs.input_ids.shape[1]
        for output_sequence in outputs:
            generated_tokens = output_sequence[input_length:] if output_sequence.shape[0] > input_length else torch.tensor([], dtype=torch.long, device=effective_device)
            decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(decoded_text.strip())

        return generated_texts

# --- Single Prompt Generation (Gemini) ---
def generate_single_gemini(
    prompt_text: str,
    model_name: str
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
    config.log(f"Generating with Gemini model: {model_name}", level=config.logging.DEBUG, do_print=False) # DEBUG level

    if not config.GOOGLE_API_KEY:
        config.log("GOOGLE_API_KEY is not set. Cannot use Gemini API.", level=config.logging.ERROR)
        return []

    try:
        generation_config = {
            "temperature": config.get_config("temperature", 0.7),
            "top_p": config.get_config("top_p", 1.0), # Ensure top_p is included
            "max_output_tokens": config.get_config("max_new_tokens", 2048), # Optional: map max_new_tokens
            "candidate_count": config.get_config("num_return_sequences", 1) # Map num_return_sequences
        }

        model = genai.GenerativeModel(
            model_name,
            generation_config=generation_config,
        )

        response = model.generate_content(prompt_text)

        generated_texts = []
        if response and response.candidates:
            for candidate in response.candidates:
                # Safely access text
                if candidate.content and candidate.content.parts:
                    text = ""
                    for part in candidate.content.parts:
                        if part.text:
                            text += part.text
                    generated_texts.append(text.strip())
                else:
                     config.log("Gemini candidate has no content parts.", level=config.logging.WARNING)

            config.log(f"Gemini generation successful. Metadata: {getattr(response, 'usage_metadata', 'N/A')}", level=config.logging.INFO, do_print=False)
            return generated_texts
        else:
             config.log(f"Gemini generation failed or returned empty response for prompt: {prompt_text[:100]}...", level=config.logging.WARNING)
             return []

    except Exception as e:
        config.log(f"Error during Gemini API call for model {model_name}: {e}", level=config.logging.ERROR, exc_info=True)
        return []


# --- Batch Generation from File (Modified) ---
def generate_batch(domain:str, number_of_problems_per_domain: Union[int, str] = "all"):
    """
    Generates plans for instances in a test file using a pre-loaded model (HF)
    or the Gemini API, saves results including original instance data and multiple
    generated plans.

    Parameters:
        domain: The domain name to process.
        number_of_problems_per_domain: Criteria for selecting instances ("all", "basic", "long", or int).
    """
    start_time = datetime.datetime.now()
    model_name = config.get_config("model_name")


    config.log(
        f"Starting generation batch with model '{model_name}' – time: {start_time}",
        level=config.logging.INFO
    )

    # --- Determine Model Type ---
    is_gemini_model = model_name and model_name.lower().startswith("gemini")

    if is_gemini_model:
        config.log(f"Using Gemini model: {model_name}. API should be pre-configured.", level=config.logging.INFO)
        if not config.GOOGLE_API_KEY:
             m = "GOOGLE_API_KEY not set. Cannot use Gemini API for generation."
             config.log(m, level=config.logging.ERROR)
             raise ValueError(m)
        # Ensure API is configured if not done globally (optional, depends on structure)
        try:
            genai.configure(api_key=config.GOOGLE_API_KEY)
        except Exception as e:
            config.log(f"Error re-configuring Gemini API (might be harmless if already configured): {e}", level=config.logging.WARNING)
    else:
        config.log(f"Using Hugging Face model: {model_name}. Model/tokenizer expected to be pre-loaded.", level=config.logging.INFO)
        # Load model and tokenizer
        model, tokenizer = config.load_model_and_tokenizer(checkpoint_dir=config.get_checkpoint_dir(domain, model_name))
        assert tokenizer is not None, "Tokenizer loading failed."
        assert model is not None, "Model loading failed."
        config.log("Model and tokenizer loaded successfully.", level=config.logging.INFO)

   # --- Load Dataset ---
    data_file_path = config.PROCESSED_DATA_FILE_PATH
    config.log(f"Loading tasks from {data_file_path}")
    try:
        # Get tasks from JSONL file
        tasks: set[task.Task] = task.get_tasks_from_jsonl(data_file_path)
        assert len(tasks) > 0, f"No tasks found in {data_file_path}."

        # Get test tasks
        test_tasks = {t for t in tasks if t._type == task.Task.TaskType.TEST}
        assert len(test_tasks) > 0, f"No test tasks found in {data_file_path}."

        # Filter test tasks by domain
        domain_specifict_test_tasks = {t for t in test_tasks if t._domain == domain}
        assert len(domain_specifict_test_tasks) > 0, f"No test tasks found in {data_file_path} for domain {domain}."
        config.log(f"Loaded {len(tasks)} from {data_file_path}, of which {len(test_tasks)} are test tasks and {len(domain_specifict_test_tasks)} are for domain {domain}.", level=config.logging.INFO)

        tasks_to_process: set[task.Task] = set()
        selection_criteria = number_of_problems_per_domain if number_of_problems_per_domain is not None else "all"
        if isinstance(selection_criteria, str) and selection_criteria.lower() == "all":
            tasks_to_process = domain_specifict_test_tasks
            config.log(f"Selecting all {len(tasks_to_process)} test tasks.")
        elif isinstance(selection_criteria, str):
            size = selection_criteria.lower()
            if size == "basic":
                tasks_to_process = {t for t in domain_specifict_test_tasks if not t._is_longer_plan}
                config.log(f"Selecting {len(tasks_to_process)} basic test tasks.")
            elif size == "long":
                tasks_to_process = {t for t in domain_specifict_test_tasks if t._is_longer_plan}
                config.log(f"Selecting {len(tasks_to_process)} long test tasks.")
            else:
                raise ValueError(f"Invalid string value for selection: '{selection_criteria}'. Expected 'all', 'basic', or 'long'.")
        elif isinstance(selection_criteria, int):
            if selection_criteria > 0:
                sorted_domain_specifict_test_tasks = sorted(list(test_tasks))
                tasks_to_process = set(sorted_domain_specifict_test_tasks[:min(selection_criteria, len(sorted_domain_specifict_test_tasks))])
                config.log(f"Selecting the first {len(tasks_to_process)} sorted test tasks (requested {selection_criteria}).")
            else:
                raise ValueError(f"Number of problems must be a positive integer, got: {selection_criteria}.")
        else:
            raise TypeError(f"Unsupported type for selection criteria: {type(selection_criteria)}. Expected int, str, or None.")

        config.log(f"Selected {len(tasks_to_process)} final instances for generation.")
        tasks = tasks - tasks_to_process

    except Exception as e:
        config.log(f"Error loading or selecting tasks from {data_file_path}: {e}", level=config.logging.ERROR, exc_info=True)
        raise e

    # --- Generate Plans ---
    config.log("Starting plan generation loop...")
    for t in tqdm(tasks_to_process, total=len(tasks_to_process), desc="Generating plans"):
        try:
            # config.log(f"Generating plans for task {t._id} with model {model_name}", level=config.logging.DEBUG) # DEBUG level
            prompt_text = t.get_prompt(eos_token=tokenizer.eos_token if tokenizer else None, with_plan=False)

            if is_gemini_model:
                generated_plans = generate_single_gemini(
                    prompt_text=prompt_text,
                    model_name=model_name
                )
            else: # Hugging Face model
                generated_plans = generate_single_hf(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text
                )

        except Exception as e:
            config.log(f"Error generating plan for task {t._id} with model {model_name}: {e}", level=config.logging.ERROR, exc_info=True)
            generated_plans = [f"Generation Error: {e}"]
        
        # Add generated plans (or error message) to the task object
        t.add_generated_plans(model_name, generated_plans, overwrite=True)
        tasks.add(t) 

    config.log(f"Plan generation loop completed for {len(tasks_to_process)} instances.")

    # --- Save Results ---
    try:
        config.log(f"Saving {len(tasks)} tasks back to {data_file_path}...")
        task.save_tasks_to_jsonl(tasks, data_file_path)
        config.log(f"Results saved to {data_file_path}.")
    except Exception as e:
        config.log(f"Error saving tasks to {data_file_path}: {e}", level=config.logging.ERROR, exc_info=True)
        raise e

    end_time = datetime.datetime.now()
    config.log(f"Generation batch finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}. Total time: {end_time - start_time}", level=config.logging.INFO)

    # --- Clean up GPU memory if HF model was used ---
    if not is_gemini_model and model is not None:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            config.log("Cleaned GPU memory after HF generation.", level=config.logging.INFO)

