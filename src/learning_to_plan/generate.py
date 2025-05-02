# generate.py (Modified)

import datetime
from typing import List, Union # Added Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
import google.generativeai as genai

# Import project modules
import learning_to_plan.config as config
from learning_to_plan import task # Import task module
import numpy as np

logger = config.get_logger(__name__)

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
    logger.debug("Generating with Hugging Face model.") # Use logger
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
    logger.debug(f"Generating with Gemini model: {model_name}") # Use logger

    if not config.GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY is not set. Cannot use Gemini API.") # Use logger
        return []

    try:
        generation_config = {
            "temperature": config.get_config("temperature", 0.7),
            "top_p": config.get_config("top_p", 0.93), # Ensure top_p is included
            "top_k": config.get_config("top_k", 50), # Optional: top-k sampling
            "max_output_tokens": config.get_config("max_new_tokens", 2048), # Optional: max output tokens
            "candidate_count": config.get_config("candidate_count", 1), # Optional: number of candidates
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
                     logger.warning("Gemini candidate has no content parts.") # Use logger

            logger.info(f"Gemini generation successful. Metadata: {getattr(response, 'usage_metadata', 'N/A')}") # Use logger
            return generated_texts
        else:
             logger.warning(f"Gemini generation failed or returned empty response for prompt: {prompt_text[:100]}...") # Use logger
             return []

    except Exception as e:
        logger.error(f"Error during Gemini API call for model {model_name}: {e}", exc_info=True) # Use logger
        return []


# --- Batch Generation from File (Modified) ---
def generate_batch(domain:str, number_of_problems_per_domain: Union[int, str] = "all", number_of_cot_examples:int = 0, random_seed:int = 42):
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
    rng = np.random.RandomState(random_seed)


    logger.info(
        f"Starting generation batch with model '{model_name}' – time: {start_time}" # Use logger
    )

    # --- Determine Model Type ---
    is_gemini_model = model_name and model_name.lower().startswith("gemini")
    model, tokenizer = None, None
    if is_gemini_model:
        logger.info(f"Using Gemini model: {model_name}. API should be pre-configured.") # Use logger
        if not config.GOOGLE_API_KEY:
             m = "GOOGLE_API_KEY not set. Cannot use Gemini API for generation."
             logger.error(m) # Use logger
             raise ValueError(m)
        # Ensure API is configured if not done globally (optional, depends on structure)
        try:
            genai.configure(api_key=config.GOOGLE_API_KEY)
        except Exception as e:
            logger.warning(f"Error re-configuring Gemini API (might be harmless if already configured): {e}") # Use logger
    else:
        logger.info(f"Using Hugging Face model: {model_name}. Model/tokenizer expected to be pre-loaded.") # Use logger
        # Load model and tokenizer
        model, tokenizer = config.load_model_and_tokenizer(checkpoint_dir=config.get_checkpoint_dir(domain, model_name))
        assert tokenizer is not None, "Tokenizer loading failed."
        assert model is not None, "Model loading failed."
        logger.info("Model and tokenizer loaded successfully.") # Use logger

   # --- Load Dataset ---
    data_file_path = config.PROCESSED_DATA_FILE_PATH
    logger.info(f"Loading tasks from {data_file_path}") # Use logger
    try:
        # Get tasks from JSONL file
        tasks: set[task.Task] = task.get_tasks_from_jsonl(data_file_path)
        assert len(tasks) > 0, f"No tasks found in {data_file_path}."

        # Filter tasks by domain
        domain_specific_tasks = {t for t in tasks if t._domain == domain}
        assert len(domain_specific_tasks) > 0, f"No tasks found in {data_file_path} for domain {domain}."

        # Filter by test type
        domain_specifict_test_tasks = {t for t in domain_specific_tasks if t._type == task.Task.TaskType.TEST}
        assert len(domain_specifict_test_tasks) > 0, f"No test tasks found in {data_file_path} for domain {domain}."
        logger.info(f"Loaded {len(tasks)} tasks. Found {len(domain_specific_tasks)} tasks for domain '{domain}', including {len(domain_specifict_test_tasks)} test tasks.") # Use logger

        tasks_to_process: set[task.Task] = set()
        selection_criteria = number_of_problems_per_domain if number_of_problems_per_domain is not None else "all"
        if isinstance(selection_criteria, str) and selection_criteria.lower() == "all":
            tasks_to_process = domain_specifict_test_tasks
            logger.info(f"Selecting all {len(tasks_to_process)} test tasks.") # Use logger
        elif isinstance(selection_criteria, str):
            size = selection_criteria.lower()
            if size == "basic":
                tasks_to_process = {t for t in domain_specifict_test_tasks if not t._is_longer_plan}
                logger.info(f"Selecting {len(tasks_to_process)} basic test tasks.") # Use logger
            elif size == "long":
                tasks_to_process = {t for t in domain_specifict_test_tasks if t._is_longer_plan}
                logger.info(f"Selecting {len(tasks_to_process)} long test tasks.") # Use logger
            else:
                raise ValueError(f"Invalid string value for selection: '{selection_criteria}'. Expected 'all', 'basic', or 'long'.")
        elif isinstance(selection_criteria, int):
            if selection_criteria > 0:
                sorted_domain_specifict_test_tasks = sorted(list(domain_specifict_test_tasks))
                tasks_to_process = set(sorted_domain_specifict_test_tasks[:min(selection_criteria, len(sorted_domain_specifict_test_tasks))])
                logger.info(f"Selecting the first {len(tasks_to_process)} sorted test tasks (requested {selection_criteria}).") # Use logger
            else:
                raise ValueError(f"Number of problems must be a positive integer, got: {selection_criteria}.")
        else:
            raise TypeError(f"Unsupported type for selection criteria: {type(selection_criteria)}. Expected int, str, or None.")

        logger.info(f"Selected {len(tasks_to_process)} final instances for generation.") # Use logger
        tasks = tasks - tasks_to_process
        possible_cot_examples = domain_specific_tasks - tasks_to_process
        cot_examples:set[task.Task] = set()
    except Exception as e:
        logger.error(f"Error loading or selecting tasks from {data_file_path}: {e}", exc_info=True) # Use logger
        raise e

    # --- Generate Plans ---
    logger.info("Starting plan generation loop...") # Use logger
    for t in tqdm(tasks_to_process, total=len(tasks_to_process), desc="Generating plans"):
        try:
            if number_of_cot_examples > 0:
                cot_examples = set(
                    rng.choice(
                        list(possible_cot_examples),
                        size=min(number_of_cot_examples, len(possible_cot_examples)),
                        replace=False
                    )
                )

            prompt_text = t.get_prompt(eos_token=tokenizer.eos_token if tokenizer else None, with_plan=False, cot_examples=cot_examples)

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
            logger.error(f"Error generating plan for task {t._id} with model {model_name}: {e}", exc_info=True) # Use logger
            generated_plans = [f"Generation Error: {e}"]

        # Add generated plans (or error message) to the task object
        prompt_type = task.PlanManager.PromptType.COT if len(cot_examples) > 0 else task.PlanManager.PromptType.IO
        # TODO: for now, add only a single prompt, later a list of prompts
        t.add_plan(model_name, prompt_type, generated_plans[0])
        tasks.add(t)

    logger.info(f"Plan generation loop completed for {len(tasks_to_process)} instances.") # Use logger

    # --- Save Results ---
    try:
        logger.info(f"Saving {len(tasks)} tasks back to {data_file_path}...") # Use logger
        task.save_tasks_to_jsonl(tasks, data_file_path)
        logger.info(f"Results saved to {data_file_path}.") # Use logger
    except Exception as e:
        logger.error(f"Error saving tasks to {data_file_path}: {e}", exc_info=True) # Use logger
        raise e

    end_time = datetime.datetime.now()
    logger.info(f"Generation batch finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}. Total time: {end_time - start_time}") # Use logger

    # --- Clean up GPU memory if HF model was used ---
    if not is_gemini_model and model is not None:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleaned GPU memory after HF generation.") # Use logger

