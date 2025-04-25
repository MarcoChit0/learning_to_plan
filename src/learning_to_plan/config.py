# config.py

import os
import json
import logging
import argparse
from typing import Optional, Dict, Any

# --- Global Store for Configuration ---
_CONFIG_STORE: Dict[str, Any] = {} # Holds model/training/generation parameters

# --- Global Variables for Paths and Token (Set during initialization) ---
DATA_DIR = "data/"
RAW_DIR: Optional[str] = None
PROCESSED_DATA_DIR: Optional[str] = None
CHECKPOINTS_DIR: Optional[str] = None
HUGGINGFACE_TOKEN: Optional[str] = None
GOOGLE_API_KEY: Optional[str] = None # Added Google API Key variable
LOGGING_INITIALIZED: bool = False # Flag to prevent duplicate logging setup
# --- Configure root logger minimally initially ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s]: %(message)s',
    handlers=[logging.StreamHandler()] # Log to console initially
)
logger = logging.getLogger(__name__) # Module-level logger


# --- Constants for directory/file names ---
CONFIGS_DIR_NAME = "configs"
CONFIGS_DIR = os.path.join("src", CONFIGS_DIR_NAME)
DEFAULT_TRAIN_CONFIG = "train_config.json"
DEFAULT_GENERATE_CONFIG = "generate_config.json"
BASIC_INSTANCES = "generated_basic"
LONG_INSTANCES = "generated_basic_longer_plan_len"
PROCESSED_DATA_FILE_NAME = "data.jsonl"
DOMAIN_FILE_NAME = "generated_domain.pddl"
LOGGING_FILE_NAME = "logs.log"
# --- End Constants ---

# --- New Print and Log Function ---
def log(
    message: str,
    level: int = logging.INFO,
    do_print: bool = True,
    exc_info = False # Add exc_info for logging exceptions
    ) -> None:
    """
    Logs a message using the configured logger and optionally prints it.

    Args:
        message: The message string to log and potentially print.
        level: The logging level (e.g., logging.INFO, logging.WARNING).
        do_print: If True, prints the message to the console.
        exc_info: If True, includes exception information in the log.
    """
    # Use the logger instance defined at the module level
    logger.log(level, message, exc_info=exc_info)
    if do_print:
        # Basic print, might not handle multi-line/formatting exactly like logger
        print(message)


def initialize(
    args: Optional[argparse.Namespace] = None,
    config_path: Optional[str] = None, # User can specify a config file path
) -> None:
    """
    Initializes configuration by loading from JSON (explicit path or default based on context),
    applying argument overrides, and setting up paths/logging.

    Priority order for config values:
    1. Command-line arguments (`args`) overrides everything else.
    2. Explicit file path (`config_path`) provided by user.
    3. Default config file (train_config.json or generate_config.json based on args.train/args.evaluate)
       - Only loaded if config_path is NOT provided AND context is Train/Evaluate.

    Args:
        args: Parsed arguments from argparse. Used for overrides and context detection.
        config_path: Path to a specific JSON configuration file to load (optional).
    """
    global _CONFIG_STORE, HUGGINGFACE_TOKEN, GOOGLE_API_KEY # Added GOOGLE_API_KEY
    global DATA_DIR, RAW_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR
    global LOGGING_INITIALIZED, logger # Use the global logger

    # Initial message uses the basic setup
    log("Basic console logging initialized.", level=logging.INFO)

    if args is None:
        e = "Config initialization called without 'args'. Cannot determine context or apply overrides."
        log(e, level=logging.ERROR)
        raise ValueError(e) # Raise error as it's critical

    # --- 1. Load Base Model/Train/Generate Configuration ---
    loaded_config = {}
    config_source = "None (Context not Train/Evaluate or no path/default found)"
    load_attempted = False

    if config_path:
        log(f"Attempting to load configuration from specified path: {config_path}")
        load_attempted = True
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            config_source = f"Explicit File: {config_path}"
        except FileNotFoundError:
            msg = f"Specified configuration file not found: {config_path}. Proceeding without model config."
            log(msg, level=logging.ERROR)
            config_source = f"Error (File not found: {config_path})"
            # Decide if this should be fatal? Maybe raise FileNotFoundError(msg)
        except json.JSONDecodeError as e:
            msg = f"Error decoding JSON from {config_path}: {e}. Proceeding without model config."
            log(msg, level=logging.ERROR)
            config_source = f"Error (Invalid JSON: {config_path})"
            # Decide if this should be fatal? Maybe raise json.JSONDecodeError(...)
        except Exception as e:
            msg = f"Unexpected error loading {config_path}: {e}. Proceeding without model config."
            log(msg, level=logging.ERROR, exc_info=True)
            config_source = f"Error (Load error: {config_path})"
            # Decide if this should be fatal? Maybe raise e
    else:
        # No explicit path, check context from args to load default
        default_config_to_load = None
        context = "Unknown"
        if hasattr(args, 'train') and args.train:
            default_config_to_load = DEFAULT_TRAIN_CONFIG
            context = "Training"
        # Check for evaluate or a potential generate flag
        elif hasattr(args, 'generate') and args.generate: # Assuming args.generate exists now
             default_config_to_load = DEFAULT_GENERATE_CONFIG
             context = "Generation"
        # Add back evaluate if needed, map it to generate config?
        # elif hasattr(args, 'evaluate') and args.evaluate:
        #     default_config_to_load = DEFAULT_GENERATE_CONFIG
        #     context = "Evaluation"


        if default_config_to_load:
            log(f"No explicit config path. Attempting to load default '{context}' config: {default_config_to_load}")
            load_attempted = True
            default_path = os.path.join(CONFIGS_DIR, default_config_to_load)
            if os.path.exists(default_path):
                try:
                    with open(default_path, 'r', encoding='utf-8') as f:
                        loaded_config = json.load(f)
                    config_source = f"Default {context} File: {default_path}"
                except Exception as e:
                    msg = f"Error loading default {context} config {default_path}: {e}."
                    log(msg, level=logging.ERROR, exc_info=True)
                    config_source = f"Error (Load error: {default_path})"
                    # Decide if this should be fatal? Maybe raise e
            else:
                error_message = f"Default {context} config file '{default_path}' not found."
                log(error_message, level=logging.ERROR)
                # Only raise FileNotFoundError if it's a train or generate context and default config is missing
                if context in ["Training", "Generation"]:
                    raise FileNotFoundError(error_message) # Make missing default fatal
                else:
                    log(f"Skipping fatal error for missing default config in '{context}' context.", level=logging.WARNING)



    # --- 2. Apply Overrides from Args ---
    if load_attempted or loaded_config:
        log("Applying command-line argument overrides to configuration...")
        # Define which command-line arguments can override config values
        # Added gemini specific overrides
        override_keys = {"model_name", "num_train_epochs"} # Use a set for efficient lookups

        for arg_name, arg_value in vars(args).items():
            if arg_name in override_keys and arg_value is not None:
                original_value = loaded_config.get(arg_name)
                # Only log if the value is different or being newly set
                if arg_name not in loaded_config or original_value != arg_value:
                    log(f"  Overriding '{arg_name}': {original_value if arg_name in loaded_config else '<Not Set>'} -> {arg_value}")
                    loaded_config[arg_name] = arg_value

        # Handle 4bit/8bit/bf16 overrides
        if hasattr(args, 'load_in_4bit') and args.load_in_4bit:
            if loaded_config.get("load_in_8bit", False):
                log("  Overriding 'load_in_8bit': True -> False (due to --load_in_4bit)")
                loaded_config["load_in_8bit"] = False
            # Ensure load_in_4bit is set if arg is true, even if not in JSON
            if loaded_config.get("bf16", False):
                log("  Overriding 'bf16': True -> False (due to --load_in_4bit)")
                loaded_config["bf16"] = False
            loaded_config["load_in_4bit"] = True


        if hasattr(args, 'load_in_8bit') and args.load_in_8bit:
            if loaded_config.get("bf16", False):
                log("  Overriding 'bf16': True -> False (due to --load_in_8bit)")
                loaded_config["bf16"] = False
            # Ensure load_in_8bit is set if arg is true, even if not in JSON
            loaded_config["load_in_8bit"] = True


    # --- 3. Store Final Model/Train/Generate Configuration ---
    _CONFIG_STORE = loaded_config
    log(f"Model configuration source: {config_source}")
    log(f"Final Configuration: {json.dumps(_CONFIG_STORE, indent=2)}", level=logging.DEBUG, do_print=False)


    # --- 4. Setup Paths, Logging (File Handler), and Tokens (Always Run) ---
    _setup_paths_and_logging(args)


def _setup_paths_and_logging(args: Optional[argparse.Namespace]):
    """Helper function to set up paths and file logging."""
    global DATA_DIR, RAW_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR
    global HUGGINGFACE_TOKEN, GOOGLE_API_KEY, LOGGING_INITIALIZED, logger

    # --- Handle Hugging Face Token ---
    temp_hf_token = None
    if args is not None and hasattr(args, 'huggingface_token') and args.huggingface_token:
        temp_hf_token = args.huggingface_token
    else:
        import dotenv
        dotenv.load_dotenv()
        temp_hf_token = os.getenv("HUGGINGFACE_TOKEN")

    HUGGINGFACE_TOKEN = temp_hf_token
    if not HUGGINGFACE_TOKEN:
        # Log warning, but don't make fatal unless a HF model is explicitly requested later
        log("Hugging Face token not provided. Set it via --huggingface_token or HUGGINGFACE_TOKEN environment variable. HF model loading may fail.", level=logging.WARNING)

    # --- Handle Google API Key ---
    temp_google_api_key = None
    if args is not None and hasattr(args, 'google_api_key') and args.google_api_key:
        temp_google_api_key = args.google_api_key
    else:
        import dotenv
        dotenv.load_dotenv()
        temp_google_api_key = os.getenv("GOOGLE_API_KEY")

    GOOGLE_API_KEY = temp_google_api_key
    if not GOOGLE_API_KEY:
        # Log warning, but don't make fatal unless a Gemini model is explicitly requested later
        log("Google API key not provided. Set it via --google_api_key or GOOGLE_API_KEY environment variable. Gemini model generation may fail.", level=logging.WARNING)


    # --- Initialize Directories ---

    if args is not None and hasattr(args, 'data_dir_path') and args.data_dir_path:
        DATA_DIR = args.data_dir_path
    log(f"Using DATA_DIR: {DATA_DIR}")
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")

    for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR]:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as e:
            log(f"Failed to create directory {dir_path}: {e}", level=logging.ERROR, exc_info=True)
            # Decide if this is fatal? For now, allow continuing.
    log("Data directories ensured/created.")

    # --- Initialize File Logging (Add Handler Once) ---
    root_logger = logging.getLogger()
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)

    # Check if file logging should be setup
    should_setup_file_logging = not has_file_handler and DATA_DIR

    if should_setup_file_logging:
        log_file_path = os.path.join(DATA_DIR, LOGGING_FILE_NAME)
        try:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            # Use a more detailed formatter for the file
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s [%(name)s:%(lineno)d] %(message)s')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            if root_logger.level > logging.INFO: # Ensure root level is appropriate
                 root_logger.setLevel(logging.INFO)
            log(f"File logging initialized. Log file: {log_file_path}")
        except Exception as e:
            log(f"Failed to configure file logging to {log_file_path}: {e}. Continuing with console logging only.", level=logging.ERROR, exc_info=True)
    elif has_file_handler:
        # Use debug level, don't print to console
        log("File logging handler already exists.", level=logging.DEBUG, do_print=False)
    elif not DATA_DIR:
        log("DATA_DIR not set. File logging skipped.", level=logging.WARNING)


def get_config(key: str, default: Any = None) -> Any:
    """
    Retrieves a model/train/generate configuration value.

    Args:
        key: The configuration key to retrieve.
        default: The value to return if the key is not found.

    Returns:
        The configuration value or the default.
    """
    if not _CONFIG_STORE:
        # Use debug level, don't print
        log(f"Model config store accessed for key '{key}' but is empty. Returning default.", level=logging.DEBUG, do_print=False)
    return _CONFIG_STORE.get(key, default)

def create_necessary_dirs(file_path: str) -> None:
    """Creates parent directories for a given file path if they don't exist."""
    try:
        dirs = os.path.dirname(file_path)
        if dirs:
            os.makedirs(dirs, exist_ok=True)
    except OSError as e:
        log(f"Error creating directories for {file_path}: {e}", level=logging.ERROR, exc_info=True)
    except Exception as e:
        log(f"Unexpected error in create_necessary_dirs for {file_path}: {e}", level=logging.ERROR, exc_info=True)


import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from typing import Tuple, Optional

def load_model_and_tokenizer(checkpoint_dir: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads a Hugging Face model and tokenizer, handling checkpoints,
    quantization (8-bit), and data types (bf16/fp16).

    Args:
        checkpoint_dir: The directory where checkpoints are saved/looked for.

    Returns:
        A tuple containing the loaded model and tokenizer.

    Raises:
        ValueError: If loading fails or parameters are incompatible.
    """

    last_checkpoint = get_last_checkpoint(checkpoint_dir)
    # Use the model name from config as the base if no checkpoint is found
    model_source = last_checkpoint if last_checkpoint else get_config("model_name", None)
    assert model_source, "Model source is None. Check your configuration."

    # Check if the requested model is a Gemini model. If so, we don't load a HF model here.
    if model_source and model_source.lower().startswith("gemini"):
        log(f"Requested model '{model_source}' is a Gemini model. Skipping Hugging Face model/tokenizer loading.", level=logging.INFO)
        # Return None for model and tokenizer as they are not needed for API calls
        return None, None


    if get_config("load_in_4bit", False):
        log("Applying 4-bit quantization.", level=logging.INFO, do_print=False)
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif get_config("load_in_8bit", False):
        log("Applying 8-bit quantization.", level=logging.INFO, do_print=False)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        log("Loading model without quantization.", level=logging.INFO, do_print=False)
        quantization_config = None


    # --- Load Tokenizer ---
    assert HUGGINGFACE_TOKEN, "Hugging Face token is not set. Cannot load tokenizer."
    try:
        log(f"Loading tokenizer from: {model_source}", level=logging.INFO)
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=get_config("trust_remote_code", True),
            token=HUGGINGFACE_TOKEN,
        )
        # Set pad token if missing (common practice)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            log("Tokenizer pad token set to EOS token.", level=logging.INFO, do_print=False)
        log(f"Tokenizer loaded successfully from {model_source}.", level=logging.INFO)
    except Exception as e:
        m = f"Failed to load tokenizer from {model_source}: {e}"
        log(m, level=logging.ERROR, exc_info=True)
        raise ValueError(m) from e


    # --- Load Model ---
    try:
        log(f"Loading model from: {model_source}", level=logging.INFO)
        if quantization_config:
            log(f"Loading model with quantization {quantization_config}.", level=logging.INFO, do_print=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                trust_remote_code=get_config("trust_remote_code", True),
                token=HUGGINGFACE_TOKEN,
                quantization_config=quantization_config,
            )
            from peft import LoraConfig, get_peft_model
            r = get_config("lora_r", 8)
            alpha = get_config("lora_alpha", r * 4)
            lora_cfg = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "up_proj", "down_proj", "gate_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
            log("LoRA model loaded successfully.", level=logging.INFO)
        else:
            log("Loading model without quantization.", level=logging.INFO, do_print=False)
            if get_config("bf16", False):
                log("Using bfloat16.", level=logging.INFO, do_print=False)
                torch_dtype = torch.bfloat16
            else:
                log("Using float16.", level=logging.INFO, do_print=False)
                torch_dtype = torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                trust_remote_code=get_config("trust_remote_code", True),
                torch_dtype=torch_dtype,
                token=HUGGINGFACE_TOKEN,
            )
            log(f"Model loaded successfully from {model_source}.", level=logging.INFO)
    except Exception as e:
        m = f"Failed to load model from {model_source}: {e}"
        log(m, level=logging.ERROR, exc_info=True)
        raise ValueError(m) from e


    return model, tokenizer