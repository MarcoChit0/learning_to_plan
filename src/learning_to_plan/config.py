# config.py

import os
import json
import logging
import argparse
import torch # Added torch import
from typing import Optional, Dict, Any, Tuple # Added Tuple
import dotenv

# Import necessary HF classes
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
# Import PEFT components
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Global Store for Configuration ---
_CONFIG_STORE: Dict[str, Any] = {} # Holds model/training/generation parameters

# --- Global Variables for Paths and Token (Set during initialization) ---
DATA_DIR = "data/"
RAW_DIR: Optional[str] = None
CHECKPOINTS_DIR: Optional[str] = None
HUGGINGFACE_TOKEN: Optional[str] = None
GOOGLE_API_KEY: Optional[str] = None
PROCESSED_DATA_FILE_PATH: Optional[str] = None
LOGGING_INITIALIZED: bool = False
# --- Configure root logger minimally initially ---

def get_logger(name: str = __name__) -> logging.Logger:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s]: %(message)s',
        handlers=[logging.StreamHandler()] # Log to console initially
    )
    logger = logging.getLogger(name=name) # Module-level logger
    return logger

logger = get_logger() # Initialize logger


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


def initialize(
    args: argparse.Namespace,
) -> None:
    """
    Initializes configuration by loading from JSON (explicit path or default based on context),
    applying argument overrides, and setting up paths/logging.

    Priority order for config values:
    1. Command-line arguments (`args`) overrides everything else.
    2. Explicit file path (`config_path`) provided by user.
    3. Default config file (train_config.json or generate_config.json based on args.train/args.generate)
       - Only loaded if config_path is NOT provided AND context is Train/Generate.

    Args:
        args: Parsed arguments from argparse. Used for overrides and context detection.
        config_path: Path to a specific JSON configuration file to load (optional).
    """
    global _CONFIG_STORE, HUGGINGFACE_TOKEN, GOOGLE_API_KEY
    global DATA_DIR, RAW_DIR, CHECKPOINTS_DIR, PROCESSED_DATA_FILE_PATH, PROCESSED_DATA_FILE_NAME
    global LOGGING_INITIALIZED, logger

    logger.info("Basic console logging initialized.")

    if args is None:
        e = "Config initialization called without 'args'. Cannot determine context or apply overrides."
        logger.error(e)
        raise ValueError(e) # Raise error as it's critical

    # --- 1. Determine and Load Base Configuration ---
    loaded_config = {}
    context = "Unknown"
    config_file_path = None

    if hasattr(args, 'train') and args.train:
        config_file_path = os.path.join(CONFIGS_DIR, DEFAULT_TRAIN_CONFIG)
        context = "Training"
    elif hasattr(args, 'generate') and args.generate:
        config_file_path = os.path.join(CONFIGS_DIR, DEFAULT_GENERATE_CONFIG)
        context = "Generation"

    if context in ["Training", "Generation"]:
        if hasattr(args, 'config_path') and args.config_path:
            config_file_path = args.config_path
            logger.info(f"Explicit config path provided.")
        else:
            logger.info("No explicit config path provided and context is not Train or Generate. No config file will be loaded by default.")
        logger.info(f"Using config file: {config_file_path}")

        if not os.path.exists(config_file_path):
            logger.error(f"Config file '{config_file_path}' does not exist. Cannot load configuration.")
            raise FileNotFoundError(f"Config file '{config_file_path}' not found.")
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            logger.info(f"{context} Configuration File: {config_file_path}")
            logger.info(f"Successfully loaded configuration from {config_file_path}.")
        except Exception as e:
            m = f"Error loading config file {config_file_path}: {e}"
            logger.error(m, exc_info=True)
            raise ValueError(m) from e

    # --- 2. Apply Overrides from Args ---
    if loaded_config:
        logger.info("Applying command-line argument overrides to configuration...")
        override_keys = {"model_name", "num_train_epochs"}
        for arg_name, arg_value in vars(args).items():
            if arg_name in override_keys and arg_value is not None:
                original_value = loaded_config.get(arg_name)
                if arg_name not in loaded_config or original_value != arg_value:
                    logger.info(f"Overriding '{arg_name}': {original_value if arg_name in loaded_config else '<Not Set>'} -> {arg_value}")
                    loaded_config[arg_name] = arg_value

        # Handle 4bit/8bit/bf16 overrides
        # Priority: 8bit > bf16 > fp16
        if (hasattr(args, 'load_in_8bit') and args.load_in_8bit) or loaded_config.get("load_in_8bit", False):
            loaded_config["load_in_8bit"] = True
            loaded_config["bf16"] = False
            logger.info("Set bf16 to False due to 8bit override.")

    # --- 3. Store Final Model/Train/Generate Configuration ---
    _CONFIG_STORE = loaded_config
    logger.debug(f"Final Configuration: {json.dumps(_CONFIG_STORE, indent=2)}")


    # --- 4. Setup Tokens and API Keys ---
    dotenv.load_dotenv()

    # --- Handle Hugging Face Token ---
    HUGGINGFACE_TOKEN = args.huggingface_token if hasattr(args, 'huggingface_token') and args.huggingface_token else os.getenv("HUGGINGFACE_TOKEN")
    if not HUGGINGFACE_TOKEN:
        logger.warning("Hugging Face token not provided. Set it via --huggingface_token or HUGGINGFACE_TOKEN environment variable. HF model loading may fail.")

    # --- Handle Google API Key ---
    GOOGLE_API_KEY = args.google_api_key if hasattr(args, 'google_api_key') and args.google_api_key else os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        logger.warning("Google API key not provided. Set it via --google_api_key or GOOGLE_API_KEY environment variable. Gemini model generation may fail.")


    # --- 5. Setup Data Directories ---
    if hasattr(args, 'data_dir_path') and args.data_dir_path:
        DATA_DIR = args.data_dir_path
    logger.info(f"Using DATA_DIR: {DATA_DIR}")
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
    PROCESSED_DATA_FILE_PATH = os.path.join(DATA_DIR, PROCESSED_DATA_FILE_NAME)

    for dir_path in [DATA_DIR, RAW_DIR, CHECKPOINTS_DIR]:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory {dir_path}: {e}", exc_info=True)
    logger.info("Data directories ensured/created.")

    # --- 6. Initialize File Logging (Add Handler Once) ---
    root_logger = logging.getLogger()
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)

    # Check if file logging should be setup
    should_setup_file_logging = not has_file_handler and DATA_DIR

    if should_setup_file_logging:
        log_file_path = os.path.join(DATA_DIR, LOGGING_FILE_NAME)
        try:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s [%(name)s:%(lineno)d] %(message)s')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            if root_logger.level > logging.INFO:
                 root_logger.setLevel(logging.INFO)
            logger.info(f"File logging initialized. Log file: {log_file_path}")
        except Exception as e:
            logger.error(f"Failed to configure file logging to {log_file_path}: {e}. Continuing with console logging only.", exc_info=True)
    elif has_file_handler:
        logger.debug("File logging handler already exists.")
    elif not DATA_DIR:
        logger.warning("DATA_DIR not set. File logging skipped.")


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
        logger.debug(f"Model config store accessed for key '{key}' but is empty. Returning default.")
    return _CONFIG_STORE.get(key, default)

def create_necessary_dirs(file_path: str) -> None:
    """Creates parent directories for a given file path if they don't exist."""
    try:
        dirs = os.path.dirname(file_path)
        if dirs:
            os.makedirs(dirs, exist_ok=True)
    except Exception as e:
        logger.error(f"Unexpected error in create_necessary_dirs for {file_path}: {e}", exc_info=True)

def get_checkpoint_dir(domain: str, model_name: str) -> str:
    """
    Constructs the checkpoint directory path based on domain and model name.

    Args:
        domain: The domain name.
        model_name: The model name.

    Returns:
        The constructed checkpoint directory path.
    """
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, domain, model_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def load_model_and_tokenizer(checkpoint_dir: Optional[str]) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
    """
    Loads a Hugging Face model and tokenizer, handling checkpoints,
    quantization (4/8-bit), LoRA, and data types (bf16/fp16).
    Also handles Gemini model case (returns None, None).

    Args:
        checkpoint_dir: The directory where checkpoints are saved/looked for.
                        Used to find the latest checkpoint if available.

    Returns:
        A tuple containing the loaded model and tokenizer, or (None, None) for Gemini.

    Raises:
        ValueError: If loading fails or parameters are incompatible.
        FileNotFoundError: If the base model specified in config is not found.
    """
    model_name_from_config = get_config("model_name", None)
    assert model_name_from_config, "model_name not found in configuration."

    # --- Handle Gemini Case ---
    if model_name_from_config.lower().startswith("gemini"):
        logger.info(f"Requested model '{model_name_from_config}' is Gemini. Skipping HF load.")
        return None, None

    # --- Determine Model Source (Checkpoint or Base) ---

    print(f"Checkpoint dir: {checkpoint_dir}")
    last_checkpoint = None
    model_source = model_name_from_config

    # -- directly from checkpoint_dir
    if checkpoint_dir:
        last_checkpoint = get_last_checkpoint(checkpoint_dir)

    if last_checkpoint:
        model_source = last_checkpoint

    logger.info(f"Determined model source: {model_source} ({'Checkpoint' if last_checkpoint else 'Base Model'})")

    assert HUGGINGFACE_TOKEN, "Hugging Face token is required for model loading."
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, token=HUGGINGFACE_TOKEN)
    if get_config("load_in_8bit"):
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            token=HUGGINGFACE_TOKEN,
        )
        lora_r = get_config("lora_r", 8)
        # attach LoRA adapter so the model becomes trainable
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=get_config("lora_alpha", lora_r*4),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "up_proj", "down_proj", "gate_proj"],
            lora_dropout=get_config("lora_dropout", 0.05),
            bias=get_config("lora_bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if get_config("bf16") else torch.float16,
            token=HUGGINGFACE_TOKEN,
        )
    return model, tokenizer

