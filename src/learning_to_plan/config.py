import os
import json
import logging
import argparse
from typing import Optional, Dict, Any
import dotenv

# --- Global Variables for Paths and Token (Set during initialization) ---
DATA_DIR = "data/"
RAW_DIR: Optional[str] = None
CHECKPOINTS_DIR: Optional[str] = None
MODELS_DIR: Optional[str] = None
HUGGINGFACE_TOKEN: Optional[str] = None
GOOGLE_API_KEY: Optional[str] = None
DATABASE_FILE_PATH: Optional[str] = None
RAW_DIR_STRUCTURE_FILE_PATH: Optional[str] = None
LOGGING_INITIALIZED: bool = False
METRICS_FILE_PATH: Optional[str] = None
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

logger = get_logger(__name__) # Initialize logger


# --- Constants for directory/file names ---
BASIC_INSTANCES = "generated_basic"
LONG_INSTANCES = "generated_basic_longer_plan_len"
DATABASE_FILE_NAME = "learning_to_plan.db"
DOMAIN_FILE_NAME = "generated_domain.pddl"
LOGGING_FILE_NAME = "logs.log"
RAW_DIR_STRUCTURE_FILE_NAME = "structure.json"
METRICS_FILE_NAME = "metrics.csv"

# --- Default Config Files ---
CONFIG_DIR_PATH : str = os.path.join("src", "configs")
GENERATE_CONFIG_DIR_PATH : str = os.path.join(CONFIG_DIR_PATH, "generate")
TRAIN_CONFIG_DIR_PATH : str = os.path.join(CONFIG_DIR_PATH, "train")
DEFAULT_TRAIN_CONFIG_FILE_PATH: str = os.path.join(TRAIN_CONFIG_DIR_PATH, "qwen.json")
DEFAULT_GENERATE_CONFIG_FILE_PATH: str = os.path.join(GENERATE_CONFIG_DIR_PATH, "qwen.json")


RANDOM_SEED = 42 

from enum import Enum
class TOKENS(Enum):
    PLAN_START = "<plan>"
    PLAN_END = "</plan>"
    DOMAIN_START = "<domain>"
    DOMAIN_END = "</domain>"
    GOAL_START = "<goal>"
    GOAL_END = "</goal>"
    INITIAL_STATE_START = "<initial_state>"
    INITIAL_STATE_END = "</initial_state>"
    EXAMPLE_START = "<example>"
    EXAMPLE_END = "</example>"
    PROBLEM_START = "<instance>"
    PROBLEM_END = "</instance>"
    CHECKLIST_START = "<checklist>"
    CHECKLIST_END = "</checklist>"
    LANDMARKS_START = "<landmarks>"
    LANDMARKS_END = "</landmarks>"

class STATUS(Enum):
    OK = "ok"
    ERROR = "error"


# TO ADD A NEW PROMPT TYPE, YOU MUST CHANGE THE FOLLOWING METHODS:
# 1. PROMPT_TYPE Enum to include the new type.
# 2. config.get_special_tokens function to return the appropriate tokens for the new type.
# 3. task.Task.get_prompt_metadata method to handle the new prompt type.
class PROMPT_TYPE(Enum):
    IO = "io"
    FEW_SHOT = "few_shot"
    PDDL = "pddl"
    LANDMARKS = "landmarks"

    def __lt__(self, other):
        if isinstance(other, PROMPT_TYPE):
            return self.value < other.value
        return NotImplemented

def get_special_tokens(prompt_type: PROMPT_TYPE) -> list[str]:
    if prompt_type == PROMPT_TYPE.IO:
        tokens = [
            TOKENS.PLAN_START,
            TOKENS.PLAN_END,
        ]
    elif prompt_type == PROMPT_TYPE.FEW_SHOT:
        tokens = [
            TOKENS.PLAN_START,
            TOKENS.PLAN_END,
            TOKENS.DOMAIN_START,
            TOKENS.DOMAIN_END,
            TOKENS.GOAL_START,
            TOKENS.GOAL_END,
            TOKENS.INITIAL_STATE_START,
            TOKENS.INITIAL_STATE_END,
            TOKENS.EXAMPLE_START,
            TOKENS.EXAMPLE_END,
            TOKENS.CHECKLIST_START,
            TOKENS.CHECKLIST_END,
        ]
    elif prompt_type == PROMPT_TYPE.PDDL or prompt_type == PROMPT_TYPE.LANDMARKS:
        tokens = [
            TOKENS.PLAN_START,
            TOKENS.PLAN_END,
            TOKENS.DOMAIN_START,
            TOKENS.DOMAIN_END,
            TOKENS.EXAMPLE_START,
            TOKENS.EXAMPLE_END,
            TOKENS.PROBLEM_START,
            TOKENS.PROBLEM_END,
            TOKENS.CHECKLIST_START,
            TOKENS.CHECKLIST_END,
        ]
        if prompt_type == PROMPT_TYPE.LANDMARKS:
            tokens.append(TOKENS.LANDMARKS_START)
            tokens.append(TOKENS.LANDMARKS_END)

    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Must be one of {list(PROMPT_TYPE)}.")
    return [token.value for token in tokens]
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
    global HUGGINGFACE_TOKEN, GOOGLE_API_KEY
    global DATA_DIR, RAW_DIR, CHECKPOINTS_DIR, MODELS_DIR, DATABASE_FILE_NAME, DATABASE_FILE_PATH, RAW_DIR_STRUCTURE_FILE_PATH, RAW_DIR_STRUCTURE_FILE_NAME
    global LOGGING_INITIALIZED, logger

    logger.info("Initializing environment variables...")
    dotenv.load_dotenv()

    # --- Handle Hugging Face Token ---
    HUGGINGFACE_TOKEN = args.huggingface_token if hasattr(args, 'huggingface_token') and args.huggingface_token else os.getenv("HUGGINGFACE_TOKEN")
    if not HUGGINGFACE_TOKEN:
        logger.warning("Hugging Face token not provided. Set it via --huggingface_token or HUGGINGFACE_TOKEN environment variable. HF model loading may fail.")

    # --- Handle Google API Key ---
    GOOGLE_API_KEY = args.google_api_key if hasattr(args, 'google_api_key') and args.google_api_key else os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        logger.warning("Google API key not provided. Set it via --google_api_key or GOOGLE_API_KEY environment variable. Gemini model generation may fail.")


    # --- Setup Data Directories ---
    logger.info("Setting up data directories...")
    if hasattr(args, 'data_dir_path') and args.data_dir_path:
        DATA_DIR = args.data_dir_path
    logger.info(f"Using DATA_DIR: {DATA_DIR}")
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
    MODELS_DIR = os.path.join(DATA_DIR, "models")
    
    if hasattr(args, 'database_file_path') and args.database_file_path:
        DATABASE_FILE_PATH = args.database_file_path
    else:
        DATABASE_FILE_PATH = os.path.join(DATA_DIR, DATABASE_FILE_NAME)

    for dir_path in [DATA_DIR, RAW_DIR, CHECKPOINTS_DIR, MODELS_DIR]:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory {dir_path}: {e}", exc_info=True)
    logger.info("Data directories ensured/created.")

    RAW_DIR_STRUCTURE_FILE_PATH = os.path.join(RAW_DIR, RAW_DIR_STRUCTURE_FILE_NAME)
    assert os.path.exists(RAW_DIR_STRUCTURE_FILE_PATH), f"Raw directory structure file {RAW_DIR_STRUCTURE_FILE_PATH} does not exist. Please ensure it is created before running the application."

    global METRICS_FILE_PATH, METRICS_FILE_NAME
    METRICS_FILE_PATH = os.path.join(DATA_DIR, METRICS_FILE_NAME)

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

    # --- Initialize Databases ---
    from learning_to_plan import database
    database.initialize()


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

def get_selected_domains(domains : str) -> set[str]:
    from learning_to_plan import database
    tasks = database.task_database.get()
    available_domains = {t.domain for t in tasks}
    if domains.lower() == "all":
        return available_domains
    else:
        selected = set(s.strip() for s in domains.split(","))
        assert selected.issubset(available_domains), f"Selected domains {selected} are not in available domains {available_domains}."
        selected = selected.intersection(available_domains)
        assert len(selected) > 0, f"No valid domains selected from {domains}."
        return selected

def get_config(config_file_path = None, **kwargs) -> Dict[str, Any]:
    if not config_file_path:
        return kwargs

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file '{config_file_path}' not found.")
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            file_config = json.load(f)

        config = kwargs.copy()
        config.update(file_config)

        domains = config.pop("domains", None)
        assert domains is not None, "No domains specified in config. Please check your configuration."
        config["domains"] = get_selected_domains(domains)

        if not config.get("num_samples") or config.get("num_samples") <= 0:
            logger.warning("num_samples not set or invalid in config. Defaulting to 1.")
            config["num_samples"] = 1

        if not config.get("model_name"):
            raise ValueError("Model name not found in config. Please check your configuration.")

        try:
            from learning_to_plan.prompt_builder.utils import get_prompt_builder
            prompt_builder = get_prompt_builder(**config)
            config['prompt_builder'] = prompt_builder
        except ValueError as e:
            raise ValueError(f"Error initializing prompt builder: {e}") from e

        logger.info(f"Final configuration: {config}")
        return config
    except Exception as e:
        msg = f"Error loading config file {config_file_path}: {e}"
        logger.error(msg, exc_info=True)
        raise ValueError(msg) from e
    
def get_enum_value(value, enum_cls, name):
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            logger.error(f"Invalid {name} value: {value}. Defaulting to None.")
    return None