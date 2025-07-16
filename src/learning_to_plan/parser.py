import argparse

from typing import Optional
from learning_to_plan import config
from learning_to_plan.data import task
from learning_to_plan import database
def parse_args():
    parser = argparse.ArgumentParser(description="Learning to Plan")
    parser.add_argument(
        "-d", "--domains",
        type = str,
        default="all",
        help="Comma-separated list of domains to process. Use 'all' to process all available domains. Defaults to 'all'."
    )
    # --- Action Flags ---
    parser.add_argument(
        "--get_tasks_from_raw_data",
        action="store_true",
        help="Create tasks from raw data."
    )
    parser.add_argument(
        "--call_paas",
        action="store_true",
        help="Call planning as a service to generate plans."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a model on the finetuning dataset."
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Use a trained model to generate plans for test instances."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate model generated plans using VAL"
    )
    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        help="Compute metrics for the generated plans."
    )
    # --- Configuration & Overrides ---
    parser.add_argument(
        "--landmark_factory",
        type=str,
        default="lm_zg",
        help="Landmark factory to use for generating landmarks. Default is 'lm_zg'."
    )
    parser.add_argument(
        "-c", "--config_file_path",
        type=str,
        default=None,
        help="Path to a custom JSON configuration file (overrides defaults)."
    )
    parser.add_argument(
        "-m", "--model_name",
        type=str,
        default=None,
        help="Override model name (e.g., 'Qwen/Qwen2.5-7B-Instruct' or 'gemini-pro') specified in config."
    )
    parser.add_argument(
        "-e", "--num_train_epochs",
        type=int,
        default=None,
        help="Override number of training epochs specified in config."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Override config to load the base model in 8bit (for training/generation)."
    )
    parser.add_argument(
        "--data_dir_path",
        type=str,
        default=None, # Default is handled in config.py now
        help="Path to the base data directory (containing raw, paas_plans, etc.). Defaults to './data/'."
    )
    parser.add_argument(
        "--database_file_path",
        type=str,
        default=None,  # Default is handled in config.py now
        help="Path to the SQLite database file. Defaults to './data/learning_to_plan.db'."
    )
    parser.add_argument(
        "--overwrite_generated_plans",
        action="store_true",
        help="Overwrite the generated plans if they already exist."
    )
    def number_of_instances_type(value):
        if value.isdigit():
            assert int(value) > 0, "Number of instances must be a positive integer."
            return int(value)
        elif value == "all":
            return value
        else:
            raise argparse.ArgumentTypeError(f"Invalid value for number_of_instances: {value}. Must be 'all',  or a positive integer.")
    parser.add_argument(
        "-n", "--number_of_instances",
        type=number_of_instances_type,
        default="all",
        help="Number of instances to generate plans for. Can be 'all', or a positive integer."
    )
    def task_type_converter(value: Optional[str] = None) -> Optional[task.Task.TYPE]:
        if not value:
            return None
        try:
            return config.get_enum_value(value, task.Task.TYPE, "task_type")
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"Invalid task_type: {value}. Valid options are: {', '.join([t.value for t in task.Task.TYPE])}."
            )
    parser.add_argument(
        "--task_type",
        type=task_type_converter,
        default=None,
        help="Type of task to filter by. Options: 'indistribution', 'outofdistribution', 'unseen', 'obfuscated'. Default is None (no filtering)."
    )
    def prompt_type_converter(value: Optional[str] = None) -> Optional[config.PROMPT_TYPE]:
        if not value:
            return None
        try:
            return config.PROMPT_TYPE[value.upper()]
        except KeyError:
            valid = ", ".join([pt.value for pt in config.PROMPT_TYPE])
            raise argparse.ArgumentTypeError(f"Invalid prompt_type: {value}. Valid options are: {valid}.")
    parser.add_argument(
        "--prompt_type",
        type=prompt_type_converter,
        default=None,
        help=f"Type of prompt to use for plan generation. Options: {list(config.PROMPT_TYPE)}. Default is {config.PROMPT_TYPE.IO.name}."
    )
    parser.add_argument(
        "--few_shot",
        type=int,
        default=0,
        help="Number of few-shot examples to use for generation. Default is 0 (no few-shot examples)."
    )
    parser.add_argument(
        "--dont_use_checkpoint",
        action="store_true",
        help="Do not use the latest checkpoint for training."
    )
    # --- Credentials ---
    parser.add_argument(
        "--huggingface_token",
        type=str,
        default=None,
        help="Hugging Face token (overrides HUGGINGFACE_TOKEN env var)."
    )
    parser.add_argument(
        "--google_api_key", # Added argument for Google API Key
        type=str,
        default=None,
        help="Google API Key (overrides GOOGLE_API_KEY env var)."
    )
    parser.add_argument(
        "-s", "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate per task. Default is 1."
    )

    return parser.parse_args()