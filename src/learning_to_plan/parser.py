# main.py
import os
import argparse
from learning_to_plan import config
def parse_args():
    parser = argparse.ArgumentParser(description="Learning to Plan")
    parser.add_argument(
        "-d", "--domain",
        type=str,
        default="all",
        help="List of domains separated by commas (e.g., 'blocksworld,logistics') or 'all'."
    )
    # --- Action Flags ---
    parser.add_argument(
        "--call_paas",
        action="store_true",
        help="Call planning as a service to generate plans."
    )
    parser.add_argument(
        "--split_dataset",
        action="store_true",
        help="Split the dataset into training, validation, and test sets."
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
    parser.add_argument(
        "--clear_model_dir",
        action="store_true",
        help="Clear the models directory before training or generation."
    )
    parser.add_argument(
        "--landmarks_generation",
        action="store_true",
        help="Generate landmark graphs for all tasks in the dataset."
    )
    # TODO: CREATE A FUNCTION TO SAVE THE MOST RECENT CHECKPOINT FOR ALL MODELS (OR A SINGLE MODEL).
    # IT SHOULD SAVE THE FILE NAME, THE DATE, THE CHECKPOINT AND SOME OTHER METADATA.
    # --- Configuration & Overrides ---
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
        "--tasks_dataset_file_path",
        type=str,
        default=None,
        help="Path to the tasks dataset file (e.g., 'data/tasks.jsonl'). Defaults to './data/tasks.jsonl'."
    )
    parser.add_argument(
        "--overwrite_generated_plans",
        action="store_true",
        help="Overwrite the generated plans if they already exist."
    )
    def number_of_instances_type(value):
        if value.isdigit():
            return int(value)
        elif value in ["all", "long", "basic"]:
            return value
        else:
            raise argparse.ArgumentTypeError(f"Invalid value for number_of_instances: {value}. Must be 'all', 'long', 'basic', or a positive integer.")
    parser.add_argument(
        "-n", "--number_of_instances",
        type=number_of_instances_type,
        default="all",
        help="Number of instances to generate plans for. Can be 'all', 'long', 'basic', or a positive integer."
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

from typing import Optional
def get_selected_domains(args, dir:Optional[str]=None, is_file:bool=False) -> set[str]:
    if not args.domain:
        logger.error("Please specify a domain with --domain <domain_name> or 'all'.")
        raise ValueError("Domain not specified.")

    if dir:
        assert os.path.isdir(dir), f"Directory {dir} does not exist."
        try:
            available_domains = {d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))}
        except OSError as e:
            logger.error(f"Error listing domains in {dir}: {e}", exc_info=True)
            raise e
        assert available_domains and len(available_domains) > 0, f"No domains found in {dir}."
    elif is_file:
        tasks = database.get_dataset()
        assert tasks, f"No tasks found in {config.TASKS_DATASET_FILE_PATH}."
        available_domains = {t._domain for t in tasks}
        assert available_domains, f"No domains found in {config.TASKS_DATASET_FILE_PATH}."
    else:
        logger.error("No directory or file specified for domain selection.")
        raise ValueError("No directory or file specified for domain selection.")

    if args.domain.lower() == "all":
        logger.info(f"Processing all found domains: {', '.join(available_domains)}")
        return available_domains
    else:
        selected = set(s.strip() for s in args.domain.split(","))
        assert selected.issubset(available_domains), f"Selected domains {selected} are not in available domains {available_domains}."
        selected = selected.intersection(available_domains)
        assert len(selected) > 0, f"No valid domains selected from {args.domain}."
        logger.info(f"Processing selected domains: {', '.join(selected)}")
        return selected
