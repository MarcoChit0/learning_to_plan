# main.py
import os
import argparse
import asyncio

# Import project modules
from learning_to_plan import train
from learning_to_plan import utils
from learning_to_plan import task
from learning_to_plan import config # Import the refactored config
from learning_to_plan import generate # Import the new generate module

logger = config.get_logger(__name__)

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
        "-n", "--number_of_instances",
        type=str,
        default="all",
        help="Number of problems per domain: positive integer or 'all', 'long', 'basic'"
    )
    # --- Generate Specific ---
    parser.add_argument(
        "--cot",
        type=int,
        default=0,
        help="Number of Chain of Thought (CoT) steps to use."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
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
        tasks = task.get_dataset()
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

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    config.initialize(args) # Config initialization likely sets up logging

    # --- Action Blocks ---
    if args.call_paas:
        logger.info("--- Starting Planning as a Service (PaaS) Calls ---")
        domains = get_selected_domains(args, dir=config.RAW_DIR)
        for domain in domains:
            logger.info(f"Processing PaaS for domain: {domain}")
            asyncio.run(utils.call_paas(domain=domain))
            logger.info(f"Finished PaaS calls for domain: {domain}")
        logger.info("--- Finished All PaaS Calls ---")

    elif args.split_dataset:
        logger.info("--- Starting Dataset Splitting ---")
        utils.split_dataset(random_seed=args.random_seed)
        logger.info("--- Finished All Dataset Splitting ---")

    elif args.train:
        logger.info("--- Starting Model Training ---")
        config_file_path = args.config_file_path or os.path.join(config.CONFIGS_DIR, config.DEFAULT_TRAIN_CONFIG)
        train_kwargs = config.get_config(config_file_path=config_file_path, args=args)
        assert train_kwargs["model_name"], "Model name not found in config. Please check your configuration."
        domains = get_selected_domains(args=args, is_file=True)
        for domain in domains:
            logger.info(f"Starting training for domain: {domain}")
            train.run_training_procedure(model_name=train_kwargs["model_name"], domain=domain, **train_kwargs)
            logger.info(f"Finished training for domain: {domain}")
        logger.info("--- Finished All Training ---")

    elif args.generate:
        logger.info("--- Starting Generation ---")
        config_file_path = args.config_file_path or os.path.join(config.CONFIGS_DIR, config.DEFAULT_GENERATE_CONFIG)
        generate_kwargs = config.get_config(config_file_path=config_file_path, args=args)
        assert generate_kwargs["model_name"], "Model name not found in config. Please check your configuration."
        domains = get_selected_domains(args=args, is_file=True)
        for domain in domains:
            logger.info(f"Starting generation for domain: {domain}")
            checkpoint_dir = config.get_checkpoint_dir(domain, generate_kwargs["model_name"]) if not args.dont_use_checkpoint else None
            generate.generate_batch(model_name=generate_kwargs["model_name"], domain=domain, number_of_instances=args.number_of_instances, random_seed=args.random_seed, number_of_cot_examples=args.cot, checkpoint_dir=checkpoint_dir, **generate_kwargs)
            logger.info(f"Finished generation for domain: {domain}")
        logger.info("--- Finished All Generation ---")

    # elif args.validate:
    #     logger.info("--- Starting Validation ---")
    #     validate.validate_plans(data_file_path=data_file_path)
    #     logger.info("--- Finished All Validation ---")

    # elif args.compute_metrics:
    #     logger.info("--- Starting Metric Computation ---")
    #     metrics.compute_metrics(data_file_path=data_file_path)
    #     logger.info("--- Finished Metric Computation ---")

    else:
        logger.warning("No action requested (e.g., --train, --generate). Exiting.")

    logger.info("--- Main script execution finished ---")