# main.py
import os
import argparse
import asyncio
import logging # Import standard logging for potential direct use if needed
from typing import Optional

# Import project modules
from learning_to_plan import build_finetuning_dataset
from learning_to_plan import train
from learning_to_plan import utils
from learning_to_plan import task
from learning_to_plan import config # Import the refactored config
from learning_to_plan import generate # Import the new generate module

def parse_args():
    parser = argparse.ArgumentParser(description="Learning to Plan")
    parser.add_argument(
        "-d", "--domain",
        type=str,
        default="",
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
    # --- Configuration & Overrides ---
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to a custom JSON configuration file (overrides defaults)."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Override model name (e.g., 'Qwen/Qwen2.5-7B-Instruct') specified in config."
    )
    parser.add_argument(
        "-e", "--epochs",
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
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save generated outputs (e.g., generated_plans.jsonl). Required if --generate is used."
    )
    # --- PaaS Specific ---
    parser.add_argument(
        "--overwrite_paas_plans",
        action="store_true",
        help="Overwrite existing PaaS plan files."
    )
    # Custom type function to accept positive integers or specific strings
    def parse_number_of_problems_arg(arg):
        if arg in ["all", "long", "basic"]:
            return arg
        try:
            val = int(arg)
            if val <= 0:
                raise argparse.ArgumentTypeError("Integer must be positive")
            return val
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a positive integer or 'all', 'long', or 'basic'")
    
    parser.add_argument(
        "-n", "--number_of_problems",
        type=parse_number_of_problems_arg,
        default="all",
        help="Number of problems per domain: positive integer or 'all', 'long', 'basic'"
    )
    # --- Credentials ---
    parser.add_argument(
        "--huggingface_token",
        type=str,
        default=None,
        help="Hugging Face token (overrides HUGGINGFACE_TOKEN env var)."
    )

    return parser.parse_args()

# --- Helper Functions ---
def verify_domain(args):
    if not args.domain:
        # Use config's logger/printer
        config.log("Please specify a domain with --domain <domain_name> or 'all'.", level=logging.ERROR)
        raise ValueError("Domain not specified.")

def get_selected_domains(args, base_dir):
    if not os.path.isdir(base_dir):
         config.log(f"Base directory for domains not found: {base_dir}", level=logging.ERROR)
         raise FileNotFoundError(f"Base directory not found: {base_dir}")
    try:
        available_domains = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    except OSError as e:
        config.log(f"Error listing domains in {base_dir}: {e}", level=logging.ERROR, exc_info=True)
        raise e

    if not available_domains:
        config.log(f"No domain subdirectories found in {base_dir}.", level=logging.WARNING)
        return []

    if args.domain.lower() == "all":
        config.log(f"Processing all found domains: {', '.join(available_domains)}")
        return available_domains
    else:
        selected = args.domain.split(",")
        valid_domains = []
        for d in selected:
            if d in available_domains:
                valid_domains.append(d)
            else:
                config.log(f"Specified domain '{d}' not found in {base_dir}. Skipping.", level=logging.WARNING)
        if not valid_domains:
             config.log(f"None of the specified domains ({args.domain}) were found in {base_dir}.", level=logging.ERROR)
             raise ValueError(f"Specified domains not found: {args.domain}")
        config.log(f"Processing selected domains: {', '.join(valid_domains)}")
        return valid_domains

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()

    config.initialize(args, config_path=args.config_path)

    # --- Action Blocks ---
    if args.call_paas:
        config.log("--- Starting Planning as a Service (PaaS) Calls ---")
        verify_domain(args)
        domains = get_selected_domains(args, config.RAW_DIR)
        for domain in domains:
            config.log(f"Processing PaaS for domain: {domain}")
            tasks = task.get_tasks_from_domain_directory(domain, args.number_of_problems_per_domain, args.problem_size)
            if not tasks:
                config.log(f"No tasks found for domain {domain}. Skipping.", level=logging.WARNING)
                continue
            data_file_path = os.path.join(config.PROCESSED_DATA_DIR, domain, config.PROCESSED_DATA_FILE_NAME)
            config.log(f"Outputting PaaS results to: {data_file_path}")
            asyncio.run(utils.call_paas(tasks, data_file_path, overwrite=args.overwrite_paas_plans))
            config.log(f"Finished PaaS calls for domain: {domain}")
        config.log("--- Finished All PaaS Calls ---")

    elif args.split_dataset:
        config.log("--- Starting Dataset Splitting ---")
        verify_domain(args)
        domains = get_selected_domains(args, config.PROCESSED_DATA_DIR)
        for domain in domains:
            config.log(f"Splitting dataset for domain: {domain}")
            data_file_path = os.path.join(config.PROCESSED_DATA_DIR, domain, config.PROCESSED_DATA_FILE_NAME)
            try:
                utils.split_dataset(data_file_path)
                config.log(f"Finished splitting dataset for domain: {domain}")
            except Exception as e:
                config.log(f"Error splitting dataset for domain {domain}: {e}", level=logging.ERROR)
                continue
        config.log("--- Finished All Dataset Splitting ---")

    elif args.train:
        config.log("--- Starting Model Training ---")
        verify_domain(args)

        domains = get_selected_domains(args, config.FINETUNING_DATASET_DIR)
        for domain in domains:
            config.log(f"Starting training for domain: {domain}")
            train_file = os.path.join(config.FINETUNING_DATASET_DIR, domain, config.TRAIN_FILE_NAME)
            val_file   = os.path.join(config.FINETUNING_DATASET_DIR, domain, config.VAL_FILE_NAME)
            # Construct checkpoint dir using the potentially overridden model name
            current_model_name = config.get_config("model_name") # Get final model name after potential override
            model_checkpoint_dir = os.path.join(config.CHECKPOINTS_DIR, current_model_name, domain)

            config.create_necessary_dirs(model_checkpoint_dir) # Use helper from config
            config.log(f"Checkpoints will be saved to: {model_checkpoint_dir}")

            # Ensure input files exist
            if not os.path.exists(train_file) or not os.path.exists(val_file):
                config.log(f"Train ({train_file}) or Validation ({val_file}) file missing for domain {domain}. Skipping training.", level=logging.ERROR)
                raise FileNotFoundError(f"Missing train/validation files for domain {domain}.")

            train.run_training_procedure(model_checkpoint_dir, train_file, val_file)
            config.log(f"Finished training for domain: {domain}")
        config.log("--- Finished All Training ---")

    elif args.generate:
        config.log("--- Starting Plan Generation ---")
        verify_domain(args)
        if not args.output_dir:
             config.log("--output_dir is required when using --generate.", level=logging.ERROR)
             raise ValueError("--output_dir not specified.")

        model_name_from_config = config.get_config("model_name") # Get name used/loaded during init
        if not model_name_from_config:
             config.log("Model name not found in configuration. Cannot determine checkpoint directory.", level=logging.ERROR)
             raise ValueError("Model name missing in config for generation.")

        # Base directory where domain-specific checkpoints are stored
        model_checkpoints_base_dir = os.path.join(config.CHECKPOINTS_DIR, model_name_from_config)
        config.log(f"Looking for checkpoints in base directory: {model_checkpoints_base_dir}")

        domains = get_selected_domains(args, model_checkpoints_base_dir) # Check domains within the specific model's checkpoint dir
        if not domains:
             config.log(f"No trained domain checkpoints found for model '{model_name_from_config}' in {model_checkpoints_base_dir}.", level=logging.ERROR)
             raise FileNotFoundError("No matching domain checkpoints found.")

        for domain in domains:
            config.log(f"Starting generation for domain: {domain}")
            test_file = os.path.join(config.FINETUNING_DATASET_DIR, domain, config.TEST_FILE_NAME)
            model_domain_checkpoint_dir = os.path.join(model_checkpoints_base_dir, domain) # Path to the specific trained checkpoint
            data_file_path = os.path.join(args.output_dir, model_name_from_config, domain, "generated_plans.jsonl")

            # Ensure input test file exists
            if not os.path.exists(test_file):
                config.log(f"Test file not found: {test_file}. Skipping generation for domain {domain}.", level=logging.ERROR)
                continue

            # Ensure checkpoint directory exists
            if not os.path.isdir(model_domain_checkpoint_dir):
                 config.log(f"Checkpoint directory not found: {model_domain_checkpoint_dir}. Skipping generation for domain {domain}.", level=logging.ERROR)
                 continue

            config.log(f"Using model checkpoint: {model_domain_checkpoint_dir}")
            config.log(f"Using test instances: {test_file}")
            config.log(f"Saving generated outputs to: {data_file_path}")

            # Create output directory if needed
            config.create_necessary_dirs(data_file_path)

            # Call the generation function from generate.py
            generate.generate_batch(
                checkpoint_model_dir=model_domain_checkpoint_dir,
                test_file=test_file,
                output_jsonl_path=data_file_path
                # Pass other args like max_instances if needed
            )
            config.log(f"Finished generation for domain: {domain}")
        config.log("--- Finished All Generation ---")

    else:
        config.log("No action requested (e.g., --train, --generate). Exiting.", level=logging.WARNING)

    config.log("--- Main script execution finished ---")