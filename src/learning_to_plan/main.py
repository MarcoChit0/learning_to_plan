# main.py
import os
# Removed: os.environ["NCCL_P2P_DISABLE"] = "1" (Didn't solve the issue)
# os.environ["NCCL_DEBUG"] = "INFO" # Optional: Uncomment for more detailed NCCL logs

import argparse
import asyncio
import logging

# Import project modules
from learning_to_plan import train
from learning_to_plan import utils
from learning_to_plan import task
from learning_to_plan import config # Import the refactored config
from learning_to_plan import generate # Import the new generate module

# --- NOTE for Kaggle/Multi-GPU Execution ---
# If still encountering NCCL/CUDA errors during training,
# try launching this script using accelerate:
# Example: accelerate launch main.py --train --domain blocksworld [other_args...]
# This ensures the environment is correctly set up for DistributedDataParallel (DDP).
# The Trainer class should automatically detect and use the accelerate environment.
# ------------------------------------------

def parse_args():
    """Parses command-line arguments."""
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
    # --- Configuration & Overrides ---
    parser.add_argument(
        "--config_path",
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
        "--load_in_4bit",
        action="store_true",
        help="Override config to load the base model in 4bit (for training/generation)."
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
    # --- PaaS Specific ---
    parser.add_argument(
        "--overwrite_paas_plans",
        action="store_true",
        help="Overwrite existing PaaS plan files."
    )
    # Custom type function to accept positive integers or specific strings
    def parse_number_of_problems_per_domain_arg(arg):
        """Validates the number_of_problems_per_domain argument."""
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
        "-n", "--number_of_problems_per_domain",
        type=parse_number_of_problems_per_domain_arg,
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
    parser.add_argument(
        "--google_api_key", # Added argument for Google API Key
        type=str,
        default=None,
        help="Google API Key (overrides GOOGLE_API_KEY env var)."
    )

    return parser.parse_args()

def get_selected_domains(args, base_dir):
    """Determines the list of domains to process based on args and available directories."""
    if not args.domain:
        # Use config's logger/printer
        config.log("Please specify a domain with --domain <domain_name> or 'all'.", level=logging.ERROR)
        raise ValueError("Domain not specified.")

    if not os.path.isdir(base_dir):
        m = f"Base directory for domains not found: {base_dir}"
        config.log(m, level=logging.ERROR)
        raise FileNotFoundError(m)
    try:
        # List only directories within the base_dir
        available_domains = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    except OSError as e:
        config.log(f"Error listing domains in {base_dir}: {e}", level=logging.ERROR, exc_info=True)
        raise e

    if not available_domains:
        m = f"No domain subdirectories found in {base_dir}."
        config.log(m, level=logging.ERROR)
        raise ValueError(m)

    if args.domain.lower() == "all":
        config.log(f"Processing all found domains: {', '.join(available_domains)}")
        return available_domains
    else:
        selected = [d.strip() for d in args.domain.split(",")] # Strip whitespace
        valid_selected = []
        for d in selected:
            if d not in available_domains:
                m = f"Domain '{d}' not found in {base_dir}. Available domains: {', '.join(available_domains)}"
                config.log(m, level=logging.ERROR)
                # Optionally raise error immediately or collect all errors
                # For now, let's raise immediately
                raise ValueError(m)
            valid_selected.append(d) # Add if valid
        config.log(f"Processing selected domains: {', '.join(valid_selected)}")
        return valid_selected

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()

    # Initialize configuration
    config.initialize(args, config_path=args.config_path)

    # --- Action Blocks ---
    if args.call_paas:
        config.log("--- Starting Planning as a Service (PaaS) Calls ---")
        domains = get_selected_domains(args, config.RAW_DIR)
        for domain in domains:
            config.log(f"Processing PaaS for domain: {domain}")
            try:
                tasks = task.get_tasks_from_domain_directory(domain, args.number_of_problems_per_domain)
                if not tasks:
                    config.log(f"No tasks found or selected for domain {domain}. Skipping.", level=logging.WARNING)
                    continue
                data_file_path = os.path.join(config.PROCESSED_DATA_DIR, domain, config.PROCESSED_DATA_FILE_NAME)
                config.log(f"Outputting PaaS results to: {data_file_path}")
                asyncio.run(utils.call_paas(tasks, data_file_path, overwrite=args.overwrite_paas_plans))
                config.log(f"Finished PaaS calls for domain: {domain}")
            except Exception as e:
                 config.log(f"Error processing PaaS for domain {domain}: {e}", level=logging.ERROR, exc_info=True)
                 continue # Continue to the next domain
        config.log("--- Finished All PaaS Calls ---")

    elif args.split_dataset:
        config.log("--- Starting Dataset Splitting ---")
        domains = get_selected_domains(args, config.PROCESSED_DATA_DIR)
        for domain in domains:
            config.log(f"Splitting dataset for domain: {domain}")
            data_file_path = os.path.join(config.PROCESSED_DATA_DIR, domain, config.PROCESSED_DATA_FILE_NAME)
            try:
                utils.split_dataset(data_file_path)
                config.log(f"Finished splitting dataset for domain: {domain}")
            except FileNotFoundError:
                 config.log(f"Data file not found for splitting: {data_file_path}. Skipping domain {domain}.", level=logging.ERROR)
                 continue
            except Exception as e:
                config.log(f"Error splitting dataset for domain {domain}: {e}", level=logging.ERROR, exc_info=True)
                continue # Continue to the next domain
        config.log("--- Finished All Dataset Splitting ---")

    elif args.train:
        config.log("--- Starting Model Training ---")

        model_name_from_config = config.get_config("model_name")
        if model_name_from_config and model_name_from_config.lower().startswith("gemini"):
            m = f"Model '{model_name_from_config}' is a Gemini model. Training is not supported for Gemini."
            config.log(m, level=logging.ERROR)
            raise ValueError(m)

        domains = get_selected_domains(args, config.PROCESSED_DATA_DIR)
        for domain in domains:
            config.log(f"Starting training for domain: {domain}")
            data_file_path = os.path.join(config.PROCESSED_DATA_DIR, domain, config.PROCESSED_DATA_FILE_NAME)
            if not os.path.exists(data_file_path):
                 config.log(f"Data file not found for training: {data_file_path}. Skipping domain {domain}.", level=logging.ERROR)
                 continue

            # Construct checkpoint dir using the potentially overridden model name
            current_model_name = config.get_config("model_name") # Get final model name after potential override
            if not current_model_name:
                 config.log(f"Model name not configured. Skipping training for domain {domain}.", level=logging.ERROR)
                 continue
            model_checkpoint_dir = os.path.join(config.CHECKPOINTS_DIR, current_model_name, domain)
            config.create_necessary_dirs(model_checkpoint_dir) # Use helper from config
            config.log(f"Checkpoints will be saved to: {model_checkpoint_dir}")

            try:
                train.run_training_procedure(model_checkpoint_dir, data_file_path)
                config.log(f"Finished training for domain: {domain}")
            except Exception as e:
                 config.log(f"Error during training for domain {domain}: {e}", level=logging.ERROR, exc_info=True)
                 # Decide if you want to stop all training or continue to the next domain
                 continue # Continue to the next domain
        config.log("--- Finished All Training ---")

    elif args.generate:
        config.log("--- Starting Plan Generation ---")
        model_name_from_config = config.get_config("model_name")
        assert model_name_from_config, "Model name not found in config. Please check your configuration."

        # Determine base directory for finding domains based on model type
        if model_name_from_config.lower().startswith("gemini"):
            config.log(f"Model '{model_name_from_config}' is a Gemini model. Calling API for generation.")
            # For Gemini, domains are defined by processed data directories
            domain_base_dir = config.PROCESSED_DATA_DIR
        else:
            # For HF models, domains are defined by checkpoint directories
            domain_base_dir = os.path.join(config.CHECKPOINTS_DIR, model_name_from_config)
            config.log(f"Looking for model checkpoints/domains in base directory: {domain_base_dir}")

        try:
            domains = get_selected_domains(args, domain_base_dir)
            if not domains:
                 config.log(f"No valid domains found in '{domain_base_dir}' for model '{model_name_from_config}'. Generation cannot proceed.", level=logging.ERROR)
            else:
                for domain in domains:
                    config.log(f"Starting generation for domain: {domain}")
                    data_file_path = os.path.join(config.PROCESSED_DATA_DIR, domain, config.PROCESSED_DATA_FILE_NAME)
                    if not os.path.exists(data_file_path):
                        config.log(f"Data file not found for generation: {data_file_path}. Skipping domain {domain}.", level=logging.ERROR)
                        continue
                    config.log(f"Using data: {data_file_path}.")

                    model_checkpoint_dir = None # Default to None (for Gemini)
                    if not model_name_from_config.lower().startswith("gemini"):
                        # Construct and check checkpoint path for HF models
                        model_checkpoint_dir = os.path.join(config.CHECKPOINTS_DIR, model_name_from_config, domain)
                        if not os.path.exists(model_checkpoint_dir):
                             config.log(f"Checkpoint directory not found: {model_checkpoint_dir}. Skipping generation for domain {domain}.", level=logging.ERROR)
                             continue
                        config.log(f"Using model checkpoint for generation: {model_checkpoint_dir}.")
                    else:
                        config.log(f"Calling Gemini API for generation (no local checkpoint needed).")

                    try:
                        generate.generate_batch(
                            checkpoint_model_dir=model_checkpoint_dir, # Will be None for Gemini
                            data_file_path=data_file_path,
                            number_of_problems_per_domain=args.number_of_problems_per_domain
                        )
                        config.log(f"Finished generation for domain: {domain}")
                    except Exception as e:
                         config.log(f"Error during generation for domain {domain}: {e}", level=logging.ERROR, exc_info=True)
                         continue # Continue to the next domain
        except (FileNotFoundError, ValueError) as e:
             config.log(f"Error selecting domains or base directory issue: {e}", level=logging.ERROR)

        config.log("--- Finished All Generation ---")


    else:
        config.log("No action requested (e.g., --train, --generate). Exiting.", level=logging.WARNING)

    config.log("--- Main script execution finished ---")

