# main.py
import os
import argparse
import asyncio
import logging

# Import project modules
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
    # --- Generate Specific ---
    parser.add_argument(
        "--load_with_finetuned_checkpoints",
        action="store_true",
        help="Load the model with fine-tuned checkpoints."
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
    if not args.domain:
        # Use config's logger/printer
        config.log("Please specify a domain with --domain <domain_name> or 'all'.", level=logging.ERROR)
        raise ValueError("Domain not specified.")

    if not os.path.isdir(base_dir):
        m = f"Base directory for domains not found: {base_dir}"
        config.log(m, level=logging.ERROR)
        raise FileNotFoundError(m)
    try:
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
        selected = args.domain.split(",")
        for d in selected:
            if d not in available_domains:
                m = f"Domain '{d}' not found in {base_dir}. Available domains: {', '.join(available_domains)}"
                config.log(m, level=logging.ERROR)
                raise ValueError(m)
        config.log(f"Processing selected domains: {', '.join(selected)}")
        return selected

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()

    config.initialize(args)

    # --- Action Blocks ---
    if args.call_paas:
        config.log("--- Starting Planning as a Service (PaaS) Calls ---")
        domains = get_selected_domains(args, config.RAW_DIR)
        for domain in domains:
            config.log(f"Processing PaaS for domain: {domain}")
            tasks = task.get_tasks_from_domain_directory(domain, args.number_of_problems_per_domain)
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

        model_name_from_config = config.get_config("model_name")
        if model_name_from_config and model_name_from_config.lower().startswith("gemini"):
            m = f"Model '{model_name_from_config}' is a Gemini model. Training is not supported for Gemini."
            config.log(m, level=logging.ERROR)
            raise ValueError(m) 

        domains = get_selected_domains(args, config.PROCESSED_DATA_DIR)
        for domain in domains:
            config.log(f"Starting training for domain: {domain}")
            data_file_path = os.path.join(config.PROCESSED_DATA_DIR, domain, config.PROCESSED_DATA_FILE_NAME)
            assert os.path.exists(data_file_path), f"Data file not found: {data_file_path}"

            # Construct checkpoint dir using the potentially overridden model name
            current_model_name = config.get_config("model_name") # Get final model name after potential override
            model_checkpoint_dir = os.path.join(config.CHECKPOINTS_DIR, current_model_name, domain)
            config.create_necessary_dirs(model_checkpoint_dir) # Use helper from config
            config.log(f"Checkpoints will be saved to: {model_checkpoint_dir}")

            train.run_training_procedure(model_checkpoint_dir, data_file_path)
            config.log(f"Finished training for domain: {domain}")
        config.log("--- Finished All Training ---")

    elif args.generate:
        model_name_from_config = config.get_config("model_name")
        assert model_name_from_config, "Model name not found in config. Please check your configuration."

        if model_name_from_config.lower().startswith("gemini"):
            config.log(f"Model '{model_name_from_config}' is a Gemini model. Calling API for generation.")

            domains = get_selected_domains(args, config.PROCESSED_DATA_DIR)
            assert domains != [], f"No valid domains found for Gemini model in the {config.PROCESSED_DATA_DIR} dir. Please check your configuration."

        else:
            model_checkpoints_base_dir = os.path.join(config.CHECKPOINTS_DIR, model_name_from_config)
            using_finetuning_checkpoint =  True if not args.load_with_finetuned_checkpoint else False
            if using_finetuning_checkpoint:
                config.log(f"Using fine-tuning checkpoints for model '{model_name_from_config}'.")
                domains = get_selected_domains(args, model_checkpoints_base_dir)
                if not domains:
                    config.log(f"Could not find or use fine-tuned checkpoints for model checkpoint dir '{model_checkpoints_base_dir}'"
                            f"Attempting fallback to base model '{model_name_from_config}' using processed data.",
                            level=logging.WARNING)
                    try:
                        domains = get_selected_domains(args, config.PROCESSED_DATA_DIR)
                        if not domains:
                            # If no domains are found even in the processed data directory, cannot proceed.
                            m = (f"Fallback failed: No domains found in processed data dir '{config.PROCESSED_DATA_DIR}' "
                                f"matching --domain '{args.domain}'. Cannot proceed with generation.")
                            config.log(m, level=logging.ERROR)
                            raise ValueError(m)
                        else:
                            config.log(f"Found domains in processed data: {', '.join(domains)}. Proceeding with base model.")
                            using_finetuning_checkpoint = True
                    except Exception as fallback_e:
                        # Catch errors during fallback domain search or model loading
                        config.log(f"Error during fallback attempt: {fallback_e}", level=logging.ERROR)
                        raise fallback_e # Re-raise the error encountered during fallback
            else:
                config.log(f"Not using fine-tuning checkpoints. Attempting to load from base model '{model_name_from_config}'.")
                domains = get_selected_domains(args, config.PROCESSED_DATA_DIR)
                if not domains:
                    m = f"No valid domains found in {config.PROCESSED_DATA_DIR} for generation. Please check your configuration."
                    config.log(m, level=logging.ERROR)
                    raise ValueError(m)

        for domain in domains:
            hf_model, hf_tokenizer = None, None
            
            config.log(f"Starting generation for domain: {domain}")
            data_file_path = os.path.join(config.PROCESSED_DATA_DIR, domain, config.PROCESSED_DATA_FILE_NAME)
            assert os.path.exists(data_file_path), f"Data file not found: {data_file_path}"
            config.log(f"Using data: {data_file_path}.")


            if model_name_from_config.lower().startswith("gemini"):
                config.log(f"Calling Gemini API for generation.")
            else:
                if using_finetuning_checkpoint:
                    model_checkpoint_dir = os.path.join(model_checkpoints_base_dir, domain)
                    assert os.path.exists(model_checkpoint_dir), f"Model checkpoint directory not found: {model_checkpoint_dir}"
                    config.log(f"Loading model from fine-tuning checkpoint: {model_checkpoint_dir}.")
                else:
                    model_checkpoint_dir = None
                    config.log(f"Loading model from base model directory: {model_checkpoints_base_dir}.")
                hf_model, hf_tokenizer = config.load_model_and_tokenizer(model_checkpoint_dir)
                assert hf_model is not None, f"Failed to load model from {model_checkpoint_dir}."
                assert hf_tokenizer is not None, f"Failed to load tokenizer from {model_checkpoint_dir}."
                config.log(f"Using model checkpoint for generation: {model_checkpoint_dir}.")

            generate.generate_batch(
                hf_model=hf_model,
                hf_tokenizer=hf_tokenizer,
                data_file_path=data_file_path,
                number_of_problems_per_domain=args.number_of_problems_per_domain
            )
            config.log(f"Finished generation for domain: {domain}")
        config.log("--- Finished All Generation ---")

    else:
        config.log("No action requested (e.g., --train, --generate). Exiting.", level=logging.WARNING)

    config.log("--- Main script execution finished ---")