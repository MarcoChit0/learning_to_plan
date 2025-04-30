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
from learning_to_plan import validate
from learning_to_plan import metrics

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
        "--load_without_finetuned_checkpoints",
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

def get_selected_domains(args, dir=None, file=None) -> set[str]:
    if not args.domain:
        # Use config's logger/printer
        config.log("Please specify a domain with --domain <domain_name> or 'all'.", level=logging.ERROR)
        raise ValueError("Domain not specified.")

    if dir:
        assert os.path.isdir(dir), f"Directory {dir} does not exist."
        try:
            available_domains = {d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))}
        except OSError as e:
            config.log(f"Error listing domains in {dir}: {e}", level=logging.ERROR, exc_info=True)
            raise e
        assert available_domains and len(available_domains) > 0, f"No domains found in {dir}."
    elif file:
        assert os.path.isfile(file), f"File {file} does not exist."
        tasks = task.get_tasks_from_jsonl(file)
        assert tasks, f"No tasks found in {file}."
        available_domains = {t._domain for t in tasks}
        assert available_domains, f"No domains found in {file}."
    else:
        config.log("No directory or file specified for domain selection.", level=logging.ERROR)
        raise ValueError("No directory or file specified for domain selection.")

    if args.domain.lower() == "all":
        config.log(f"Processing all found domains: {', '.join(available_domains)}")
        return available_domains
    else:
        selected = set(s.strip() for s in args.domain.split(","))
        assert selected.issubset(available_domains), f"Selected domains {selected} are not in available domains {available_domains}."
        selected = selected.intersection(available_domains)
        config.log(f"Processing selected domains: {', '.join(selected)}")
        return selected

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()

    config.initialize(args)

    # --- Action Blocks ---
    if args.call_paas:
        config.log("--- Starting Planning as a Service (PaaS) Calls ---")
        domains = get_selected_domains(args, dir=config.RAW_DIR)
        for domain in domains:
            config.log(f"Processing PaaS for domain: {domain}")
            tasks = task.get_tasks_from_domain_directory(domain, args.number_of_problems_per_domain)
            if not tasks:
                config.log(f"No tasks found for domain {domain}. Skipping.", level=logging.WARNING)
                continue
            config.log(f"Outputting PaaS results to: {config.PROCESSED_DATA_FILE_PATH}")
            asyncio.run(utils.call_paas(tasks))
            config.log(f"Finished PaaS calls for domain: {domain}")
        config.log("--- Finished All PaaS Calls ---")

    elif args.split_dataset:
        config.log("--- Starting Dataset Splitting ---")
        try:
            utils.split_dataset(random_seed=42)
        except Exception as e:
            raise e
        config.log("--- Finished All Dataset Splitting ---")

    elif args.train:
        config.log("--- Starting Model Training ---")

        model_name = config.get_config("model_name")
        if model_name and model_name.lower().startswith("gemini"):
            m = f"Model '{model_name}' is a Gemini model. Training is not supported for Gemini."
            config.log(m, level=logging.ERROR)
            raise ValueError(m) 

        domains = get_selected_domains(args, file=config.PROCESSED_DATA_FILE_PATH)
        for domain in domains:
            config.log(f"Starting training for domain: {domain}")
            train.run_training_procedure(domain)
            config.log(f"Finished training for domain: {domain}")
        config.log("--- Finished All Training ---")

    elif args.generate:
        model_name = config.get_config("model_name")
        assert model_name, "Model name not found in config. Please check your configuration."
        model_checkpoints_base_dir = None
        domais = []
        if model_name.lower().startswith("gemini"):
            domains = get_selected_domains(args, config.PROCESSED_DATA_DIR)
            config.log(f"Using Gemini API for generation. Gemini model: {model_name}")
        else:
            if hasattr(args, "load_without_finetuned_checkpoints") and args.load_without_finetuned_checkpoints:
                domains = get_selected_domains(args, config.PROCESSED_DATA_DIR)
                config.log(f"Using base model {model_name} for generation.")
            else:
                model_checkpoints_base_dir = os.path.join(config.CHECKPOINTS_DIR, model_name)
                domains = get_selected_domains(args, model_checkpoints_base_dir)
                config.log(f"Using fine-tuning checkpoints at {model_checkpoints_base_dir} for generation.")
        assert domains != [], f"No valid domains found for generation. Please check your configuration."
        config.log(f"Domains selected for generation: {', '.join(domains)}")


        for domain in domains:
            hf_model, hf_tokenizer = None, None
            
            config.log(f"Starting generation for domain: {domain}")
            data_file_path = os.path.join(config.PROCESSED_DATA_DIR, domain, config.PROCESSED_DATA_FILE_NAME)
            assert os.path.exists(data_file_path), f"Data file not found: {data_file_path}"
            config.log(f"Using data: {data_file_path}.")

            if model_name.lower().startswith("gemini"):
                config.log(f"Calling Gemini API for generation.")
            else:
                model_checkpoint_dir = os.path.join(model_checkpoints_base_dir, domain) if model_checkpoints_base_dir else None
                if model_checkpoint_dir and not os.path.exists(model_checkpoint_dir):
                    m = f"Model checkpoint {model_checkpoint_dir} does not exist for domain {domain}."
                    config.log(m, level=logging.ERROR)
                    raise FileNotFoundError(m)
                hf_model, hf_tokenizer = config.load_model_and_tokenizer(model_checkpoint_dir)
                assert hf_model is not None, f"Failed to load model from {model_checkpoint_dir}."
                assert hf_tokenizer is not None, f"Failed to load tokenizer from {model_checkpoint_dir}."
                config.log(f"Using model checkpoint for generation: {model_checkpoint_dir}.")

            generate.generate_batch(
                model=hf_model,
                tokenizer=hf_tokenizer,
                data_file_path=data_file_path,
                number_of_problems_per_domain=args.number_of_problems_per_domain
            )
            config.log(f"Finished generation for domain: {domain}")
        config.log("--- Finished All Generation ---")

    elif args.validate:
        config.log("--- Starting Validation ---")
        domains = get_selected_domains(args, config.PROCESSED_DATA_DIR)
        for domain in domains:
            config.log(f"Validating plans for domain: {domain}")
            data_file_path = os.path.join(config.PROCESSED_DATA_DIR, domain, config.PROCESSED_DATA_FILE_NAME)
            assert os.path.exists(data_file_path), f"Data file not found: {data_file_path}"
            validate.validate_plans(data_file_path)
            config.log(f"Finished validation for domain: {domain}")
        config.log("--- Finished All Validation ---")

    elif args.compute_metrics:
        pass

    else:
        config.log("No action requested (e.g., --train, --generate). Exiting.", level=logging.WARNING)

    config.log("--- Main script execution finished ---")