# main.py
import os
import asyncio

# Import project modules
from learning_to_plan import train
from learning_to_plan import utils
from learning_to_plan import database
from learning_to_plan import config
from learning_to_plan import generate
from learning_to_plan import processing_data
from learning_to_plan.models import base
from learning_to_plan import parser

logger = config.get_logger(__name__)

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
        tasks = database.get_task_database()
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
    args = parser.parse_args()
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
        utils.split_dataset()
        logger.info("--- Finished All Dataset Splitting ---")

    elif args.train:
        logger.info("--- Starting Model Training ---")
        config_file_path = args.config_file_path or os.path.join(config.CONFIGS_DIR, config.DEFAULT_TRAIN_CONFIG)
        train_kwargs = config.get_config(config_file_path=config_file_path, args=args)
        assert train_kwargs["model_name"], "Model name not found in config. Please check your configuration."
        domains = get_selected_domains(args=args, is_file=True)
        for domain in domains:
            logger.info(f"Starting training for domain: {domain}")
            train.run_training_procedure(domain=domain, **train_kwargs)
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
            checkpoint_dir = None if args.dont_use_checkpoint else config.get_checkpoint_dir(domain, generate_kwargs["model_name"])
            generate.generate_batch(
                domain=domain, 
                number_of_instances=args.number_of_instances, 
                ## --- generation kwargs ---
                checkpoint_dir=checkpoint_dir, 
                overwrite_generated_plans=args.overwrite_generated_plans, 
                **generate_kwargs)
            logger.info(f"Finished generation for domain: {domain}")
        logger.info("--- Finished All Generation ---")
    
    elif args.validate:
        logger.info("--- Starting Plan Validation ---")
        utils.apply_function_to_all_models(
            function=processing_data.validate_plans
        )
        logger.info("--- Finished All Validation ---")
    
    elif args.compute_metrics:
        logger.info("--- Starting Metrics Computation ---")
        utils.apply_function_to_all_models(
            function=processing_data.compute_metrics
        )
        logger.info("--- Finished All Metrics Computation ---")
    elif args.clear_model_dir:
        if args.model_name:
            try:
                base = base.get_model(model_name=args.model_name)
                base.clear_model_dir()
            except Exception as e:
                raise "Error clearing model directory: {e}"
        else:
            def clear_model_dir_helper(model: base.Model, **kwargs):
                model.clear_model_dir()
            utils.apply_function_to_all_models(
                function=clear_model_dir_helper
            )
    elif args.landmarks_generation:
        logger.info("--- Starting Landmark Graph Generation ---")
        utils.get_landmark_graph()
        logger.info("--- Finished All Landmark Graph Generation ---")
    else:
        logger.warning("No action requested (e.g., --train, --generate). Exiting.")

    logger.info("--- Main script execution finished ---")
