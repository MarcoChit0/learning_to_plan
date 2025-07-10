# main.py
import os
import asyncio

# Import project modules
from learning_to_plan import train
from learning_to_plan import utils
from learning_to_plan import config
from learning_to_plan import generate
from learning_to_plan import parser
from learning_to_plan.data import task

logger = config.get_logger(__name__)

def get_selected_domains(args) -> set[str]:
    if not args.domain:
        logger.error("Please specify a domain with --domain <domain_name> or 'all'.")
        raise ValueError("Domain not specified.")

    tasks = task.task_database.get()
    available_domains = {t.domain for t in tasks}
    logger.info(f"Available domains: {', '.join(available_domains)}")

    if args.domain.lower() == "all":
        logger.info("Processing all available domains.")
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
    if args.get_tasks_from_raw_data:
        logger.info("--- Starting Task Creation from Raw Data ---")
        utils.get_tasks_from_raw_data()
        logger.info("--- Finished Task Creation from Raw Data ---")
    elif args.call_paas:
        logger.info("--- Starting Planning as a Service (PaaS) Calls ---")
        domains = get_selected_domains(args)
        for domain in domains:
            logger.info(f"Processing PaaS for domain: {domain}")
            asyncio.run(utils.call_paas(domain=domain))
            logger.info(f"Finished PaaS calls for domain: {domain}")
        logger.info("--- Finished All PaaS Calls ---")

    elif args.train:
        logger.info("--- Starting Model Training ---")
        config_file_path = args.config_file_path or os.path.join(config.CONFIGS_DIR, config.DEFAULT_TRAIN_CONFIG)
        train_kwargs = config.get_config(config_file_path=config_file_path, args=args)
        assert train_kwargs["model_name"], "Model name not found in config. Please check your configuration."
        domains = get_selected_domains(args=args)
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
        domains = get_selected_domains(args=args)
        for domain in domains:
            logger.info(f"Starting generation for domain: {domain}")
            checkpoint_dir = None if args.dont_use_checkpoint else config.get_checkpoint_dir(domain, generate_kwargs["model_name"])
            generate.generate_batch(
                domain=domain, 
                number_of_instances=args.number_of_instances, 
                task_type=args.task_type,
                ## --- generation kwargs ---
                checkpoint_dir=checkpoint_dir, 
                overwrite_generated_plans=args.overwrite_generated_plans, 
                **generate_kwargs)
            logger.info(f"Finished generation for domain: {domain}")
        logger.info("--- Finished All Generation ---")
    
    # elif args.validate:
    #     logger.info("--- Starting Plan Validation ---")
    #     utils.apply_function_to_all_models(
    #         function=processing_data.validate_plans
    #     )
    #     logger.info("--- Finished All Validation ---")
    
    # elif args.compute_metrics:
    #     logger.info("--- Starting Metrics Computation ---")
    #     utils.apply_function_to_all_models(
    #         function=processing_data.compute_metrics
    #     )
    #     logger.info("--- Finished All Metrics Computation ---")
    elif args.landmarks_generation:
        logger.info("--- Starting Landmark Graph Generation ---")
        utils.get_landmark_graph()
        logger.info("--- Finished All Landmark Graph Generation ---")
    else:
        logger.warning("No action requested (e.g., --train, --generate). Exiting.")

    logger.info("--- Main script execution finished ---")
