# src/learning_to_plan/main.py
from learning_to_plan import train
from learning_to_plan import utils
from learning_to_plan import config
from learning_to_plan import generate
from learning_to_plan import parser
from learning_to_plan.data import process

logger = config.get_logger(__name__)

# --- Main Execution ---
if __name__ == "__main__":
    args = parser.parse_args()
    config.initialize(args)
    args.config_file_path = args.config_file_path \
        or (config.DEFAULT_TRAIN_CONFIG_FILE_PATH if args.train else None) \
        or (config.DEFAULT_GENERATE_CONFIG_FILE_PATH if args.generate else None)
    try:
        cfg = config.get_config(**vars(args))
    except FileNotFoundError as e:
        raise ValueError(f"Could not set up configuration: {e}")

    mapping = {
        "get_tasks_from_raw_data": {
            "fn": utils.get_tasks_from_raw_data,
        },
        "call_paas": {
            "fn": utils.call_paas,
            "args": cfg,
            "run_on_domains": True,
        },
        "train": {
            "fn" : train.run,
            "args" : cfg,
            "run_on_domains": True,
        },
        "generate": {
            "fn": generate.run,
            "args": {
                **cfg,
                "raise_on_error": False
            },
            "run_on_domains": True,
        },
        "validate": {
            "fn": process.validate_plans,
        },
        "compute_metrics": {
            "fn": process.compute_metrics,
        },
    }

    for action, details in mapping.items():
        if getattr(args, action):
            logger.info(f"--- Starting {action.replace('_', ' ').title()} ---")
            if details.get("run_on_domains", False):
                utils.run_on_domains(fn=details["fn"], **details.get("args", {}))
            else:
                details["fn"](**details.get("args", {}))
            logger.info(f"--- Finished {action.replace('_', ' ').title()} ---")

    logger.info("--- Main script execution finished ---")
