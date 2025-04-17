import datetime
import os
import argparse
import asyncio

from learning_to_plan import build_finetuning_dataset
from learning_to_plan import train
from learning_to_plan import call_paas
from learning_to_plan import task
from learning_to_plan import config
from learning_to_plan import test

def parse_args():
    parser = argparse.ArgumentParser(description="Learning to Plan")
    parser.add_argument(
        "-d", "--domain",
        type=str,
        default="",
        help="List of domains separated by commas."
    )
    parser.add_argument(
        "--call_paas",
        action="store_true",
        help="Whether to call planning as a service to generate plans or not."
    )
    parser.add_argument(
        "-n", "--number_of_problems_per_domain",
        type=int,
        default=None,
        help="Selects the first n problems to call planning as a service for each selected domain, or 'all' for all tasks."
    )
    parser.add_argument(
        "--overwrite_paas_plans",
        action="store_true",
        help="Whether to overwrite the existing plans or not."
    )
    parser.add_argument(
        "--build_finetuning_dataset",
        action="store_true",
        help="Whether to build the finetuning dataset given or not. Requires that planning as a service is called first."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Whether to train or not. Requires that the finetuning dataset is built first."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether to evaluate or not."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Model name to use for training/evaluation."
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=None,
        help="Number of epochs for training."
    )
    parser.add_argument(
        "--run_on_google_colab",
        action="store_true",
        help="Whether to run on google colab or not."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Whether to load the model in 8bit or not."
    )
    parser.add_argument(
        "--problem_size",
        type=str,
        default=None,
        choices=["basic", "long"],
        help="Problem size to call planning as a service for."
    )
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    config.initilize(args)

    def verify_domain():
        if args.domain == "":
            e = "Please specify a domain with --domain <domain_name> or 'all'."
            config.logging.error(e)
            raise ValueError(e)
    def get_selected_domains(dir_path):
        available_domains = os.listdir(dir_path)
        if args.domain == "all":
            return available_domains
        else:
            domains = args.domain.split(",")
            for d in domains:
                if d not in available_domains:
                    raise ValueError(f"Domain {d} not found in {dir_path}.")
            return domains
        
    if args.call_paas:
        verify_domain()
        domains = get_selected_domains(config.RAW_DIR)
        for domain in domains:
            tasks = task.get_tasks_from_domain_directory(domain, args.number_of_problems_per_domain, args.problem_size)
            if not tasks:
                raise ValueError(f"No tasks found in {domain} domain.")
            output_file_path = os.path.join(config.PAAS_PLANS_DIR, domain, config.PAAS_PLAN_FILE_NAME)
            config.logging.info(f"Calling planning as a service for domain {domain} at time {datetime.datetime.now()}.")
            asyncio.run(call_paas.call_paas(tasks, output_file_path, overwrite=args.overwrite_paas_plans, max_retries=args.max_retries))
            config.logging.info(f"Finished calling planning as a service for domain {domain} at time {datetime.datetime.now()}.")
        
    if args.build_finetuning_dataset: 
        config.logging.info("Building finetuning dataset.")
        for domain in os.listdir(config.PAAS_PLANS_DIR):
            build_finetuning_dataset.build_finetuining_dataset(
                os.path.join(config.PAAS_PLANS_DIR, domain, config.PAAS_PLAN_FILE_NAME),
                train_output=os.path.join(config.FINETUNING_DATASET_DIR, domain, config.TRAIN_FILE_NAME),
                validation_output=os.path.join(config.FINETUNING_DATASET_DIR, domain, config.VAL_FILE_NAME),
                test_output=os.path.join(config.FINETUNING_DATASET_DIR, domain, config.TEST_FILE_NAME)
            )
        config.logging.info("Finished building finetuning dataset.")
        
    if args.train:
        verify_domain()
        domains = get_selected_domains(config.FINETUNING_DATASET_DIR)
        config.logging.info(f"Training model {config.MODEL_TRAINING_CONFIG['model_name']} for {config.MODEL_TRAINING_CONFIG['num_train_epochs']} epochs. Starting at {datetime.datetime.now()}.")
        
        for domain in domains:
            train_file = os.path.join(config.FINETUNING_DATASET_DIR, domain, config.TRAIN_FILE_NAME)
            val_file   = os.path.join(config.FINETUNING_DATASET_DIR, domain, config.VAL_FILE_NAME)
            model_checkpoint_dir = os.path.join(config.CHECKPOINTS_DIR, config.MODEL_TRAINING_CONFIG["model_name"], domain)
            config.create_necessary_dirs(model_checkpoint_dir)
            train.run_training_procedure(model_checkpoint_dir, train_file, val_file)
            config.logging.info(f"Finished training model {config.MODEL_TRAINING_CONFIG['model_name']} for domain {domain}. Ending at {datetime.datetime.now()}.")
    
    if args.evaluate:
        model_dir = os.path.join(config.CHECKPOINTS_DIR, config.MODEL_TRAINING_CONFIG["model_name"])
        verify_domain()
        domains = get_selected_domains(model_dir)
        if args.model:
            config.MODEL_TRAINING_CONFIG["model_name"] = args.model
        for domain in domains:
            test_file = os.path.join(config.FINETUNING_DATASET_DIR, domain, config.TEST_FILE_NAME)
            model_checkpoint_dir = os.path.join(model_dir, domain)
            config.logging.info("Evaluating model %s for domain %s", config.MODEL_TRAINING_CONFIG["model_name"], domain)
            test.run_evaluation_procedure(model_checkpoint_dir, test_file)