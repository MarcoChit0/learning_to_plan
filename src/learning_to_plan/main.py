import datetime
from learning_to_plan.build_finetuning_dataset import *
from learning_to_plan.train import *
from learning_to_plan.call_paas import *
from learning_to_plan.task import *
from learning_to_plan.config import *
import os

# create a parser for command line arguments
import argparse

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
        help="Whether to call planing as a service to generate plans or not."
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
        "-m","--model",
        type=str,
        default=None,
        help="Model name to use for training."
    )
    parser.add_argument(
        "-e","--epochs",
        type=int,
        default=None,
        help="Number of epochs for training."
    )
    return parser.parse_args()

if __name__ == "__main__":
    create_data_dirs()
    args = parse_args()

    if args.call_paas:
        if args.domain == "":
            print("Please specify a domain with --domain <domain_name> or 'all'.")
            exit(0)
        available_domains = os.listdir(RAW_DIR)
        print(available_domains)
        if args.domain == "all": 
            domains = available_domains
        else:
            domains = args.domain.split(",")
            for d in domains:
                if d not in available_domains:
                    raise ValueError(f"Domain {d} not found in {RAW_DIR}.")
        for domain in domains:
            tasks = get_tasks_from_domain_directory(domain, args.number_of_problems_per_domain)
            output_file_path = os.path.join(PAAS_PLANS_DIR, domain, PAAS_PLAN_FILE_NAME)
            print(f"Calling planning as a service for domain {domain} at time {datetime.datetime.now()}.")
            asyncio.run(call_paas(tasks, output_file_path, overwrite=args.overwrite_paas_plans))
            print(f"Finished calling planning as a service for domain {domain} at time {datetime.datetime.now()}.")
        
    if args.build_finetuning_dataset: 
        for domain in os.listdir(PAAS_PLANS_DIR):
            build_finetuining_dataset(
                os.path.join(PAAS_PLANS_DIR, domain, PAAS_PLAN_FILE_NAME),
                train_output=os.path.join(FINETUNING_DATASET_DIR, domain, TRAIN_FILE_NAME),
                test_output=os.path.join(FINETUNING_DATASET_DIR, domain, VAL_FILE_NAME),
            )
    
    if args.train:
        if args.domain == "":
            print("Please specify a domain with --domain <domain_name> or 'all'.")
            exit(0)
        available_domains = os.listdir(FINETUNING_DATASET_DIR)
        if args.domain == "all": 
            domains = available_domains
        else:
            domains = args.domain.split(",")
            for d in domains:
                if d not in available_domains:
                    raise ValueError(f"Domain {d} not found in {FINETUNING_DATASET_DIR}.")
        
        # TODO: add two functionalities to 'run_training_procedure' 
        # 1. add the possibility to pass model checkpoints
        # 2. add the possibility to train accross multiple domains
        for domain in domains:
            train_file = os.path.join(FINETUNING_DATASET_DIR, domain, TRAIN_FILE_NAME)
            val_file   = os.path.join(FINETUNING_DATASET_DIR, domain, VAL_FILE_NAME)
            domain_output_dir = os.path.join(CHECKPOINTS_DIR, domain)
            create_necessary_dirs(domain_output_dir)
            run_training_procedure(domain_output_dir, train_file, val_file)