import os
import logging

MODEL_TRAINING_CONFIG = {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",

    "max_seq_length": 2048,
    "max_new_tokens": 2048,
    "report_to": "none",

    # Training
    "batch_size": 1, # 1 - 4,
    "gradient_accumulation_steps": 1, # 1 - 2
    
    # Evaluation
    "eval_strategy": "epoch",
    "eval_batch_size": 1, # 8, 16, 32

    # Optimization
    "learning_rate": 1.0e-5,
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 2,
    "weight_decay": 0.02,
    "optimizer": "adamw_8bit",

    # Salvamento e logging
    "logging_steps": 400,
    "logging_strategy": "steps",
    "save_steps": 800,
    "save_strategy": "steps",
    "save_total_limit": 1,

    # Outros
    "bf16": True,
    "load_in_8bit": False,
    "lora_r": 8,
    # "deepspeed_config": "deepspeed_zero3.json"
}

# Inside each domain directory
BASIC_INSTANCES = "generated_basic"
LONG_INSTANCES = "generated_basic_longer_plan_len"

# Data folders
GOOGLE_COLAB_DATA_DIR = "../drive/MyDrive/projects/learning_to_plan/data/"
DATA_DIR = "data/"
RAW_DIR = None
PAAS_PLANS_DIR = None
FINETUNING_DATASET_DIR = None
CHECKPOINTS_DIR = None

def initilize(args):
    global DATA_DIR, GOOGLE_COLAB_DATA_DIR, RAW_DIR, PAAS_PLANS_DIR, FINETUNING_DATASET_DIR, CHECKPOINTS_DIR
    # initialize parameters
    if args.load_in_8bit:
        MODEL_TRAINING_CONFIG["load_in_8bit"] = True
        MODEL_TRAINING_CONFIG["bf16"] = False
    
    if args.model:
        MODEL_TRAINING_CONFIG["model_name"] = args.model
    
    if args.epochs:
        MODEL_TRAINING_CONFIG["num_train_epochs"] = args.epochs

    if args.run_on_google_colab: 
        DATA_DIR = GOOGLE_COLAB_DATA_DIR

    # initialize directories
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    PAAS_PLANS_DIR = os.path.join(DATA_DIR, "paas_plans")
    FINETUNING_DATASET_DIR = os.path.join(DATA_DIR, "finetuning_dataset")
    CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
    for dir in [DATA_DIR, RAW_DIR, PAAS_PLANS_DIR, FINETUNING_DATASET_DIR, CHECKPOINTS_DIR]:
        os.makedirs(dir, exist_ok=True)
    
    # initialize logging
    logging.basicConfig(
        filename=os.path.join(DATA_DIR, LOGGING_FILE_NAME),
        filemode='a',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO
    )

def create_necessary_dirs(file_path):
    dirs = file_path.split("/")[:-1]
    path = "/".join(dirs)
    os.makedirs(path, exist_ok=True)

# File names
PAAS_PLAN_FILE_NAME = "paas_plans.csv"
TRAIN_FILE_NAME = "train.jsonl"
VAL_FILE_NAME = "validation.jsonl" # used to compute loss function during training
TEST_FILE_NAME = "test.jsonl" # ood data to check generalization at each epoch
DOMAIN_FILE_NAME = "generated_domain.pddl"
LOGGING_FILE_NAME = "logs.log"
TRAINING_PARAMETERS_FILE_NAME = "training_parameters.json"
TEST_METRICS_FILE_NAME = "metrics.csv"
TEST_DATA_FILE_NAME = "test_{index}.jsonl"