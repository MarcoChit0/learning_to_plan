import os
import logging

MODEL_TRAINING_CONFIG = {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",

    "max_seq_length": 2048,
    "max_new_tokens": 2048,

    # Batch
    "batch_size": 1, # 1 - 4,
    "eval_batch_size": 1, # 8, 16, 32
    "gradient_accumulation_steps": 1, # 1 - 2

    # Optimization
    "learning_rate": 1.0e-5,
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 2,
    "weight_decay": 0.02,
    "optimizer": "adamw_8bit",

    # Salvamento e logging
    "logging_steps": 100,
    "logging_strategy": "steps",
    "save_steps": 200,
    "save_strategy": "steps",

    # Outros
    "bf16": True,
    "deepspeed_config": "deepspeed_zero3.json"
}

# Inside each domain directory
INSTANCES_SUBDIRECTORY = "generated_basic"


# Data folders
GOOGLE_COLAB_DATA_DIR = "../drive/MyDrive/projects/learning_to_plan/data/"
DATA_DIR = "data/"
RAW_DIR = None
PAAS_PLANS_DIR = None
CHECKPOINTS_DIR = None
FINETUNING_DATASET_DIR = None

def initilize(run_on_google_colab=False):
    # initialize directories
    global DATA_DIR, GOOGLE_COLAB_DATA_DIR, RAW_DIR, PAAS_PLANS_DIR, FINETUNING_DATASET_DIR, CHECKPOINTS_DIR
    if run_on_google_colab: 
        DATA_DIR = GOOGLE_COLAB_DATA_DIR
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    PAAS_PLANS_DIR = os.path.join(DATA_DIR, "paas_plans")
    FINETUNING_DATASET_DIR = os.path.join(DATA_DIR, "finetuning_dataset")
    CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
    for dir in [DATA_DIR, RAW_DIR, PAAS_PLANS_DIR, FINETUNING_DATASET_DIR, CHECKPOINTS_DIR]:
        print(dir)
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
VAL_FILE_NAME = "test.jsonl"
TRAIN_FILE_NAME = "train.jsonl"
TEST_FILE_NAME = "test.jsonl"
DOMAIN_FILE_NAME = "generated_domain.pddl"
LOGGING_FILE_NAME = "logs.log"
TRAINING_HISTORY_FILE_NAME = "training_data.csv"