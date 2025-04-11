import os

MODEL_TRAINING_CONFIG = {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",

    "max_seq_length": 4096,
    "max_new_tokens": 4096,

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
    "logging_steps": 500,
    "save_steps": 1000,

    # Outros
    "bf16": True,
    "deepspeed_config": "deepspeed_zero3.json"
}


# Data directories
DATA_DIR = "data/"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PAAS_PLANS_DIR = os.path.join(DATA_DIR, "paas_plans")
FINETUNING_DATASET_DIR = os.path.join(DATA_DIR, "finetuning_dataset")
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")

# Inside each domain directory
INSTANCES_SUBDIRECTORY = "generated_basic"


def create_data_dirs():
    for dir in [DATA_DIR, RAW_DIR, PAAS_PLANS_DIR, FINETUNING_DATASET_DIR, CHECKPOINTS_DIR]:
        os.makedirs(dir, exist_ok=True)

def create_necessary_dirs(file_path):
    dirs = file_path.split("/")[:-1]
    path = "/".join(dirs)
    os.makedirs(path, exist_ok=True)

# File names
PAAS_PLAN_FILE_NAME = "paas_plans.csv"
VAL_FILE_NAME = "test.jsonl"
TRAIN_FILE_NAME = "train.jsonl"
DOMAIN_FILE_NAME = "generated_domain.pddl"