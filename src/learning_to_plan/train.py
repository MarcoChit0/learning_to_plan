import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
import learning_to_plan.config as config
from dotenv import load_dotenv
import numpy as np
import shutil
import datetime
import json

# Disable tokenizer parallelism to avoid warnings after forking.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def save_model_params(output_dir):
    """Save current training parameters to a JSON file."""
    params = {k: v for k, v in config.MODEL_TRAINING_CONFIG.items()}
    params_file = os.path.join(output_dir, "training_params.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    return params

def params_changed(output_dir):
    """Return True if the training parameters have changed from a previous run."""
    params_file = os.path.join(output_dir, "training_params.json")
    if not os.path.exists(params_file):
        return True  # No previous parameters, so consider them changed.
    
    with open(params_file, 'r') as f:
        previous_params = json.load(f)
    
    # We ignore the number of epochs to allow continuation.
    current_params = {k: v for k, v in config.MODEL_TRAINING_CONFIG.items() if k != "num_train_epochs"}
    return current_params != previous_params

def run_training_procedure(output_dir, train_file, val_file, test_file, overwrite=False):
    config.logging.info("Starting training procedure for model %s at %s",
                        config.MODEL_TRAINING_CONFIG["model_name"],
                        datetime.datetime.now())

    # Check if a model checkpoint exists
    model_exists = os.path.exists(os.path.join(output_dir, "pytorch_model.bin"))
    if model_exists and params_changed(output_dir):
        config.logging.info("Training parameters have changed from previous run. Forcing overwrite.")
        overwrite = True

    if model_exists:
        if overwrite:
            config.logging.info("Overwriting existing model in: %s", output_dir)
            model_source = config.MODEL_TRAINING_CONFIG["model_name"]
        else:
            config.logging.info("Resuming training from existing checkpoint at: %s", output_dir)
            model_source = output_dir
    else:
        model_source = config.MODEL_TRAINING_CONFIG["model_name"]
        config.logging.info("Starting new training, output to: %s", output_dir)

    # Save current training parameters.
    save_model_params(output_dir)

    # For training only, we load the training split. (Ignore validation/test.)
    dataset = load_dataset("json", data_files={"train": train_file})
    if len(dataset["train"]) == 0:
        e = "Train dataset is empty."
        config.logging.error(e)
        raise ValueError(e)

    load_dotenv()
    autentication_token = os.getenv("HUGGINGFACE_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        use_auth_token=autentication_token
    )

    # Load model: if a checkpoint exists and parameters are unchanged, resume training.
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.MODEL_TRAINING_CONFIG["bf16"] else torch.float16,
        use_auth_token=autentication_token
    )

    def tokenize_fn(example):
        return tokenizer(
            example["prompt"],
            max_length=config.MODEL_TRAINING_CONFIG["max_seq_length"],
            truncation=True,
            padding="max_length"
        )

    tokenized = dataset["train"].map(tokenize_fn, batched=True, remove_columns=["prompt"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=f"{config.MODEL_TRAINING_CONFIG['model_name']}-{os.path.basename(output_dir)}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        num_train_epochs=config.MODEL_TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=config.MODEL_TRAINING_CONFIG["batch_size"],
        gradient_accumulation_steps=config.MODEL_TRAINING_CONFIG["gradient_accumulation_steps"],
        fp16=not config.MODEL_TRAINING_CONFIG["bf16"],
        bf16=config.MODEL_TRAINING_CONFIG["bf16"],
        learning_rate=config.MODEL_TRAINING_CONFIG["learning_rate"],
        lr_scheduler_type=config.MODEL_TRAINING_CONFIG["lr_scheduler_type"],
        weight_decay=config.MODEL_TRAINING_CONFIG["weight_decay"],
        save_strategy=config.MODEL_TRAINING_CONFIG["save_strategy"],
        save_steps=config.MODEL_TRAINING_CONFIG["save_steps"],
        logging_strategy=config.MODEL_TRAINING_CONFIG["logging_strategy"],
        logging_steps=config.MODEL_TRAINING_CONFIG["logging_steps"],
        optim=config.MODEL_TRAINING_CONFIG["optimizer"],
        # Do not define an eval strategy since we're not running evaluation.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        tokenizer=tokenizer
    )

    # Check for a checkpoint to resume from
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            config.logging.info("Resuming training from checkpoint: %s", last_checkpoint)

    config.logging.info("Starting training at %s", datetime.datetime.now())
    trainer.train(resume_from_checkpoint=last_checkpoint)
    config.logging.info("Training completed at %s", datetime.datetime.now())

    trainer.save_model(output_dir)
    config.logging.info("Model saved to %s", output_dir)