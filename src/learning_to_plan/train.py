import os
import datetime
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint

import learning_to_plan.config as config

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_training_procedure(model_checkpoint_dir, data_file_path):
    start_time = datetime.datetime.now()
    model_name = config.training_params('model_name')
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    config.log(f"Training {model_name} -- started {start_time_str}", level=config.logging.INFO, exc_info=True)

    os.makedirs(model_checkpoint_dir, exist_ok=True)
    # Load the dataset from the single JSON file
    dataset = load_dataset("json", data_files=data_file_path)
    
    # Filter into train and validation sets based on the "type" field
    train_dataset = dataset["train"].filter(lambda example: example["type"] == "train")
    validation_dataset = dataset["train"].filter(lambda example: example["type"] == "validation")
    
    # Combine into a dataset dictionary with train and validation splits
    dataset = {
        "train": train_dataset,
        "validation": validation_dataset
    }
    if len(dataset["train"]) == 0 or len(dataset["validation"]) == 0:
        raise ValueError("Train/validation dataset is empty.")
    model, tokenizer = config.load_model_and_tokenizer(checkpoint_dir=model_checkpoint_dir)

    def tokenize_fn(example):
        # Concatenate prompt and plan
        full_prompt = example["prompt"] + example["plan"]  # You may want to add a separator if needed
        return tokenizer(
            full_prompt,
            max_length=config.training_params("max_seq_length"),
            truncation=True,
            padding="max_length",
        )

    # Update the map function to remove both columns
    tokenized_train = dataset["train"].map(tokenize_fn, batched=True, remove_columns=["prompt", "plan"])
    tokenized_val = dataset["validation"].map(tokenize_fn, batched=True, remove_columns=["prompt", "plan"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=model_checkpoint_dir,
        run_name=f"{config.training_params('model_name')}-{os.path.basename(model_checkpoint_dir)}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        report_to=config.training_params("report_to"),
        num_train_epochs=config.training_params("num_train_epochs"),
        per_device_train_batch_size=config.training_params("batch_size"),
        per_device_eval_batch_size=config.training_params("per_device_eval_batch_size"),
        gradient_accumulation_steps=config.training_params("gradient_accumulation_steps"),
        fp16=not config.training_params("bf16"),
        bf16=config.training_params("bf16"),
        learning_rate=config.training_params("learning_rate"),
        lr_scheduler_type=config.training_params("lr_scheduler_type"),
        weight_decay=config.training_params("weight_decay"),
        save_strategy=config.training_params("save_strategy"),
        save_steps=config.training_params("save_steps"),
        save_total_limit=config.training_params("save_total_limit"),
        logging_strategy=config.training_params("logging_strategy"),
        logging_steps=config.training_params("logging_steps"),
        eval_strategy=config.training_params("eval_strategy"),
        optim=config.training_params("optimizer"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        processing_class=tokenizer,
    )

    last_checkpoint = get_last_checkpoint(model_checkpoint_dir)
    if last_checkpoint is None:
        config.log("No checkpoint found. Starting training from scratch.", level=config.logging.INFO, exc_info=True)
    else:
        config.log(f"Resuming training from checkpoint: {last_checkpoint}", level=config.logging.INFO, exc_info=True)

    config.log(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", level=config.logging.INFO, exc_info=True)
    trainer.train(resume_from_checkpoint=last_checkpoint)
    config.log(f"Model saved to {model_checkpoint_dir}", level=config.logging.INFO, exc_info=True)

    end_time = datetime.datetime.now()
    model_name = config.training_params('model_name') # Re-fetch in case it's needed again, or use the one from above if scope allows
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    config.log(f"Training {model_name} -- finished {end_time_str}", level=config.logging.INFO, exc_info=True)
    config.log(f"Total training time: {end_time - start_time}", level=config.logging.INFO, exc_info=True)
    config.log(f"Training {config.training_params('model_name')} -- finished {end_time.strftime('%Y-%m-%d %H:%M:%S')}", level=config.logging.INFO, exc_info=True)
    config.log(f"Total training time: {end_time - start_time}", level=config.logging.INFO, exc_info=True)