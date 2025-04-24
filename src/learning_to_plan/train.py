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
    model_name = config.get_config('model_name')
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

    def tokenize_fn(batch):
        # Concatenate prompt and plan for each example in the batch
        full_prompts = [p + pl for p, pl in zip(batch["prompt"], batch["plan"])] # You may want to add a separator if needed
        return tokenizer(
            full_prompts,
            max_length=config.get_config("max_seq_length"),
            truncation=True,
            padding="max_length",
        )

    # Update the map function to remove both columns
    tokenized_train = dataset["train"].map(tokenize_fn, batched=True, remove_columns=["prompt", "plan"])
    tokenized_val = dataset["validation"].map(tokenize_fn, batched=True, remove_columns=["prompt", "plan"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=model_checkpoint_dir,
        run_name=f"{config.get_config('model_name')}-{os.path.basename(model_checkpoint_dir)}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        report_to=config.get_config("report_to"),
        num_train_epochs=config.get_config("num_train_epochs"),
        per_device_train_batch_size=config.get_config("batch_size"),
        per_device_eval_batch_size=config.get_config("per_device_eval_batch_size"),
        gradient_accumulation_steps=config.get_config("gradient_accumulation_steps"),
        fp16=not config.get_config("bf16"),
        bf16=config.get_config("bf16"),
        learning_rate=config.get_config("learning_rate"),
        lr_scheduler_type=config.get_config("lr_scheduler_type"),
        weight_decay=config.get_config("weight_decay"),
        save_strategy=config.get_config("save_strategy"),
        save_steps=config.get_config("save_steps"),
        save_total_limit=config.get_config("save_total_limit"),
        logging_strategy=config.get_config("logging_strategy"),
        logging_steps=config.get_config("logging_steps"),
        eval_strategy=config.get_config("eval_strategy"),
        optim=config.get_config("optimizer"),
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
    model_name = config.get_config('model_name') # Re-fetch in case it's needed again, or use the one from above if scope allows
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    config.log(f"Training {model_name} -- finished {end_time_str}", level=config.logging.INFO, exc_info=True)
    config.log(f"Total training time: {end_time - start_time}", level=config.logging.INFO, exc_info=True)
    config.log(f"Training {config.get_config('model_name')} -- finished {end_time.strftime('%Y-%m-%d %H:%M:%S')}", level=config.logging.INFO, exc_info=True)
    config.log(f"Total training time: {end_time - start_time}", level=config.logging.INFO, exc_info=True)