import os
import datetime
from learning_to_plan import task
from datasets import DatasetDict
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint
import torch # Import torch

import learning_to_plan.config as config

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_training_procedure(model_checkpoint_dir, data_file_path):
    start_time = datetime.datetime.now()
    model_name = config.get_config('model_name')
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    config.log(f"Starting training for {model_name} -- started {start_time_str}", level=config.logging.INFO)

    os.makedirs(model_checkpoint_dir, exist_ok=True)


    if len(dataset["train"]) == 0 or len(dataset["validation"]) == 0:
        raise ValueError("Train/validation dataset split resulted in zero examples.")

    # --- Load Model and Tokenizer ---
    config.log(f"Loading model and tokenizer (checkpoint dir: {model_checkpoint_dir})...", level=config.logging.INFO)
    try:
        model, tokenizer = config.load_model_and_tokenizer(checkpoint_dir=model_checkpoint_dir)
        assert tokenizer is not None, "Tokenizer loading failed."
        assert model is not None, "Model loading failed."
        config.log("Model and tokenizer loaded successfully.", level=config.logging.INFO)
    except Exception as e:
        config.log(f"Fatal error loading model/tokenizer: {e}", level=config.logging.ERROR, exc_info=True)
        raise e # Stop execution if model/tokenizer fails

    # --- Load and Prepare Dataset ---
    config.log(f"Loading dataset from: {data_file_path}", level=config.logging.INFO)
    try:
        assert os.path.exists(data_file_path), f"Data file {data_file_path} does not exist."
        tasks = config.load_tasks(data_file_path)
        assert len(tasks) > 0, f"Dataset {data_file_path} is empty."

        training_tasks = {t for t in tasks if t.type == config.TaskType.TRAIN}
        assert len(training_tasks) > 0, f"No training tasks found in {data_file_path}."
        validation_tasks = {t for t in tasks if t.type == config.TaskType.VALIDATION}
        assert len(validation_tasks) > 0, f"No validation tasks found in {data_file_path}."
        config.log(f"Loaded {len(tasks)} tasks from {data_file_path}.", level=config.logging.INFO)
        config.log(f"Found {len(training_tasks)} training tasks and {len(validation_tasks)} validation tasks.", level=config.logging.INFO)
        
        # Convert tasks to DatasetDict format
        train_dataset = task.convert_tasks_into_dataset(training_tasks, tokenizer_eos=tokenizer.eos_token, with_plan=True)
        validation_dataset = task.convert_tasks_into_dataset(validation_tasks, tokenizer_eos=tokenizer.eos_token, with_plan=True)
        dataset = DatasetDict({
            "train": train_dataset,
            "validation": validation_dataset
        })
        config.log(f"Dataset loaded and converted successfully.", level=config.logging.INFO)

    except Exception as e:
        config.log(f"Error loading or splitting dataset from {data_file_path}: {e}", level=config.logging.ERROR, exc_info=True)
        raise e

    # --- Tokenization ---
    config.log("Starting tokenization...", level=config.logging.INFO)
    def tokenize_fn(examples):
        return tokenizer(
            examples["task"],
            max_length=config.get_config("max_length", 512), # Default max length
            truncation=True
        )

    try:
        config.log("Tokenizing datasets...", level=config.logging.INFO)
        tokenized_train = dataset["train"].map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
        tokenized_val = dataset["validation"].map(tokenize_fn, batched=True, remove_columns=dataset["validation"].column_names)
        config.log("Tokenization complete.", level=config.logging.INFO)
    except Exception as e:
        config.log(f"Error during tokenization: {e}", level=config.logging.ERROR, exc_info=True)
        raise e


    # --- Data Collator ---
    # DataCollatorForLanguageModeling handles dynamic padding by default if tokenizer wasn't called with padding=True/max_length
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    config.log("Data collator initialized.", level=config.logging.INFO)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=model_checkpoint_dir,
        run_name=f"{config.get_config('model_name')}-{os.path.basename(model_checkpoint_dir)}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        report_to=config.get_config("report_to", "none"), # Default to none
        num_train_epochs=config.get_config("num_train_epochs", 3), # Default epochs
        per_device_train_batch_size=config.get_config("batch_size", 1), # Default batch size
        per_device_eval_batch_size=config.get_config("eval_batch_size", 1), # Default eval batch size
        gradient_accumulation_steps=config.get_config("gradient_accumulation_steps", 1), # Default grad accum
        # --- Precision ---
        fp16=not config.get_config("bf16", False) and torch.cuda.is_available(), # Use fp16 if not bf16 and cuda available
        bf16=config.get_config("bf16", False) and torch.cuda.is_available() and torch.cuda.is_bf16_supported(), # Use bf16 if configured and supported
        # --- Optimizer ---
        learning_rate=config.get_config("learning_rate", 5e-5), # Default LR
        lr_scheduler_type=config.get_config("lr_scheduler_type", "cosine"), # Default scheduler
        weight_decay=config.get_config("weight_decay", 0.01), # Default weight decay
        # optim=config.get_config("optimizer", "adamw_torch"), # Default optimizer (adamw_torch is often good)
         # Use adamw_bnb_8bit if 8bit loading is enabled
        optim="adamw_bnb_8bit" if config.get_config("load_in_8bit") else config.get_config("optimizer", "adamw_torch"),
        # --- Saving & Logging ---
        save_strategy=config.get_config("save_strategy", "steps"),
        save_steps=config.get_config("save_steps", 500), # Default save steps
        save_total_limit=config.get_config("save_total_limit", 1), # Default save limit
        logging_strategy=config.get_config("logging_strategy", "steps"),
        logging_steps=config.get_config("logging_steps", 100), # Default log steps
        # --- Evaluation ---
        eval_strategy=config.get_config("eval_strategy", "steps"), # Evaluate periodically by steps
        eval_steps=config.get_config("eval_steps", 500), # Default eval steps (match save_steps?)
        # --- Other ---
        gradient_checkpointing=config.get_config("gradient_checkpointing", True), # Enable gradient checkpointing to save memory
        # deepspeed=config.get_config("deepspeed_config", None), # Add deepspeed if configured
    )
    config.log(f"Training Arguments: {training_args.to_dict()}", level=config.logging.DEBUG, do_print=False)


    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        tokenizer=tokenizer, # Pass tokenizer directly
    )

    # --- Check for Checkpoint and Train ---
    last_checkpoint = get_last_checkpoint(model_checkpoint_dir)
    resume_from_checkpoint = None
    if last_checkpoint is None:
        config.log("No checkpoint found. Starting training from scratch.", level=config.logging.INFO)
    else:
        config.log(f"Resuming training from checkpoint: {last_checkpoint}", level=config.logging.INFO)
        resume_from_checkpoint = last_checkpoint

    config.log(f"Starting trainer.train() at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", level=config.logging.INFO)
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        config.log("Training finished successfully.", level=config.logging.INFO)
    except Exception as e:
         config.log(f"Error during trainer.train(): {e}", level=config.logging.ERROR, exc_info=True)
         # Optionally try saving state even on error
         try:
             trainer.save_state()
             config.log("Trainer state saved after error.", level=config.logging.WARNING)
         except Exception as se:
             config.log(f"Failed to save trainer state after error: {se}", level=config.logging.ERROR)
         raise e # Re-raise the training error

    # --- Save Final Model ---
    try:
        trainer.save_model(model_checkpoint_dir)
        config.log(f"Final model saved to {model_checkpoint_dir}", level=config.logging.INFO)
        # trainer.save_state() # Save final trainer state
        # config.log(f"Final trainer state saved to {model_checkpoint_dir}", level=config.logging.INFO)
    except Exception as e:
        config.log(f"Error saving final model/state: {e}", level=config.logging.ERROR, exc_info=True)


    end_time = datetime.datetime.now()
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    config.log(f"Training {model_name} -- finished {end_time_str}", level=config.logging.INFO)
    config.log(f"Total training time: {end_time - start_time}", level=config.logging.INFO)

    # --- Clean up GPU memory ---
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        config.log("Cleaned GPU memory after training.", level=config.logging.INFO)

