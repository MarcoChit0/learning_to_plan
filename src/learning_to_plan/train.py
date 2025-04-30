import os
import datetime
import datasets
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, DatasetDict
import torch # Import torch

import learning_to_plan.config as config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_training_procedure(domain):
    start_time = datetime.datetime.now()
    model_name = config.get_config('model_name')
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    config.log(f"Starting training for {model_name} -- started {start_time_str}", level=config.logging.INFO)

    model_checkpoint_dir = os.path.join(config.CHECKPOINTS_DIR, domain, model_name)
    config.create_necessary_dirs(model_checkpoint_dir) 
    config.log(f"Checkpoints will be saved to: {model_checkpoint_dir}")

    # --- Load Model and Tokenizer ---
    config.log(f"Loading model and tokenizer (checkpoint dir: {model_checkpoint_dir})...", level=config.logging.INFO)

    model, tokenizer = config.load_model_and_tokenizer(checkpoint_dir=model_checkpoint_dir)
    assert tokenizer is not None, "Tokenizer loading failed."
    assert model is not None, "Model loading failed."
    config.log("Model and tokenizer loaded successfully.", level=config.logging.INFO)

    
    # --- Load and Prepare Dataset ---
    config.log(f"Loading dataset from: {config.PROCESSED_DATA_FILE_PATH}", level=config.logging.INFO)
    try:
        from learning_to_plan import task
        assert os.path.exists(config.PROCESSED_DATA_FILE_PATH), f"Data file {config.PROCESSED_DATA_FILE_PATH} does not exist."
        
        # Load tasks from JSONL file
        tasks : set[task.Task] = task.get_tasks_from_jsonl(config.PROCESSED_DATA_FILE_PATH)
        assert len(tasks) > 0, f"No tasks found in {config.PROCESSED_DATA_FILE_PATH}."
        tasks = {t for t in tasks if t._domain == domain}
        assert len(tasks) > 0, f"No tasks found in {config.PROCESSED_DATA_FILE_PATH} for domain {domain}."
        train_tasks : set[task.Task]  = {t for t in tasks if t._type == task.Task.TaskType.TRAIN}
        assert len(train_tasks) > 0, "No training tasks found."
        validation_tasks : set[task.Task]  = {t for t in tasks if t._type == task.Task.TaskType.VALIDATION}
        assert len(validation_tasks) > 0, "No validation tasks found."
        
        # Make the prompts that will be used for training and validation
        eos_token = tokenizer.eos_token if tokenizer.eos_token else ""
        training_prompts : list[str]  = [t.get_prompt(eos_token=eos_token, with_plan=True) for t in train_tasks]
        validation_prompts : list[str]  = [t.get_prompt(eos_token=eos_token, with_plan=True) for t in validation_tasks]

        # Create datasets.Dataset objects
        train_dataset = datasets.Dataset.from_dict({"text": training_prompts})
        validation_dataset = datasets.Dataset.from_dict({"text": validation_prompts})
        config.log(f"Training dataset created with {len(train_dataset)} examples.", level=config.logging.INFO)
        config.log(f"Validation dataset created with {len(validation_dataset)} examples.", level=config.logging.INFO)

        del tasks, train_tasks, validation_tasks, training_prompts, validation_prompts

        # Create DatasetDict
        dataset = datasets.DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset
        })
        config.log(f"Dataset converted to DatasetDict successfully: {dataset}", level=config.logging.INFO)
        config.log(f"Number of training examples: {len(dataset['train'])}", level=config.logging.INFO)
        config.log(f"Number of validation examples: {len(dataset['validation'])}", level=config.logging.INFO)


    except Exception as e:
        config.log(f"Error loading dataset: {e}", level=config.logging.ERROR, exc_info=True)
        raise e

    # --- Tokenization ---
    config.log("Starting tokenization...", level=config.logging.INFO)
    def tokenize_fn(examples):
        # Tokenize the 'text' field which now contains the combined prompt and plan
        return tokenizer(
            examples["text"],
            max_length=config.get_config("max_seq_length", 2048), # Use max_seq_length from config
            truncation=True,
            padding=False,
        )

    try:
        config.log("Tokenizing datasets...", level=config.logging.INFO)
        # Tokenize the datasets.Dataset objects using .map()
        tokenized_train = dataset["train"].map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"], # Remove the original text column after tokenization
            desc="Tokenizing train dataset"
        )
        tokenized_val = dataset["validation"].map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"], # Remove the original text column
            desc="Tokenizing validation dataset"
        )
        config.log("Tokenization complete.", level=config.logging.INFO)
        config.log(f"Tokenized train dataset features: {tokenized_train.features}", level=config.logging.DEBUG)
        config.log(f"Tokenized validation dataset features: {tokenized_val.features}", level=config.logging.DEBUG)


    except Exception as e:
        config.log(f"Error during tokenization: {e}", level=config.logging.ERROR, exc_info=True)
        raise e


    # --- Data Collator ---
    # DataCollatorForLanguageModeling handles dynamic padding and prepares batches for causal LM training
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    config.log("Data collator initialized.", level=config.logging.INFO)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=model_checkpoint_dir,
        run_name=f"{config.get_config('model_name')}-{os.path.basename(model_checkpoint_dir)}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        report_to=config.get_config("report_to", "none"),
        num_train_epochs=config.get_config("num_train_epochs", 2),
        per_device_train_batch_size=config.get_config("batch_size", 1),
        per_device_eval_batch_size=config.get_config("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=config.get_config("gradient_accumulation_steps", 1),
        fp16=not config.get_config("bf16", False),
        bf16=config.get_config("bf16", False),
        learning_rate=config.get_config("learning_rate", 1.0e-5),
        lr_scheduler_type=config.get_config("lr_scheduler_type", "cosine"),
        weight_decay=config.get_config("weight_decay", 0.02),
        save_strategy=config.get_config("save_strategy", "steps"),
        save_steps=config.get_config("save_steps", 800),
        save_total_limit=config.get_config("save_total_limit", 1),
        logging_strategy=config.get_config("logging_strategy", "steps"),
        logging_steps=config.get_config("logging_steps", 400),
        eval_strategy=config.get_config("eval_strategy", "epoch"),
        optim=config.get_config("optimizer", "adamw_8bit"),
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
        end_time = datetime.datetime.now()
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        config.log(f"Training {model_name} -- finished {end_time_str}", level=config.logging.INFO)
        config.log(f"Total training time: {end_time - start_time}", level=config.logging.INFO)
    except Exception as e:
        config.log(f"Training failed: {e}", level=config.logging.ERROR)
        raise e 

    # --- Save Final Model ---
    try:
        trainer.save_model(model_checkpoint_dir)
        config.logging.info("Model saved to %s", model_checkpoint_dir)
    except Exception as e:
        config.log(f"Error saving final model/state: {e}", level=config.logging.ERROR, exc_info=True)

    # --- Clean up GPU memory ---
    del model
    del tokenizer
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        config.log("Cleaned GPU memory after training.", level=config.logging.INFO)