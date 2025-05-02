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
logger = config.get_logger(__name__)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_training_procedure(domain):
    start_time = datetime.datetime.now()
    model_name = config.get_config('model_name')
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting training for {model_name} -- started {start_time_str}")

    model_checkpoint_dir = config.get_checkpoint_dir(domain, model_name)
    config.create_necessary_dirs(model_checkpoint_dir)
    logger.info(f"Checkpoints will be saved to: {model_checkpoint_dir}")

    # --- Load Model and Tokenizer ---
    logger.info(f"Loading model and tokenizer (checkpoint dir: {model_checkpoint_dir})...")

    model, tokenizer = config.load_model_and_tokenizer(checkpoint_dir=model_checkpoint_dir)
    assert tokenizer is not None, "Tokenizer loading failed."
    assert model is not None, "Model loading failed."
    logger.info("Model and tokenizer loaded successfully.")

    data_file_path = config.PROCESSED_DATA_FILE_PATH
    # --- Load and Prepare Dataset ---
    logger.info(f"Loading dataset from: {data_file_path}")
    try:
        from learning_to_plan import task
        assert os.path.exists(data_file_path), f"Data file {data_file_path} does not exist."

        # Load tasks from JSONL file
        tasks : set[task.Task] = task.get_tasks_from_jsonl(data_file_path)
        assert len(tasks) > 0, f"No tasks found in {data_file_path}."
        tasks = {t for t in tasks if t._domain == domain}
        assert len(tasks) > 0, f"No tasks found in {data_file_path} for domain {domain}."
        train_tasks : set[task.Task]  = {t for t in tasks if t._type == task.TaskType.TRAIN}
        assert len(train_tasks) > 0, "No training tasks found."
        validation_tasks : set[task.Task]  = {t for t in tasks if t._type == task.TaskType.VALIDATION}
        assert len(validation_tasks) > 0, "No validation tasks found."

        # Make the prompts that will be used for training and validation
        eos_token = tokenizer.eos_token if tokenizer.eos_token else ""
        training_prompts : list[str]  = [t.get_prompt(eos_token=eos_token, with_plan=True) for t in train_tasks]
        validation_prompts : list[str]  = [t.get_prompt(eos_token=eos_token, with_plan=True) for t in validation_tasks]

        # Create datasets.Dataset objects
        train_dataset = datasets.Dataset.from_dict({"text": training_prompts})
        validation_dataset = datasets.Dataset.from_dict({"text": validation_prompts})
        logger.info(f"Training dataset created with {len(train_dataset)} examples.")
        logger.info(f"Validation dataset created with {len(validation_dataset)} examples.")

        del tasks, train_tasks, validation_tasks, training_prompts, validation_prompts

        # Create DatasetDict
        dataset = datasets.DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset
        })
        logger.info(f"Dataset converted to DatasetDict successfully: {dataset}")
        logger.info(f"Number of training examples: {len(dataset['train'])}")
        logger.info(f"Number of validation examples: {len(dataset['validation'])}")


    except Exception as e:
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        raise e

    # --- Tokenization ---
    logger.info("Starting tokenization...")
    def tokenize_fn(examples):
        # Tokenize the 'text' field which now contains the combined prompt and plan
        return tokenizer(
            examples["text"],
            max_length=config.get_config("max_seq_length", 2048), # Use max_seq_length from config
            truncation=True,
            padding=False,
        )

    try:
        logger.info("Tokenizing datasets...")
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
        logger.info("Tokenization complete.")
        logger.debug(f"Tokenized train dataset features: {tokenized_train.features}")
        logger.debug(f"Tokenized validation dataset features: {tokenized_val.features}")


    except Exception as e:
        logger.error(f"Error during tokenization: {e}", exc_info=True)
        raise e


    # --- Data Collator ---
    # DataCollatorForLanguageModeling handles dynamic padding and prepares batches for causal LM training
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    logger.info("Data collator initialized.")

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

    logger.debug(f"Training Arguments: {training_args.to_dict()}")


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
        logger.info("No checkpoint found. Starting training from scratch.")
    else:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
        resume_from_checkpoint = last_checkpoint

    logger.info(f"Starting trainer.train() at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        end_time = datetime.datetime.now()
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Training {model_name} -- finished {end_time_str}")
        logger.info(f"Total training time: {end_time - start_time}")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True) # Added exc_info=True for better error reporting
        raise e

    # --- Save Final Model ---
    try:
        trainer.save_model(model_checkpoint_dir)
        logger.info("Model saved to %s", model_checkpoint_dir)
    except Exception as e:
        logger.error(f"Error saving final model/state: {e}", exc_info=True)

    # --- Clean up GPU memory ---
    del model
    del tokenizer
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleaned GPU memory after training.")