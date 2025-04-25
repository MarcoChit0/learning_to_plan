import os
import datetime
from datasets import load_dataset, DatasetDict, Dataset # Import DatasetDict and Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint
import torch # Import torch to check for CUDA availability

import learning_to_plan.config as config

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_training_procedure(model_checkpoint_dir, data_file_path):
    """Runs the model training procedure for a given domain."""
    start_time = datetime.datetime.now()
    model_name = config.get_config('model_name')
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    config.log(f"Training {model_name} -- started {start_time_str}", level=config.logging.INFO)

    os.makedirs(model_checkpoint_dir, exist_ok=True)

    # --- Load Model and Tokenizer (Load first to define tokenizer for the function below) ---
    model, tokenizer = config.load_model_and_tokenizer(checkpoint_dir=model_checkpoint_dir)

    assert tokenizer is not None, f"Tokenizer is None. Check the checkpoint directory: {model_checkpoint_dir}"
    assert model is not None, f"Model is None. Check the checkpoint directory: {model_checkpoint_dir}"

    # --- Define Tokenization Function (Define BEFORE using it) ---
    def tokenize_fn(batch):
        """Tokenizes prompts and plans, combining them for causal LM training."""
        # Combine prompt and plan. Ensure plan exists and is not None. Add EOS token.
        # Handle potential None values gracefully
        full_texts = [
            (p if p else "") + (pl if pl else "") + tokenizer.eos_token
            for p, pl in zip(batch.get("prompt", []), batch.get("plan", [])) # Use .get for safety
        ]
        tokenized = tokenizer(
            full_texts,
            max_length=config.get_config("max_seq_length", 2048), # Use default if not set
            truncation=True,
            # Padding handled by DataCollator
        )
        return tokenized

    # --- Load and Prepare Dataset ---
    config.log(f"Loading dataset from: {data_file_path}", level=config.logging.INFO)
    try:
        # Load the full dataset first
        full_dataset = load_dataset("json", data_files=data_file_path)["train"] # load_dataset returns a DatasetDict

        # Filter into train and validation sets based on the "type" field
        train_dataset = full_dataset.filter(lambda example: example.get("type") == "train")
        validation_dataset = full_dataset.filter(lambda example: example.get("type") == "validation")

        # Check if datasets are empty after filtering
        if len(train_dataset) == 0:
            raise ValueError(f"Training dataset is empty after filtering for 'type' == 'train' in {data_file_path}")

        config.log(f"Train dataset size: {len(train_dataset)}", level=config.logging.INFO)
        config.log(f"Validation dataset size: {len(validation_dataset)}", level=config.logging.INFO)

        # Tokenize datasets
        config.log("Tokenizing datasets...", level=config.logging.INFO)
        tokenized_train = train_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=train_dataset.column_names # Remove all original columns
        )

        tokenized_val = None
        if len(validation_dataset) > 0:
            tokenized_val = validation_dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=validation_dataset.column_names # Remove all original columns
            )
        else:
            config.log("Validation dataset is empty. Proceeding without validation.", level=config.logging.WARNING)

    except Exception as e:
        config.log(f"Error loading or processing dataset from {data_file_path}: {e}", level=config.logging.ERROR, exc_info=True)
        raise e


    # --- Setup Trainer ---
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Check CUDA availability for ddp setting
    use_ddp = torch.cuda.is_available() and torch.cuda.device_count() > 1

    training_args = TrainingArguments(
        output_dir=model_checkpoint_dir,
        run_name=f"{config.get_config('model_name')}-{os.path.basename(model_checkpoint_dir)}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        report_to=config.get_config("report_to", "none"), # Default to none if not set
        num_train_epochs=config.get_config("num_train_epochs", 1), # Default epochs
        per_device_train_batch_size=config.get_config("batch_size", 1), # Default batch size
        per_device_eval_batch_size=config.get_config("eval_batch_size", 1), # Default eval batch size
        gradient_accumulation_steps=config.get_config("gradient_accumulation_steps", 1),
        fp16=not config.get_config("bf16", False) and torch.cuda.is_available(), # Enable fp16 only if not bf16 and CUDA available
        bf16=config.get_config("bf16", False) and torch.cuda.is_bf16_supported(), # Enable bf16 only if supported
        learning_rate=config.get_config("learning_rate", 5e-5), # Default LR
        lr_scheduler_type=config.get_config("lr_scheduler_type", "linear"),
        weight_decay=config.get_config("weight_decay", 0.01),
        save_strategy=config.get_config("save_strategy", "steps"),
        save_steps=config.get_config("save_steps", 500),
        save_total_limit=config.get_config("save_total_limit", 2),
        logging_strategy=config.get_config("logging_strategy", "steps"),
        logging_steps=config.get_config("logging_steps", 100),
        eval_strategy=config.get_config("eval_strategy", "steps") if tokenized_val else "no", # Evaluate only if val data exists
        eval_steps=config.get_config("eval_steps", 500) if tokenized_val else None, # Eval steps only if val data exists
        optim=config.get_config("optimizer", "adamw_torch"), # Default optimizer
        # --- DDP Specific ---
        # Explicitly disable finding unused parameters if using DDP
        # This can sometimes resolve NCCL/DDP errors with certain model architectures
        ddp_find_unused_parameters=False if use_ddp else None,
        # Use 'auto' for device placement, Trainer handles it well with accelerate
        # no_cuda=not torch.cuda.is_available(), # Let Trainer/Accelerate handle device placement
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val, # Pass None if validation set is empty
        data_collator=collator,
        tokenizer=tokenizer, # Pass tokenizer for auto-padding/saving and collator checks
    )

    # --- Start Training ---
    last_checkpoint = get_last_checkpoint(model_checkpoint_dir)
    if last_checkpoint:
        config.log(f"Resuming training from checkpoint: {last_checkpoint}", level=config.logging.INFO)
    else:
        config.log("No checkpoint found. Starting training from scratch.", level=config.logging.INFO)

    config.log(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", level=config.logging.INFO)

    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
        config.log(f"Training finished successfully. Saving final model to {model_checkpoint_dir}", level=config.logging.INFO)
        trainer.save_model(model_checkpoint_dir) # Explicitly save final model
        # Tokenizer is often saved automatically by Trainer when passed, but saving explicitly doesn't hurt
        tokenizer.save_pretrained(model_checkpoint_dir)
    except Exception as e:
        config.log(f"Training failed with error: {e}", level=config.logging.ERROR, exc_info=True)
        # Optionally re-raise the exception if you want the script to exit with an error code
        raise e
    finally:
        # --- Log End Time and Duration ---
        end_time = datetime.datetime.now()
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        config.log(f"Training procedure for {model_name} finished at {end_time_str}", level=config.logging.INFO)
        config.log(f"Total training time: {end_time - start_time}", level=config.logging.INFO)
