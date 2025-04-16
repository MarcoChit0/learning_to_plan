import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import learning_to_plan.config as config
from dotenv import load_dotenv
import numpy as np
import shutil
import datetime
import pandas as pd
import json

# Disable tokenizer parallelism to avoid warnings after forking.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def save_model_params(output_dir):
    """Save current training parameters to a JSON file"""
    params = {k: v for k, v in config.MODEL_TRAINING_CONFIG.items()}
    params_file = os.path.join(output_dir, "training_params.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    return params

def params_changed(output_dir):
    """Check if parameters have changed from previous run"""
    params_file = os.path.join(output_dir, "training_params.json")
    if not os.path.exists(params_file):
        return True  # No previous params, so consider it changed
    
    with open(params_file, 'r') as f:
        previous_params = json.load(f)
    
    current_params = {k: v for k, v in config.MODEL_TRAINING_CONFIG.items() if k != "num_train_epochs"}
    return current_params != previous_params

def run_training_procedure(output_dir, train_file, val_file, test_file, overwrite=False):
    config.logging.info("Starting training of model %s procedure at %s",
                        config.MODEL_TRAINING_CONFIG["model_name"],
                        datetime.datetime.now())

    model_exists = os.path.exists(os.path.join(output_dir, "pytorch_model.bin"))
    params_have_changed = params_changed(output_dir) if model_exists else False

    if params_have_changed:
        config.logging.info("Model parameters have changed from previous run. Forcing overwrite.")
        overwrite = True

    if model_exists:
        if overwrite:
            config.logging.info("Overwriting existing model in: %s", output_dir)
            model_name = config.MODEL_TRAINING_CONFIG["model_name"]
        else:
            backup_dir = f"{output_dir}_backup"
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            shutil.copytree(output_dir, backup_dir)
            config.logging.info("Saved backup of previous model to %s", backup_dir)
            model_name = output_dir
            config.logging.info("Continuing training of existing model from: %s", output_dir)
    else:
        model_name = config.MODEL_TRAINING_CONFIG["model_name"]
        config.logging.info("Starting new training, output to: %s", output_dir)

    # Save current parameters
    save_model_params(output_dir)

    # Load dataset
    dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file, "test": test_file})
    for split in ["train", "validation", "test"]:
        if len(dataset[split]) == 0:
            e = f"{split.capitalize()} dataset is empty."
            config.logging.error(e)
            raise ValueError(e)

    load_dotenv()
    autentication_token = os.getenv("HUGGINGFACE_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=autentication_token
    )

    # Load initial model (or resume from checkpoint if available)
    if not model_exists or overwrite:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if config.MODEL_TRAINING_CONFIG["bf16"] else torch.float16,
            token=autentication_token
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if config.MODEL_TRAINING_CONFIG["bf16"] else torch.float16,
            token=autentication_token
        )

    def tokenize_fn(example):
        return tokenizer(
            example["prompt"],
            max_length=config.MODEL_TRAINING_CONFIG["max_seq_length"],
            truncation=True,
            padding="max_length"
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["prompt"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Shift for causal language modeling
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        preds = np.argmax(shift_logits, axis=-1)
        mask = (shift_labels != -100)
        correct = (preds == shift_labels) * mask
        total_tokens = mask.sum()
        accuracy = correct.sum() / total_tokens if total_tokens > 0 else 0
        return {"accuracy": float(accuracy)}

    # Determine starting epoch from training history (if any)
    history_file = os.path.join(output_dir, config.TRAINING_HISTORY_FILE_NAME)
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        if not history_df.empty:
            start_epoch = int(history_df['epoch'].max()) + 1
        else:
            start_epoch = 0
    else:
        start_epoch = 0

    total_epochs = config.MODEL_TRAINING_CONFIG["num_train_epochs"]

    for epoch in range(start_epoch, total_epochs):
        config.logging.info("Epoch %d: Starting training.", epoch)

        # Build training arguments for a single epoch
        train_args = TrainingArguments(
            output_dir=output_dir,
            run_name=f"{config.MODEL_TRAINING_CONFIG['model_name']}-epoch{epoch}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            logging_dir=os.path.join(output_dir, "logs"),
            report_to="none",
            num_train_epochs=1,  # one epoch at a time
            per_device_train_batch_size=config.MODEL_TRAINING_CONFIG["batch_size"],
            gradient_accumulation_steps=config.MODEL_TRAINING_CONFIG["gradient_accumulation_steps"],
            fp16=not config.MODEL_TRAINING_CONFIG["bf16"],
            bf16=config.MODEL_TRAINING_CONFIG["bf16"],
            learning_rate=config.MODEL_TRAINING_CONFIG["learning_rate"],
            lr_scheduler_type=config.MODEL_TRAINING_CONFIG["lr_scheduler_type"],
            weight_decay=config.MODEL_TRAINING_CONFIG["weight_decay"],
            save_strategy="no",  # we save manually below
            logging_strategy=config.MODEL_TRAINING_CONFIG["logging_strategy"],
            logging_steps=config.MODEL_TRAINING_CONFIG["logging_steps"],
            optim=config.MODEL_TRAINING_CONFIG["optimizer"],
        )

        # Create a Trainer for training this epoch
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=tokenized["train"],
            data_collator=collator,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer
        )

        try:
            trainer.train()
            trainer.save_model(output_dir)
            config.logging.info("Epoch %d: Training completed and model saved.", epoch)
        except Exception as train_exc:
            config.logging.error("Epoch %d: Training failed: %s", epoch, train_exc)
            trainer.save_model(output_dir)
            break
        finally:
            del trainer
            torch.cuda.empty_cache()

        # Evaluation phase in a separate session
        try:
            model_eval = AutoModelForCausalLM.from_pretrained(
                output_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if config.MODEL_TRAINING_CONFIG["bf16"] else torch.float16,
                token=autentication_token
            )
            eval_args = TrainingArguments(
                output_dir=output_dir,
                report_to="none",
                per_device_eval_batch_size=config.MODEL_TRAINING_CONFIG["eval_batch_size"],
            )
            trainer_eval = Trainer(
                model=model_eval,
                args=eval_args,
                eval_dataset=tokenized["test"],
                data_collator=collator,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer
            )
            config.logging.info("Epoch %d: Starting evaluation.", epoch)
            eval_metrics = trainer_eval.evaluate()
            config.logging.info("Epoch %d: Evaluation metrics: %s", epoch, eval_metrics)
        except Exception as eval_exc:
            config.logging.error("Epoch %d: Evaluation failed: %s", epoch, eval_exc)
            eval_metrics = {}
        finally:
            # Append evaluation metrics to training history CSV
            new_row = {
                'epoch': epoch,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            new_row.update(eval_metrics)
            if os.path.exists(history_file):
                history_df = pd.read_csv(history_file)
                history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                history_df = pd.DataFrame([new_row])
            history_df.to_csv(history_file, index=False)
            try:
                del trainer_eval, model_eval
            except Exception:
                pass
            torch.cuda.empty_cache()

        # Reload the model for the next training epoch from the saved checkpoint
        try:
            model = AutoModelForCausalLM.from_pretrained(
                output_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if config.MODEL_TRAINING_CONFIG["bf16"] else torch.float16,
                token=autentication_token
            )
        except Exception as e:
            config.logging.error("Failed to reload model for next epoch: %s", e)
            break

    config.logging.info("Training procedure completed at %s", datetime.datetime.now())