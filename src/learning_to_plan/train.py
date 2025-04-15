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
from transformers import TrainerCallback
import numpy as np

# Add this import at the top
import shutil
import datetime
import pandas as pd
import json

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

class TrainingHistoryCallback(TrainerCallback):
    """Records training metrics to a CSV file after each epoch"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.history_file = os.path.join(output_dir, config.TRAINING_HISTORY_FILE_NAME)
        self.current_epoch_metrics = {}
        
        # Load existing history if available
        if os.path.exists(self.history_file):
            self.history = pd.read_csv(self.history_file)
        else:
            # Create a new dataframe
            self.history = pd.DataFrame(columns=[
                'epoch', 'train_loss', 'eval_loss', 'eval_accuracy', 
                'test_loss', 'test_accuracy', 'timestamp'
            ])
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        
        # Store metrics for later use
        self.current_epoch_metrics.update(metrics)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare a new row with current metrics
        new_row = {
            'epoch': epoch,
            'timestamp': timestamp,
            **self.current_epoch_metrics
        }
        
        # Update the history dataframe
        self.history = pd.concat([self.history, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save the updated history to CSV
        self.history.to_csv(self.history_file, index=False)
        config.logging.info(f"Training history updated for epoch {epoch}")
        
        # Clear metrics for next epoch
        self.current_epoch_metrics = {}

class TestEvaluationCallback(TrainerCallback):
    """Evaluates on test dataset after each epoch"""
    def __init__(self, trainer, test_dataset, history_callback=None):
        self.trainer = trainer
        self.test_dataset = test_dataset
        self.history_callback = history_callback
        
    def on_epoch_end(self, args, state, control, **kwargs):
        metrics = self.trainer.evaluate(eval_dataset=self.test_dataset, metric_key_prefix="test")
        config.logging.info(
            f"Epoch {state.epoch}: Test metrics: accuracy={metrics.get('test_accuracy', 'N/A'):.4f}, "
            f"loss={metrics.get('test_loss', 'N/A'):.4f}"
        )
        
        # Update history with test results
        if self.history_callback is not None:
            self.history_callback.current_epoch_metrics.update(metrics)
        
        return control

def run_training_procedure(output_dir, train_file, val_file, test_file, overwrite=False):
    config.logging.info("Starting training of model %s procedure at %s", config.MODEL_TRAINING_CONFIG["model_name"], datetime.datetime.now())

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
            config.logging.info(f"Saved backup of previous model to {backup_dir}")
            
            model_name = output_dir
            config.logging.info("Continuing training of existing model from: %s", output_dir)
    else:
        model_name = config.MODEL_TRAINING_CONFIG["model_name"]
        config.logging.info("Starting new training, output to: %s", output_dir)
    
    # Save current parameters
    save_model_params(output_dir)
    

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
        use_auth_token=autentication_token
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
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

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["prompt"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=f"{config.MODEL_TRAINING_CONFIG['model_name']}-{os.path.basename(output_dir)}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none", 
        num_train_epochs=config.MODEL_TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=config.MODEL_TRAINING_CONFIG["batch_size"],
        per_device_eval_batch_size=config.MODEL_TRAINING_CONFIG["eval_batch_size"],
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
        eval_strategy="epoch",  # Evaluate at the end of each epoch
    )


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        
        # For causal language modeling, we need to shift predictions and labels
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        
        # Get predicted tokens
        preds = np.argmax(shift_logits, axis=-1)
        
        # Mask out padding tokens (typically -100 in transformers)
        mask = (shift_labels != -100)
        
        # Calculate token-level accuracy
        correct = (preds == shift_labels) * mask
        total_tokens = mask.sum()
        accuracy = correct.sum() / total_tokens if total_tokens > 0 else 0
        
        return {
            "accuracy": float(accuracy)
        }

    history_callback = TrainingHistoryCallback(output_dir)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    # Add callbacks
    trainer.add_callback(TrainingHistoryCallback(output_dir))
    trainer.add_callback(TestEvaluationCallback(trainer, tokenized["test"], history_callback))

    config.logging.info("Starting training at %s", datetime.datetime.now())
    trainer.train()
    config.logging.info("Training completed at %s", datetime.datetime.now())

    trainer.save_model(output_dir)
    config.logging.info("Model saved to %s", output_dir)
