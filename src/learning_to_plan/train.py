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
def run_training_procedure(model_checkpoint_dir, train_file, val_file, overwrite=False):
    config.logging.info("Starting training procedure for model %s at %s",
                        config.MODEL_TRAINING_CONFIG["model_name"],
                        datetime.datetime.now())

    last_checkpoint = get_last_checkpoint(model_checkpoint_dir)
    if last_checkpoint is not None:
        config.logging.info("Found existing checkpoint at: %s. Resuming training.", last_checkpoint)
        model_source = model_checkpoint_dir
    else:
        config.logging.info("No checkpoint found. Starting new training, output to: %s", model_checkpoint_dir)
        model_source = config.MODEL_TRAINING_CONFIG["model_name"]

    # Load datasets
    dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    for d in ["train", "validation"]:
        if len(dataset[d]) == 0:
            e = f"{d.capitalize()} dataset is empty."
            config.logging.error(e)
            raise ValueError(e)

    load_dotenv()
    autentication_token = os.getenv("HUGGINGFACE_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        use_auth_token=autentication_token
    )

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

    tokenized_train = dataset["train"].map(tokenize_fn, batched=True, remove_columns=["prompt"])
    tokenized_val = dataset["validation"].map(tokenize_fn, batched=True, remove_columns=["prompt"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=model_checkpoint_dir,
        run_name=f"{config.MODEL_TRAINING_CONFIG['model_name']}-{os.path.basename(model_checkpoint_dir)}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
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
        eval_strategy=config.MODEL_TRAINING_CONFIG["eval_strategy"], 
        per_device_eval_batch_size=config.MODEL_TRAINING_CONFIG["eval_batch_size"],
        save_total_limit=config.MODEL_TRAINING_CONFIG["save_total_limit"],
        
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        tokenizer=tokenizer
    )

    config.logging.info("Starting training at %s", datetime.datetime.now())
    trainer.train(resume_from_checkpoint=last_checkpoint)
    config.logging.info("Training completed at %s", datetime.datetime.now())

    trainer.save_model(model_checkpoint_dir)
    config.logging.info("Model saved to %s", model_checkpoint_dir)
