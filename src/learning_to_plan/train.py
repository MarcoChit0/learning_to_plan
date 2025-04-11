# train.py
#
# Minimal training script that:
#  1. Loads parameters from config.py
#  2. Respects the function signature run_training_procedure(output_dir, train_file, val_file)
#  3. Adopts the new parameters from the paper where feasible
#  4. Disclaims using a smaller model or smaller batch sizes if needed

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
from config import MODEL_TRAINING_CONFIG

def load_deepspeed_config(ds_config_path):
    if not os.path.exists(ds_config_path):
        raise FileNotFoundError(f"DeepSpeed config file not found: {ds_config_path}")
    return ds_config_path

def run_training_procedure(output_dir, train_file, val_file, model_name=None, num_train_epochs=None):
    dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    assert len(dataset["train"]) > 0, "Training dataset is empty."
    assert len(dataset["validation"]) > 0, "Validation dataset is empty."

    model_name = model_name if model_name else MODEL_TRAINING_CONFIG["model_name"]
    num_train_epochs = num_train_epochs if num_train_epochs else MODEL_TRAINING_CONFIG["num_train_epochs"]
    from dotenv import load_dotenv
    load_dotenv()
    autentication_token = os.getenv("HUGGINGFACE_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=autentication_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if MODEL_TRAINING_CONFIG["bf16"] else torch.float16,
        use_auth_token=autentication_token
    )

    def tokenize_fn(example):
        return tokenizer(
            example["prompt"],
            max_length=MODEL_TRAINING_CONFIG["max_seq_length"],
            truncation=True,
            padding="max_length"
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["prompt"])
    collator  = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # TODO: on the paper, they use deepspeed for speedup training. For now, I will not use it. I will add it later.
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=MODEL_TRAINING_CONFIG["batch_size"],
        per_device_eval_batch_size=MODEL_TRAINING_CONFIG["eval_batch_size"],
        gradient_accumulation_steps=MODEL_TRAINING_CONFIG["gradient_accumulation_steps"],
        fp16=not MODEL_TRAINING_CONFIG["bf16"],   # fallback if no bf16
        bf16=MODEL_TRAINING_CONFIG["bf16"],       # from paper
        learning_rate=MODEL_TRAINING_CONFIG["learning_rate"],
        lr_scheduler_type=MODEL_TRAINING_CONFIG["lr_scheduler_type"],
        weight_decay=MODEL_TRAINING_CONFIG["weight_decay"],
        save_steps=MODEL_TRAINING_CONFIG["save_steps"],
        logging_steps=MODEL_TRAINING_CONFIG["logging_steps"],
        optim=MODEL_TRAINING_CONFIG["optimizer"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(output_dir)
