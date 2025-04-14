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
import learning_to_plan.config as config
def run_training_procedure(output_dir, train_file, val_file):
    import datetime

    config.logging.info("Starting training of model %s procedure at %s", config.MODEL_TRAINING_CONFIG["model_name"], datetime.datetime.now())

    dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})

    if len(dataset["train"]) == 0:
        config.logging.error("Training dataset is empty.")
        raise ValueError("Training dataset is empty.")
    if len(dataset["validation"]) == 0:
        config.logging.error("Validation dataset is empty.")
        raise ValueError("Validation dataset is empty.")

    from dotenv import load_dotenv
    load_dotenv()
    autentication_token = os.getenv("HUGGINGFACE_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_TRAINING_CONFIG["model_name"],
        trust_remote_code=True,
        use_auth_token=autentication_token
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_TRAINING_CONFIG["model_name"],
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator
    )

    config.logging.info("Starting training at %s", datetime.datetime.now())
    trainer.train()
    config.logging.info("Training completed at %s", datetime.datetime.now())

    trainer.save_model(output_dir)
    config.logging.info("Model saved to %s", output_dir)
