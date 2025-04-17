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
import datetime

# Disable tokenizer parallelism to avoid warnings after forking.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_training_procedure(model_checkpoint_dir, train_file, val_file, overwrite=False):
    config.logging.info(
        "Starting training procedure for model %s at %s",
        config.MODEL_TRAINING_CONFIG["model_name"],
        datetime.datetime.now()
    )

    # Determine if there's a checkpoint to resume from
    last_checkpoint = get_last_checkpoint(model_checkpoint_dir)
    if last_checkpoint:
        config.logging.info("Found existing checkpoint at %s. Resuming training.", last_checkpoint)
    else:
        config.logging.info("No checkpoint found. Starting new training. Output to: %s", model_checkpoint_dir)

    # Always load from the original pretrained model name:
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    pretrained_name = config.MODEL_TRAINING_CONFIG["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_name,
        trust_remote_code=True,
        use_auth_token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.MODEL_TRAINING_CONFIG["bf16"] else torch.float16,
        use_auth_token=hf_token
    )

    # Load datasets
    ds = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    for split in ["train", "validation"]:
        if len(ds[split]) == 0:
            msg = f"{split.capitalize()} dataset is empty."
            config.logging.error(msg)
            raise ValueError(msg)

    def tokenize_fn(ex):
        return tokenizer(
            ex["prompt"],
            max_length=config.MODEL_TRAINING_CONFIG["max_seq_length"],
            truncation=True,
            padding="max_length"
        )

    tokenized_train = ds["train"].map(tokenize_fn, batched=True, remove_columns=["prompt"])
    tokenized_val   = ds["validation"].map(tokenize_fn, batched=True, remove_columns=["prompt"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Build TrainingArguments
    args = TrainingArguments(
        output_dir=model_checkpoint_dir,
        run_name=f"{pretrained_name}-{os.path.basename(model_checkpoint_dir)}-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}",
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
        eval_strategy=config.MODEL_TRAINING_CONFIG.get("eval_strategy", "no"), 
        per_device_eval_batch_size=config.MODEL_TRAINING_CONFIG.get("eval_batch_size", 1),
        save_total_limit=config.MODEL_TRAINING_CONFIG.get("save_total_limit", None),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        tokenizer=tokenizer
    )

    config.logging.info("Beginning training at %s", datetime.datetime.now())
    trainer.train(resume_from_checkpoint=last_checkpoint)
    config.logging.info("Training finished at %s", datetime.datetime.now())

    trainer.save_model(model_checkpoint_dir)
    config.logging.info("Model checkpoint saved to %s", model_checkpoint_dir)
