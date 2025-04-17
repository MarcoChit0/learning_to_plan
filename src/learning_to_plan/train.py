import os
import datetime
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint

import learning_to_plan.config as config

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def cfg(key, default=None):
    return config.MODEL_TRAINING_CONFIG.get(key, default)


def run_training_procedure(model_checkpoint_dir, train_file, val_file):
    config.logging.info(
        "Training %s – start: %s",
        cfg("model_name"),
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    os.makedirs(model_checkpoint_dir, exist_ok=True)
    last_checkpoint = get_last_checkpoint(model_checkpoint_dir)
    model_source = last_checkpoint if last_checkpoint else cfg("model_name")
    config.logging.info(
        "%s checkpoint %s",
        "Resuming from" if last_checkpoint else "No checkpoint found – starting fresh from",
        model_source
    )

    dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    if len(dataset["train"]) == 0 or len(dataset["validation"]) == 0:
        raise ValueError("Train/validation dataset is empty.")

    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg("bf16") else torch.float16,
        use_auth_token=hf_token,
    )

    def tokenize_fn(ex):
        return tokenizer(
            ex["prompt"],
            max_length=cfg("max_seq_length"),
            truncation=True,
            padding="max_length",
        )

    tokenized_train = dataset["train"].map(tokenize_fn, batched=True, remove_columns=["prompt"])
    tokenized_val   = dataset["validation"].map(tokenize_fn, batched=True, remove_columns=["prompt"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=model_checkpoint_dir,
        run_name=f"{cfg('model_name')}-{os.path.basename(model_checkpoint_dir)}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        report_to=cfg("report_to"),
        num_train_epochs=cfg("num_train_epochs"),
        per_device_train_batch_size=cfg("batch_size"),
        per_device_eval_batch_size=cfg("per_device_eval_batch_size"),
        gradient_accumulation_steps=cfg("gradient_accumulation_steps"),
        fp16=not cfg("bf16"),
        bf16=cfg("bf16"),
        learning_rate=cfg("learning_rate"),
        lr_scheduler_type=cfg("lr_scheduler_type"),
        weight_decay=cfg("weight_decay"),
        save_strategy=cfg("save_strategy"),
        save_steps=cfg("save_steps"),
        save_total_limit=cfg("save_total_limit"),
        logging_strategy=cfg("logging_strategy"),
        logging_steps=cfg("logging_steps"),
        eval_strategy=cfg("eval_strategy"),
        optim=cfg("optimizer"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        processing_class=tokenizer,
    )

    config.logging.info("Starting training at %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    trainer.train(resume_from_checkpoint=last_checkpoint)
    config.logging.info("Training finished at %s.", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    trainer.save_model(model_checkpoint_dir)
    config.logging.info("Model saved to %s", model_checkpoint_dir)