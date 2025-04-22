import os
import datetime
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from transformers.trainer_utils import get_last_checkpoint

import learning_to_plan.config as config

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_training_procedure(model_checkpoint_dir, train_file, val_file):
    config.logging.info(
        "Training %s – start: %s",
        config.training_params("model_name"),
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    os.makedirs(model_checkpoint_dir, exist_ok=True)
    last_checkpoint = get_last_checkpoint(model_checkpoint_dir)
    model_source = last_checkpoint if last_checkpoint else config.training_params("model_name")
    config.logging.info(
        "%s checkpoint %s",
        "Resuming from" if last_checkpoint else "No checkpoint found – starting fresh from",
        model_source
    )

    dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    if len(dataset["train"]) == 0 or len(dataset["validation"]) == 0:
        raise ValueError("Train/validation dataset is empty.")

    load_dotenv()
    config.HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, token=config.HUGGINGFACE_TOKEN)

    # ---------- quantized‑load + optional LoRA --------------------------------
    if config.training_params("load_in_8bit"):
        quant_config.training_params = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config.training_params,
            token=config.HUGGINGFACE_TOKEN,
        )

        # attach LoRA adapter so the model becomes trainable
        lora_config.training_params = LoraConfig(
            r=config.training_params("lora_r"),
            lora_alpha=config.training_params("lora_r") * 4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "up_proj", "down_proj", "gate_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config.training_params)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if config.training_params("bf16") else torch.float16,
            token=config.HUGGINGFACE_TOKEN,
        )


    def tokenize_fn(ex):
        return tokenizer(
            ex["prompt"],
            max_length=config.training_params("max_seq_length"),
            truncation=True,
            padding="max_length",
        )

    tokenized_train = dataset["train"].map(tokenize_fn, batched=True, remove_columns=["prompt"])
    tokenized_val   = dataset["validation"].map(tokenize_fn, batched=True, remove_columns=["prompt"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=model_checkpoint_dir,
        run_name=f"{config.training_params('model_name')}-{os.path.basename(model_checkpoint_dir)}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        report_to=config.training_params("report_to"),
        num_train_epochs=config.training_params("num_train_epochs"),
        per_device_train_batch_size=config.training_params("batch_size"),
        per_device_eval_batch_size=config.training_params("per_device_eval_batch_size"),
        gradient_accumulation_steps=config.training_params("gradient_accumulation_steps"),
        fp16=not config.training_params("bf16"),
        bf16=config.training_params("bf16"),
        learning_rate=config.training_params("learning_rate"),
        lr_scheduler_type=config.training_params("lr_scheduler_type"),
        weight_decay=config.training_params("weight_decay"),
        save_strategy=config.training_params("save_strategy"),
        save_steps=config.training_params("save_steps"),
        save_total_limit=config.training_params("save_total_limit"),
        logging_strategy=config.training_params("logging_strategy"),
        logging_steps=config.training_params("logging_steps"),
        eval_strategy=config.training_params("eval_strategy"),
        optim=config.training_params("optimizer"),
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