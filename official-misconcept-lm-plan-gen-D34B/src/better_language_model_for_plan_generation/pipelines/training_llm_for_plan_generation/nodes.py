"""
This is a boilerplate pipeline 'training_llm_for_plan_generation'
generated using Kedro 0.19.6
"""

from functools import partial
import pickle
import sys
from icecream import ic

from better_language_model_for_plan_generation.utils.constant import *
import random
import os
import numpy as np
from glob import glob
from natsort import natsorted
from pddl.parser.problem import ProblemParser
import re
from tqdm.auto import tqdm
from pathlib import Path
from datasets import load_dataset
from datetime import datetime
import wandb
import torch
from transformers import EvalPrediction
from functools import partial



def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


def format_size(size_in_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} PB"


# ! NODE
def model_training(cfg):
    
    # only cuda available then from unsloth import FastLanguageModel # * import the model here to avoid the error if no GPU is available
    print("Loading the model")
    if torch.cuda.is_available():
        from unsloth import (
            FastLanguageModel,
        )  # * import the model here to avoid the error if no GPU is available
        from trl import SFTTrainer
        from transformers import TrainingArguments, EarlyStoppingCallback
        from unsloth import is_bfloat16_supported
        
        from better_language_model_for_plan_generation.pipelines.text_prompt_generation.plan_getter import (
            ANCHOR_PLAN_SEC,
        )
        from better_language_model_for_plan_generation.pipelines.training_llm_for_plan_generation.custom_sft_trainer import CustomSFT_Trainer
        from .custom_compute_metrics import compute_planning_metrics_compare_with_ref, compute_planning_metrics_using_val


    # Setup env vars
    HF_HOME = os.environ.get("HF_HOME")
    if HF_HOME is None: # in case the env var is not set
        os.environ["HF_HOME"] = os.path.join(
            os.environ["WORKING_DIR"], "data/01_raw/huggingface"
        )
        print(os.environ["HF_HOME"])
        Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

    datetime_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    config_type = CONFIG_TYPES.__members__[cfg["config_type_name"]].name
    
    max_seq_length = cfg['seq_params'][config_type]['max_seq_length']
    max_new_tokens = cfg['seq_params'][config_type]['max_new_tokens']
    batch_size = cfg['seq_params'][config_type]['batch_size']
    eval_batch_size = cfg['seq_params'][config_type]['eval_batch_size']
    gradient_accumulation_steps = cfg['seq_params'][config_type]['gradient_accumulation_steps']
    
    if cfg['debug_checkpoint_dir'] != "":
        assert os.path.exists(os.path.join(os.environ['WORKING_DIR'], cfg['debug_checkpoint_dir'])), "The debug checkpoint dir does not exist"
    if cfg['continue_train_dir'] != "":
        assert os.path.exists(os.path.join(os.environ['WORKING_DIR'], cfg['continue_train_dir'])), "The continue train dir does not exist"
        if cfg['wandb_record']:
            assert cfg['last_run_id'] != "", "The last run id is empty when continue training"
    
    
    # init wandb
    if cfg["is_train"] and cfg["wandb_record"]:
        job_name = (
            f"Pretraining LLM on Plan Gen Dataset - Config: {cfg['config_type_name']}"
        )
        cfg["timestamp"] = datetime_timestamp
        wandb_kwargs = {
            "project": "Better LM for Plan Generation",
            "name": job_name,
            "config": cfg,
        }
        
        if cfg['last_run_id'] != "":
            wandb_kwargs['id'] = cfg['last_run_id']
            wandb_kwargs['resume'] = 'must'
            # remove config and name because it is resume
            wandb_kwargs.pop('config')
            wandb_kwargs.pop('name')
        
        wandb.init(
            **wandb_kwargs,
        )
    # log the config
    ic(cfg)

    # * Setup model
    if cfg["dtype"] == "bfloat16":
        dtype = torch.bfloat16
    elif cfg["dtype"] == "float16":
        dtype = torch.float16
    elif cfg["dtype"] == "float32":
        dtype = torch.float32

    if cfg["is_train"]:
        if cfg['continue_train_dir'] == "":
            model_name = cfg["model_name"]
        else:
            model_name = os.path.join(os.environ['WORKING_DIR'], cfg['continue_train_dir']) 
            print(f"Continuing training from: {model_name}")
    else:
        if cfg['debug_checkpoint_dir'] == "":
            model_name = input("Enter the abs checkpoint path: ")
            if model_name == "":
                model_name = cfg["model_name"]
        else:
            model_name = os.path.join(os.environ['WORKING_DIR'], cfg['debug_checkpoint_dir']) 
            print(f"Using the debug checkpoint: {model_name}")  
                       

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,    
        load_in_4bit=False,  # this is for 4bit quantization
    )
    model.config.use_cache = False # ! this is to avoid the error of "RuntimeError: CUDA out of memory" when using cache

    # * Tokenizer update
    # 1. add special tokens
    # tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) # ! it seems that it faces some issues https://unsloth.ai/blog/llama3 , thus we do not use it
    # 2. get EOS_TOKEN
    EOS_TOKEN = tokenizer.eos_token  # must add this token at the end of the prompt

    # * Load dataset
    def process_dataset(batch):
        full_text_prompt = batch[FEATURE_CLS.full_text_prompt.name]
        return {
            FEATURE_CLS.full_text_prompt.name: [x + EOS_TOKEN for x in full_text_prompt]
        }

    

    train_dataset = load_dataset(
        os.path.join(os.environ["WORKING_DIR"], cfg["huggingface_dataset_dir"]),
        config_type,
        split=SPLIT_NAMES.train.name,
    )
    # apply map if EOS_TOKEN is not in the prompt
    if EOS_TOKEN not in train_dataset[0][FEATURE_CLS.full_text_prompt.name]:
        print("Adding EOS_TOKEN to the train dataset")
        # assert False, "The EOS_TOKEN is not in the prompt"
        train_dataset = train_dataset.map(
            process_dataset,
            batched=True,
        )
        
    # shuffle the train dataset
    train_dataset = train_dataset.shuffle(seed=42)
    train_dataset = train_dataset.flatten_indices()

    # eval dataset
    eval_dataset = load_dataset(
        os.path.join(os.environ["WORKING_DIR"], cfg["huggingface_dataset_dir"]),
        config_type,
        split=SPLIT_NAMES.test_same_domain.name,
    )
    # ! if greater than 1600, randomly sample 1600
    if len(eval_dataset) > 1600:
        eval_dataset = eval_dataset.shuffle(seed=42).select(range(1600))
        
    if EOS_TOKEN not in eval_dataset[0][FEATURE_CLS.full_text_prompt.name]:
        print("Adding EOS_TOKEN to the eval dataset")
        # assert False, "The EOS_TOKEN
        eval_dataset = eval_dataset.map(
            process_dataset,
            batched=True,
        )
        
    config_cache_dict = dict()
    plan_output_dir = os.path.join(os.environ["WORKING_DIR"], cfg["plan_output_dir"])
    plan_output_dir = plan_output_dir.format(timestamp=datetime_timestamp)
    
    if cfg["is_train"] and torch.cuda.is_available():
        # clean cache
        torch.cuda.empty_cache()
        if cfg['continue_train_dir'] == "":
            checkpoint_output_dir = cfg["checkpoint_output_dir"].format(
                config_type= config_type,
                timestamp=datetime_timestamp
            )
            checkpoint_output_dir = os.path.join(
                os.environ["WORKING_DIR"], checkpoint_output_dir
            )
        else:
            # get parent dir
            checkpoint_output_dir = Path(model_name).parent.resolve()
        ic(checkpoint_output_dir)
        # * Training
       
        training_args_kwargs = cfg["training_args"] 
        training_args_kwargs['gradient_accumulation_steps'] = gradient_accumulation_steps
        
        trainer = CustomSFT_Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field=FEATURE_CLS.full_text_prompt.name,
            max_seq_length=max_seq_length,
            dataset_num_proc=cfg["dataset_num_proc"],
            packing=cfg["packing"],
            dataset_kwargs={
                "if_use_pddl_val": False, # will not call VAL to eval the model
            },
            args=TrainingArguments(
                **training_args_kwargs,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                output_dir=checkpoint_output_dir,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=eval_batch_size,
                prediction_loss_only=True,  # this is for training time eval
                dataloader_pin_memory=True, # set False to save memory  
                greater_is_better=True,
                eval_accumulation_steps=1, # set it smaller to avoid OOM
                metric_for_best_model="planning_metric", 
                report_to="wandb" if cfg["wandb_record"] else "none",
            ),
            compute_metrics=partial(
                compute_planning_metrics_compare_with_ref,
                config_cache_dict=config_cache_dict,
                config_dir=os.path.join(os.environ["WORKING_DIR"], cfg["planning_domain_config_dir"]),
            ),
            # special SFTTrainer compute_metrics
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=cfg["early_stopping_patience"]
                )
            ],
        )
        
        # compute_metrics=partial(
        #     compute_planning_metrics_using_val,
        #     val_fp=os.path.join(os.environ["WORKING_DIR"], cfg["val_fp"]),
        #     domain_fp_pattern=os.path.join(os.environ["WORKING_DIR"], cfg["domain_fp_pattern"]),
        #     problem_instance_fp_pattern=os.path.join(os.environ["WORKING_DIR"], cfg["problem_instance_fp_pattern"]),
        #     plan_output_dir=plan_output_dir,
        #     config_dir=os.path.join(os.environ["WORKING_DIR"], cfg["planning_domain_config_dir"]),
        #     config_cache_dict=config_cache_dict,
        #     split_name=SPLIT_NAMES.test_same_domain.name,
        # ),

        # - Print Stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        # ! train the model
        if cfg['continue_train_dir'] == "":
            trainer_stats = trainer.train()
        else:
            trainer_stats = trainer.train(resume_from_checkpoint=model_name)

        # *
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(
            f"Peak reserved memory for training % of max memory = {lora_percentage} %."
        )

        output_size = format_size(get_folder_size(checkpoint_output_dir))
        print(f"Output size = {output_size}")

    # * Try some predictions
    FastLanguageModel.for_inference(model)
    test_entry = train_dataset[12]
    prompt = (
        test_entry[FEATURE_CLS.domain_text.name]
        + test_entry[FEATURE_CLS.problem_init_text.name]
        + test_entry[FEATURE_CLS.problem_goal_text.name]
        + ANCHOR_PLAN_SEC
    )
    print("PROMPT")
    print(prompt)

    inputs = tokenizer([prompt], return_tensors="pt").to('cuda')
    # inputs = tokenizer([prompt[:4000]], return_tensors="pt").to('cuda')
    # len(inputs['input_ids'][0])
    outputs = model.generate(**inputs, 
                             max_new_tokens= max_new_tokens,
                             use_cache=True)
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print("OUTPUT")
    print(decoded_output[0])

    # end wandb
    if cfg["is_train"] and cfg["wandb_record"]:
        wandb.finish()
