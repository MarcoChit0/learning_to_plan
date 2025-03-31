"""
This is a boilerplate pipeline 'further_ppo_training'
generated using Kedro 0.19.6
"""

from copy import deepcopy
from functools import partial
import deepspeed
from icecream import ic


ANCHOR_PLAN_SEC = "My plan is as follows:\n<PLAN>\n"

from datasets import load_dataset

from better_language_model_for_plan_generation.utils.constant import *
from better_language_model_for_plan_generation.utils.eval_generated_plan_text import *
from better_language_model_for_plan_generation.utils.custom_compute_metrics import compute_planning_metrics_compare_with_ref, compute_planning_metrics_using_val
import random
import os
from glob import glob
from natsort import natsorted
import re
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime
import wandb
import torch
from functools import partial
from accelerate.utils import gather_object 
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
import time
from datasets import load_dataset, load_from_disk
from better_language_model_for_plan_generation.utils.constant import SPLIT_NAMES
from accelerate import Accelerator
import shutil

from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from trl import AutoModelForCausalLMWithValueHead
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    
)

from trl import ModelConfig
from trl import PPOv2Config
from trl.trainer.rloo_trainer import RLOOConfig
from better_language_model_for_plan_generation.utils.custom_reward_model_rloo_trainer import CustomRewardRLOOTrainer

from better_language_model_for_plan_generation.utils.custom_reward_model_ppo_trainer import CustomRewardPPOv2Trainer

from better_language_model_for_plan_generation.utils.custom_compute_metrics import compute_planning_metrics_compare_with_ref, compute_planning_metrics_using_val
import torch.nn as nn


if torch.cuda.is_available():
    from transformers import AutoModelForCausalLM, AutoTokenizer

# ! Check ref: https://github.com/huggingface/trl/blob/main/examples/scripts/rloo/rloo_tldr.py
# ref2: https://github.com/mst272/LLM-Dojo/blob/6d25868f0206a55c508cdcbc768c825011e3e78c/rlhf/rlhf_train.py#L203

def find_all_linear_names(model):
    cls = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    return lora_module_names

def my_reward_function(compute_metrics_compare_ref, compute_metrics_using_val, eval_preds):
    val_binary_score = compute_metrics_using_val(eval_preds=eval_preds)['planning_metric']
    lccs_score = compute_metrics_compare_ref(eval_preds=eval_preds, if_lccs=False)['planning_metric']
    final_output = []
    for i in range(len(val_binary_score)):
        if val_binary_score[i] < 0.5:
            final_output.append(lccs_score[i])
        else:
            final_output.append(1.0)
    return final_output

# ! NODE
def ppo_training(ppo_cfg, general_cfg):
    # HF_HOME backup 
    HF_HOME = os.environ.get("HF_HOME")
    if HF_HOME is None: # in case the env var is not set
        os.environ["HF_HOME"] = os.path.join(
            os.environ["WORKING_DIR"], "data/01_raw/huggingface"
        )
        print(os.environ["HF_HOME"])
        Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    
    config_type_name = ppo_cfg["config_type_name"] 
    model_config_type = config_type_name
    checkpoint_dir = ppo_cfg["checkpoint_dir"]
    if checkpoint_dir == "":
        raise ValueError(f"checkpoint_dir is empty for config_type_name: {config_type_name}")
    checkpoint_dir = os.path.join(os.environ["WORKING_DIR"], checkpoint_dir)
    
    seed = general_cfg["seed"]
    example_per_domain = general_cfg["example_per_domain"]
    
    # ! -- Create a separate folder later to store the ppo training checkpoints
    
    ppo_checkpoint_output_dir = os.path.join(Path(checkpoint_dir).parent, "ppo")
    
    is_llama_factory = ppo_cfg["is_llama_factory"]
    assert is_llama_factory, "Only support llama factory model for now"
    
    # Init accelerator 
    
    # kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=6)).to_kwargs()
    # accelerator_raw = Accelerator( 
    #     kwargs_handlers=[
    #         kwargs
    #     ]
    # )  
    
    kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=6)).to_kwargs()
    accelerator = PartialState(**kwargs)
    
    # message= [f"Hello this is GPU {accelerator.process_index}"]
    # messages=gather_object(message)
    # accelerator.print(messages)
    
    
    # breakpoint()
    # partial_state = PartialState()
    
    model_name = checkpoint_dir # for llama factory, the model name is the checkpoint dir
    
    # -- Wandb init
    if general_cfg['wandb_record']:
        job_name = f'PPO Further Training - Config: {config_type_name}'
        merge_dict = {**general_cfg, **ppo_cfg}
        wandb_kwargs = {
            "project": "ACCE LM for Plan Generation",
            "name": job_name,
            "config": merge_dict,
            "tags": ["ppo", "further_training"]
        }
        
    # ! -- Setup Metrics as Reward Function 
    config_cache_dict = dict()
    config_dir = os.path.join(os.environ["WORKING_DIR"], general_cfg["planning_domain_config_dir"])
    val_fp=os.path.join(os.environ["WORKING_DIR"], general_cfg["val_fp"])
    domain_fp_pattern=os.path.join(
        os.environ["WORKING_DIR"], general_cfg["domain_fp_pattern"]
    )
    problem_instance_fp_pattern=os.path.join(
        os.environ["WORKING_DIR"], general_cfg["problem_instance_fp_pattern"]
    )
    split_name_str = SPLIT_NAMES.test_longer_horizon.name
    datetime_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plan_output_dir = os.path.join(os.environ["WORKING_DIR"], general_cfg["plan_output_dir"])
    plan_output_dir = plan_output_dir.format(timestamp=datetime_timestamp)
    # remove / if it is the last character
    if plan_output_dir[-1] == "/":
        plan_output_dir = plan_output_dir[:-1]
    plan_output_dir += "_ppo_rew"
    Path(plan_output_dir).mkdir(parents=True, exist_ok=True)
    
    # * Setup the metrics as reward function (2)
    compute_metrics_compare_ref = partial(
        compute_planning_metrics_compare_with_ref,
        config_cache_dict=config_cache_dict,
        config_dir=config_dir,
        if_average=False,
    )
    
    compute_metrics_using_val = partial(
        compute_planning_metrics_using_val,
        val_fp=val_fp,
        domain_fp_pattern=domain_fp_pattern,
        problem_instance_fp_pattern=problem_instance_fp_pattern,
        plan_output_dir=plan_output_dir,
        config_dir=config_dir,
        config_cache_dict=config_cache_dict,
        split_name=split_name_str,
        if_average=False,
    )
    # these two functions require eval_preds as input, it is a OmegaConf object
    
    partial_reward_function = partial(
        my_reward_function,
        compute_metrics_compare_ref=compute_metrics_compare_ref,
        compute_metrics_using_val=compute_metrics_using_val,
    ) # this function requires eval_preds as input
    
    
    # ! -- End of Setup Metrics as Reward Function
    
    # ! -- Config init 
    sft_model_path = model_name
    num_ppo_epochs = general_cfg['num_ppo_epochs']
    num_mini_batches = general_cfg['num_mini_batches']
    learning_rate = general_cfg['learning_rate']
    per_device_train_batch_size = ppo_cfg['per_device_train_batch_size']
    gradient_accumulation_steps = ppo_cfg['gradient_accumulation_steps']
    total_episodes = general_cfg['total_episodes']
    model_name_or_path = general_cfg['model_name_or_path']
    local_rollout_forward_batch_size = int(per_device_train_batch_size * gradient_accumulation_steps)
    missing_eos_penalty = general_cfg['missing_eos_penalty']
    stop_token = general_cfg['stop_token']
    ddp_timeout = general_cfg['ddp_timeout']
    response_length = ppo_cfg['response_length']
    save_steps = general_cfg['save_steps']
    save_total_limit = general_cfg['save_total_limit']
    
    mode = general_cfg['mode']
    if mode == "ppo":
        parser = HfArgumentParser((PPOv2Config, ModelConfig))
    elif mode == "rloo":
        parser = HfArgumentParser((RLOOConfig, ModelConfig))
    else:
        raise ValueError(f"mode {mode} is not supported")
    ppo_training_args, ppo_model_config = parser.parse_args_into_dataclasses()
    # update the args and configs 
    ppo_training_args.output_dir = ppo_checkpoint_output_dir
    ppo_training_args.per_device_train_batch_size = per_device_train_batch_size
    ppo_training_args.per_device_eval_batch_size = per_device_train_batch_size
    ppo_training_args.gradient_accumulation_steps = gradient_accumulation_steps
    ppo_training_args.learning_rate = learning_rate
    ppo_training_args.logging_dir = os.path.join(Path(ppo_checkpoint_output_dir).parent, f"logs_{config_type_name}")
    Path(ppo_training_args.logging_dir).mkdir(parents=True, exist_ok=True)
    ppo_training_args.seed = seed
    ppo_training_args.run_name = f"ppo_{config_type_name}"
    ppo_training_args.sft_model_path = sft_model_path
    ppo_training_args.num_ppo_epochs = num_ppo_epochs
    ppo_training_args.num_mini_batches = num_mini_batches
    ppo_training_args.total_episodes = total_episodes
    
    ppo_model_config.model_name_or_path = model_name_or_path
    ppo_model_config.trust_remote_code = False
    ppo_training_args.local_rollout_forward_batch_size = local_rollout_forward_batch_size
    ppo_training_args.missing_eos_penalty = missing_eos_penalty
    ppo_training_args.stop_token = stop_token
    ppo_training_args.ddp_timeout = ddp_timeout
    ppo_training_args.response_length = response_length
    ppo_training_args.dataset_num_proc = 6
    
    # save checkpoints config
    ppo_training_args.save_steps = save_steps 
    ppo_training_args.save_total_limit = save_total_limit 
    
    
    # ! Load Dataset 
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=ppo_model_config.trust_remote_code,
    )
    
    
    test_split_name = SPLIT_NAMES.test_longer_horizon.name
    dataset_config = model_config_type
    dataset_path = os.path.join(os.environ["WORKING_DIR"], general_cfg['huggingface_dataset_dir'])
    
    ppo_data_cache_dir = os.path.join(os.environ["WORKING_DIR"], general_cfg['ppo_data_cache_dir'])
    ppo_data_cache_dir = os.path.join(ppo_data_cache_dir, f"{model_config_type}_{test_split_name}_ppo_dataset_dir")
    ppo_data_cache_dir_eval = ppo_data_cache_dir + "_eval"
    
    Path(ppo_data_cache_dir).parent.mkdir(parents=True, exist_ok=True)
        
    # the dataset should contain the following columns:
    # input_ids, domain, instance_id, prompt_id, raw_plan
    if accelerator.is_main_process:
        if not os.path.exists(ppo_data_cache_dir):
            previous_dataset = load_dataset(
                dataset_path,
                dataset_config,
                split=test_split_name,
            )
            
            accelerator.print("Load Previous Dataset")
            
            # for each domain, we only sample example_per_domain examples
            
            updated_data_lst = []
            updated_data_lst_eval = []
            unique_domain_lst = list(set(previous_dataset['domain']))
            random.seed(seed)
            domain_indexes_lst = [] 
            for domain_name in unique_domain_lst:
                domain_indexes = [i for i, x in enumerate(previous_dataset['domain']) if x == domain_name]
                # randomly select example_per_domain*2 examples
                domain_indexes_lst.append(random.sample(domain_indexes, int(example_per_domain * 2)))
                
            for domain_indexes in domain_indexes_lst:
                flip_flag = False
                for index in domain_indexes:
                    flip_flag = not flip_flag
                    if flip_flag:
                        updated_data_lst.append(previous_dataset[index])
                    else:
                        updated_data_lst_eval.append(previous_dataset[index])
           
                
            # convert updated_data_lst back to a dataset
            from datasets import Dataset
            updated_dataset = Dataset.from_list(updated_data_lst)
            updated_dataset_eval = Dataset.from_list(updated_data_lst_eval)
            
            
            # ! the dataset should contain the following columns:
            # input_ids, domain, instance_id, prompt_id, raw_plan
            def prepare_dataset(dataset, tokenizer):
                
                def tokenize(element):
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": element['instruction'] + '\n' + element['input']},
                    ]
                    input_ids = tokenizer.apply_chat_template(
                        messages,
                        padding=False,
                        add_generation_prompt=True,
                    )
                    return {"input_ids": input_ids, "lengths": len(input_ids)}
                
                wanted_columns = ['input_ids', 'lengths', 'domain', 'instance_id', 'prompt_id', 'raw_plan']
                remove_columns = [col for col in dataset.column_names if col not in wanted_columns]
                return dataset.map(
                    tokenize,
                    remove_columns=remove_columns,
                    num_proc=ppo_training_args.dataset_num_proc,
                    
                )
            
            # map the dataset
            updated_dataset = prepare_dataset(updated_dataset, tokenizer)
            updated_dataset_eval = prepare_dataset(updated_dataset_eval, tokenizer)
            
            # save the dataset
            updated_dataset.save_to_disk(ppo_data_cache_dir)
            updated_dataset_eval.save_to_disk(ppo_data_cache_dir_eval)
            accelerator.print("Save Updated Dataset")
            del previous_dataset
            del updated_dataset
            del updated_dataset_eval
            
        wandb.init(**wandb_kwargs) # ! Wandb init
        
   
    accelerator.wait_for_everyone()
    # Load the dataset (cached)
    assert os.path.exists(ppo_data_cache_dir), f"Dataset not found at {ppo_data_cache_dir}"
    ppo_training_dataset = load_from_disk(ppo_data_cache_dir)
    ppo_training_dataset_eval = load_from_disk(ppo_data_cache_dir_eval)
    
    assert ppo_training_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"        
        
    
    # ! PPO Training 
    # if accelerator.is_main_process:
    #     shutil.rmtree(ppo_training_args.output_dir, ignore_errors=True)
        
   

    lora_rank = ppo_cfg['lora_rank']
    lora_alpha = ppo_cfg['lora_alpha']
    lora_dropout = general_cfg['lora_dropout']
    if_qlora = ppo_cfg['if_qlora']
    
    
    lora_config = LoraConfig( # Lora Model
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
    )
    
    model_kwargs = dict(
        trust_remote_code=True
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype= torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    if if_qlora:
        model_kwargs.update(quantization_config=quantization_config)
        
        
    # TODO I do not understand why using official code will drastically slow down the training process
    # ! --- official code 
    # policy = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16,
    #     device_map = {"": accelerator.process_index},
    #     low_cpu_mem_usage = True,
    #     peft_config= lora_config,
    #     **model_kwargs
    # )
    
    # if if_qlora:
    #     policy = prepare_model_for_kbit_training(policy, use_gradient_checkpointing=False)
    #     policy = get_peft_model(policy, lora_config)
    # else:
    #     policy.enable_input_require_grads()
    #     policy = get_peft_model(policy, lora_config)
    # ! --- end of official code
    
    # ! -- Custom code
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        # device_map = {"": accelerator.process_index},
        peft_config= lora_config,
        **model_kwargs,
    )
    policy.enable_input_require_grads()
    # ! -- End of Custom code
        
    ref_model_kwargs = deepcopy(model_kwargs) 
    
    # if if_qlora:
    # if config_type_name not in ['accu_t2_t4']:
    ref_model_kwargs.update(quantization_config=quantization_config) # save memory
    
    ref_policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        # device_map = {"": accelerator.process_index},
        # low_cpu_mem_usage = True,
        **ref_model_kwargs
    )
    ref_policy.eval()
        
    if mode == "ppo":
        value_model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", num_labels=1,trust_remote_code=True)
    
    
    
    # sync GPUs
    accelerator.wait_for_everyone() 
    
    ################
    # Training
    ################
    # print trainable parameters
    accelerator.print("Trainable parameters:")
    if accelerator.is_main_process:
        if getattr(policy, "print_trainable_parameters", None):
            policy.print_trainable_parameters()
    
    if mode == "ppo":
        trainer = CustomRewardPPOv2Trainer(
        config=ppo_training_args,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_function=partial_reward_function,
        value_model=value_model,
        train_dataset=ppo_training_dataset,
        eval_dataset=ppo_training_dataset_eval,
    )
    elif mode == "rloo":
        trainer = CustomRewardRLOOTrainer(
            config=ppo_training_args,
            tokenizer=tokenizer,
            policy=policy,
            ref_policy=ref_policy,
            reward_function=partial_reward_function,
            train_dataset=ppo_training_dataset,
            eval_dataset=ppo_training_dataset_eval,
        )
        
    accelerator.print("Start Training")
    resume_checkpoint_path = general_cfg['resume_checkpoint_path']
    
    if resume_checkpoint_path != "":
        trainer.train(
            resume_from_checkpoint=os.path.join(os.environ["WORKING_DIR"], resume_checkpoint_path)
        )
    else:
        trainer.train()

    # # Save the model (Deprecated as we can directly load the checkpoint)
    # if accelerator.is_main_process:
    #     trainer.save_model(ppo_training_args.output_dir)
    
    trainer.generate_completions()
    
    # ! End of PPO Training
    if accelerator.is_main_process:
        if general_cfg['wandb_record']:
            wandb.finish()
    
    


if __name__ == "__main__":
    import json
    import logging.config
    from kedro.config import OmegaConfigLoader
    from kedro.framework.project import settings
    import os 
    
    deepspeed.init_distributed(timeout=timedelta(hours=6))
    
    conf_path = os.path.join(os.environ["WORKING_DIR"], settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path, env= "base")
    conf_params = conf_loader['parameters']
    
    config_type_name = os.environ.get("EVAL_CONFIG_TYPE", "accu_t1_t4")
    print(f"config_type_name for planning evaluation: {config_type_name}")
    time.sleep(2)
    
    ppo_cfg = conf_params["further_ppo_training_params"][config_type_name + "_params"]
    general_cfg = conf_params["further_ppo_training_general_params"]
    
    print("Please using accelerate launch to run this pipeline")
    
    num_process = len(os.environ.get("CUDA_VISIBLE_DEVICES", "1").split(","))
    print("Num process:", num_process)
    ppo_training(ppo_cfg, general_cfg)