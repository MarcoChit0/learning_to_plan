"""
This is a boilerplate pipeline 'language_model_planning_evaluation'
generated using Kedro 0.19.6
"""

from functools import partial
from icecream import ic


ANCHOR_PLAN_SEC = "My plan is as follows:\n<PLAN>\n"

from datasets import load_dataset

from better_language_model_for_plan_generation.utils.constant import *
from better_language_model_for_plan_generation.utils.eval_generated_plan_text import *
from better_language_model_for_plan_generation.utils.custom_compute_metrics import compute_planning_metrics_compare_with_ref, compute_planning_metrics_using_val

from better_language_model_for_plan_generation.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

import random
import os
import numpy as np
from glob import glob
from natsort import natsorted
from pddl.parser.problem import ProblemParser
import re
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime
import wandb
import torch
from functools import partial
from omegaconf import OmegaConf 
import pandas as pd
from multiprocessing import Process, Queue
# from multiprocessing.dummy import Process, Queue # debug
from transformers import TextStreamer
from better_language_model_for_plan_generation.utils.action_hint import obtain_action_hint_text
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
import time
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import AutoModelForCausalLMWithValueHead

# merge model ref: https://huggingface.co/docs/trl/en/use_model
# other important ref to load checkpoint: https://github.com/Lightning-AI/pytorch-lightning/issues/17704
# https://github.com/Lightning-AI/pytorch-lightning/discussions/12132
# https://github.com/Lightning-AI/pytorch-lightning/issues/10964

# only cuda available then from unsloth import FastLanguageModel # * import the model here to avoid the error if no GPU is available
if torch.cuda.is_available():
    from transformers import AutoModelForCausalLM, AutoTokenizer


def my_eval_func(evaluation_process_queue, evaluation_output_queue, compute_metrics_using_val, compute_metrics_compare_ref, config_type,):
    if config_type in [CONFIG_TYPES.t0.name, CONFIG_TYPES.accu_t1_t4.name]:
        split_dict = dict()
        while True:
            eval_preds = evaluation_process_queue.get()
            if eval_preds is None:
                break
            # split the eval_preds by domain_name
            predictions = eval_preds["predictions"]
            inputs = eval_preds["inputs"]
            domain_names = inputs[FEATURE_CLS.domain.name]
            assert len(domain_names) <= len(predictions)
            if len(domain_names) < len(predictions):
                k_size = len(predictions) // len(domain_names)
            else:
                k_size = 1
                
            for i, domain_name in enumerate(domain_names):
                if domain_name not in split_dict:
                    split_dict[domain_name] = {
                        'predictions': [],
                        'inputs': {
                            FEATURE_CLS.domain.name: [],
                            FEATURE_CLS.instance_id.name: [],
                            FEATURE_CLS.prompt_id.name: [],
                            FEATURE_CLS.raw_plan.name: [],
                        }
                    }
                split_dict[domain_name]['predictions'].extend(predictions[i*k_size: (i+1)*k_size])
                split_dict[domain_name]['inputs'][FEATURE_CLS.domain.name].append(inputs[FEATURE_CLS.domain.name][i])
                split_dict[domain_name]['inputs'][FEATURE_CLS.instance_id.name].append(inputs[FEATURE_CLS.instance_id.name][i])
                split_dict[domain_name]['inputs'][FEATURE_CLS.prompt_id.name].append(inputs[FEATURE_CLS.prompt_id.name][i])
                split_dict[domain_name]['inputs'][FEATURE_CLS.raw_plan.name].append(inputs[FEATURE_CLS.raw_plan.name][i])
                
        # average the eval_score_lst but for each domain_name
        eval_score_avg = [] 
        for domain_name, vals in split_dict.items():
            # form OmegaConf dict again 
            eval_preds = OmegaConf.create({
                "predictions": vals['predictions'],
                "inputs": vals['inputs']
            })
            
            val_binary_score = compute_metrics_using_val(eval_preds=eval_preds)['planning_metric']
            lcs_score = compute_metrics_compare_ref(eval_preds=eval_preds, if_lccs=False)['planning_metric']
            lccs_score = compute_metrics_compare_ref(eval_preds=eval_preds, if_lccs=True)['planning_metric']
            eval_score_avg.append({
                "val_binary_score": val_binary_score,
                "lcs_score": lcs_score,
                "lccs_score": lccs_score,
                "domain_name": domain_name
            })
            
        evaluation_output_queue.put(eval_score_avg)
    
    
    else:
        
        eval_score_lst = []
        while True:
            eval_preds = evaluation_process_queue.get()
            if eval_preds is None:
                break
            # three metrics (VAL binary check, LCS, LCCS)
            val_binary_score = compute_metrics_using_val(eval_preds=eval_preds)['planning_metric']
            lcs_score = compute_metrics_compare_ref(eval_preds=eval_preds, if_lccs=False)['planning_metric']
            lccs_score = compute_metrics_compare_ref(eval_preds=eval_preds, if_lccs=True)['planning_metric']
            eval_score_lst.append({
                "val_binary_score": val_binary_score,
                "lcs_score": lcs_score,
                "lccs_score": lccs_score,
            })
            
        # average the eval_score_lst
        
        eval_score_avg = {  
            "val_binary_score": sum([score["val_binary_score"] for score in eval_score_lst]) / len(eval_score_lst),
            "lcs_score": sum([score["lcs_score"] for score in eval_score_lst]) / len(eval_score_lst),
            "lccs_score": sum([score["lccs_score"] for score in eval_score_lst]) / len(eval_score_lst),
        }
        evaluation_output_queue.put(eval_score_avg)


# * so far we do not support multi-gpu inference, but you can check this https://github.com/geronimi73/accelerate_tricks/blob/main/inference_batched.py for multi-gpu inference
# ! NODE
def evaluate_language_model_planning(eval_cfg, general_cfg):

    # ! when we use llama_factory, we need to call `accelerate launch` rather than `kedro run`, so we need to save the config cfg to a local file and then re run the pipeline

    # Setup env vars
    HF_HOME = os.environ.get("HF_HOME")
    if HF_HOME is None: # in case the env var is not set
        os.environ["HF_HOME"] = os.path.join(
            os.environ["WORKING_DIR"], "data/01_raw/huggingface"
        )
        print(os.environ["HF_HOME"])
        Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

    datetime_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    config_type = CONFIG_TYPES.__members__[eval_cfg['config_type_name']].name
    
    if config_type in [CONFIG_TYPES.accu_t1_t3_t4.name, CONFIG_TYPES.accu_t1_t3.name]:
        no_repeat_ngram_size_value = 90
    elif config_type in [CONFIG_TYPES.accu_t4.name, CONFIG_TYPES.accu_t3.name]:
        no_repeat_ngram_size_value = 150
    else:
        no_repeat_ngram_size_value = 70

    checkpoint_dir = eval_cfg["checkpoint_dir"]
    debug_mode = general_cfg["debug_mode"]
    
    top_k = int(general_cfg['top_k'])
    # * swap top_k if os.environ['TOP_K_CONFIG'] is set
    if "TOP_K_CONFIG" in os.environ:
        top_k = int(os.environ['TOP_K_CONFIG'])
    
    token_top_k = int(general_cfg['token_top_k'])
    top_p = general_cfg['top_p']
    eval_batch_size = eval_cfg["eval_batch_size"]  // top_k # ! we need to divide the eval_batch_size by top_k to avoid the memory error
    if_one_shot = general_cfg['if_one_shot']

    if if_one_shot:
        eval_batch_size = eval_batch_size // 2 # we need to divide the eval_batch_size by 2 to avoid the memory error
        
    eval_batch_size = max(1, eval_batch_size) # ! we need to make sure the eval_batch_size is at least 1
    provide_action_hint = general_cfg["provide_action_hint"] # ! LLM may not accurately estimate the total number of steps in the plan, so we may give the first action to the model as a hint      
    if os.environ.get("ACTION_HINT_SIZE") is not None:
        action_hint_size = int(os.environ.get("ACTION_HINT_SIZE"))
    else:
        action_hint_size = general_cfg["action_hint_size"] 

    is_llama_factory = eval_cfg["is_llama_factory"]
    assert is_llama_factory, "We only support llama_factory for now"
    if checkpoint_dir == "":
        raise ValueError(f"The checkpoint dir is empty for config type: {config_type}")

    checkpoint_dir = os.path.join(os.environ['WORKING_DIR'], checkpoint_dir)

    max_new_tokens = general_cfg['seq_params'][config_type]['max_new_tokens'] # ! deprecated
    max_seq_length = general_cfg['seq_params'][config_type]['max_seq_length']
    max_seq_length += max_new_tokens
    if if_one_shot:
        # need to increase the max_seq_length
        max_seq_length += max_new_tokens
        
        
    # ! --- PPO and whether testing with ppo config
    ppo_checkpoint_dir = os.path.join(os.environ['WORKING_DIR'], eval_cfg["ppo_checkpoint_dir"])
    if_test_ppo_version = int(os.environ.get("TEST_PPO", 0)) == 1 
    if "lora_dropout" in general_cfg:
        lora_dropout = general_cfg["lora_dropout"]
    if "lora_rank" in eval_cfg:
        lora_rank = eval_cfg["lora_rank"]
    else:
        lora_rank = 1
    if "lora_alpha" in eval_cfg:
        lora_alpha = eval_cfg["lora_alpha"]
    else:
        lora_alpha = 2
        
    if is_llama_factory:
        # ! setup accelerate
        from accelerate import Accelerator
        from accelerate.utils import gather_object 
        import torch, time, json 
        # clean cache
        kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=6)) # * set the timeout to 6 hours
        accelerator = Accelerator( 
            kwargs_handlers=[
                kwargs
            ]
        )    

    # init wandb
    only_check_plan_executable = general_cfg["only_check_plan_executable"]
    only_check_goal_achievable = general_cfg["only_check_goal_achievable"]

    assert not (only_check_plan_executable and only_check_goal_achievable), "We cannot check both plan executable and goal achievable at the same time"
    
    if general_cfg["wandb_record"]:
        job_name = (
            f"Eval LLM on Plan Gen - Config: {eval_cfg['config_type_name']}Top K: {top_k} {'PPO Ver' if if_test_ppo_version else ''}{' Clever Hans '+str(action_hint_size) if action_hint_size >= 15 else ''}{' Only Executability (NEW)' if only_check_plan_executable else ''}{'Only Achievable (NEW)' if only_check_goal_achievable else ''}"
        )
        general_cfg["timestamp"] = datetime_timestamp
        merge_dict = {**general_cfg, **eval_cfg}
        tags_lst = ["planning_eval", f"llama_factory_{is_llama_factory}"]
        if if_test_ppo_version:
            tags_lst.append("ppo")
        wandb_kwargs = {
            "project": "ACCE LM for Plan Generation",
            "name": job_name,
            "config": merge_dict,
            "tags": tags_lst,
        }
        
        if not is_llama_factory or accelerator.is_main_process:
            wandb.init(**wandb_kwargs)
            
    # log the config
    ic(eval_cfg)
    ic(general_cfg)
    if debug_mode:
        debug_mode_num = general_cfg["debug_mode_num"]
        print("Debug Mode is ON\n\n")
    import torch
    # * Setup model
    if general_cfg["dtype"] == "bfloat16":
        dtype = torch.bfloat16
    elif general_cfg["dtype"] == "float16":
        dtype = torch.float16
    elif general_cfg["dtype"] == "float32":
        dtype = torch.float32

    model_name = checkpoint_dir # the model name is the checkpoint dir, the model will be loaded from the checkpoint dir
    test_split_list = [ # TODO debug 
        SPLIT_NAMES.test_same_domain.name,
        SPLIT_NAMES.test_longer_horizon.name,
        # SPLIT_NAMES.test_mystery_domain.name,
        SPLIT_NAMES.test_unseen_domain.name
    ]
    plan_output_dir = os.path.join(os.environ["WORKING_DIR"], general_cfg["plan_output_dir"])
    plan_output_dir = plan_output_dir.format(timestamp=datetime_timestamp)
    Path(plan_output_dir).mkdir(parents=True, exist_ok=True)

    if not is_llama_factory or accelerator.is_main_process:
        if config_type in [CONFIG_TYPES.t0.name, CONFIG_TYPES.accu_t1_t4.name]:
            eval_df = pd.DataFrame(columns=["test_case", "val_binary_score", "lcs_score", "lccs_score", "domain_name"])
        else:
            eval_df = pd.DataFrame(columns=["test_case", "val_binary_score", "lcs_score", "lccs_score"])

    # * Load Eval dataset

    dataset_path = os.path.join(os.environ["WORKING_DIR"], general_cfg["huggingface_dataset_dir"])
        
    eval_dataset_full = load_dataset(
        dataset_path,
        config_type, 
        split=test_split_list,
    )

    # ! -- Eval process --
    if not is_llama_factory:
        raise ValueError("We only support llama_factory for now")
        print("Single GPU (Unsloth Model) Inferencing")
        if torch.cuda.is_available():
            from unsloth import (
                FastLanguageModel,
            )  # * import the model here to avoid the error if no GPU is available
            print("Using Unsloth Model")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,    
            load_in_4bit=False,  # this is for 4bit quantization
        )
        model.config.use_cache = False # ! this is to avoid the error of "RuntimeError: CUDA out of memory" when using cache

        # clean cache
        torch.cuda.empty_cache()
        FastLanguageModel.for_inference(model) # * set the model to inference mode

        # - Print Stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        # * Iterate through the test dataset
        for split_name_str in tqdm(test_split_list, desc="Evaluating splits"):
            split_name = SPLIT_NAMES.__members__[split_name_str]
            config_cache_dict = dict() # * cache the config_content for compute_metrics

            compute_metrics_compare_ref = partial(
                compute_planning_metrics_compare_with_ref,
                config_cache_dict=config_cache_dict,
                config_dir=os.path.join(
                    os.environ["WORKING_DIR"], general_cfg["planning_domain_config_dir"]
                )
            )

            compute_metrics_using_val = partial(
                compute_planning_metrics_using_val,
                val_fp=os.path.join(os.environ["WORKING_DIR"], general_cfg["val_fp"]),
                domain_fp_pattern=os.path.join(
                    os.environ["WORKING_DIR"], general_cfg["domain_fp_pattern"]
                ),
                problem_instance_fp_pattern=os.path.join(
                    os.environ["WORKING_DIR"], general_cfg["problem_instance_fp_pattern"]
                ),
                plan_output_dir=plan_output_dir,
                config_dir=os.path.join(
                    os.environ["WORKING_DIR"], general_cfg["planning_domain_config_dir"]
                ),
                config_cache_dict=config_cache_dict,
                split_name=split_name.name,
            )

            # * -- loop the dataset
            eval_dataset_split = eval_dataset_full[test_split_list.index(split_name.name)]

            if config_type != CONFIG_TYPES.t0.name:
                # ! we filter to only have data with column "prompt_id" value == 0
                eval_dataset_split = eval_dataset_split.filter(lambda x: x[FEATURE_CLS.prompt_id.name] == 0)

            # ! DEBUG: we only test some examples
            if debug_mode:
                eval_dataset_split = eval_dataset_split.select(list(range(debug_mode_num)))

            eval_input_str_temp_list = []
            eval_inputs_domain_lst = []
            eval_inputs_instance_id_lst = []
            eval_inputs_prompt_id_lst = []
            eval_inputs_raw_plan_lst = []
            test_entry_hint_text_lst = [] 

            # create evaluation process queue
            evaluation_process_queue = Queue(maxsize=1024)
            evaluation_output_queue = Queue(maxsize=1)
            evaluation_process = Process(target=my_eval_func, args=(evaluation_process_queue, evaluation_output_queue, compute_metrics_using_val, compute_metrics_compare_ref))

            evaluation_process.start()

            # first loop to get domains
            domain_idx_dict = dict()
            for test_entry_idx, test_entry in enumerate(eval_dataset_split):
                domain_name = test_entry[FEATURE_CLS.domain.name]
                if domain_name not in domain_idx_dict:
                    domain_idx_dict[domain_name] = []
                domain_idx_dict[domain_name].append(test_entry_idx)

            has_tested = False 

            for test_entry_idx, test_entry in enumerate(tqdm(eval_dataset_split, desc=f"Evaluating split: {split_name.name}")):
                prompt = ""
                if if_one_shot: # we will have one shot example provided 
                    domain_name = test_entry[FEATURE_CLS.domain.name]
                    while True:
                        one_shot_domain_idx = random.choice(domain_idx_dict[domain_name])
                        if one_shot_domain_idx != test_entry_idx:
                            break
                    one_shot_test_entry = eval_dataset_split[one_shot_domain_idx]
                    prompt += (
                        one_shot_test_entry[FEATURE_CLS.full_text_prompt.name]
                        + "\n"
                    )

                prompt += (
                    test_entry[FEATURE_CLS.domain_text.name]
                    + test_entry[FEATURE_CLS.problem_init_text.name]
                    + test_entry[FEATURE_CLS.problem_goal_text.name]
                    + ANCHOR_PLAN_SEC
                )
                # * add the action hint
                if provide_action_hint and split_name_str not in [SPLIT_NAMES.test_mystery_domain.name, SPLIT_NAMES.test_unseen_domain.name]:
                    gt_plan_text = test_entry[FEATURE_CLS.plan_text.name]
                    test_entry_hint_text = obtain_action_hint_text(gt_plan_text, action_hint_size, is_llama_factory, ANCHOR_PLAN_SEC)
                    prompt += test_entry_hint_text
                    test_entry_hint_text_lst.append(test_entry_hint_text)

                if not has_tested: # we try it using streaming inference
                    print("Testing the model using streaming inference")
                    has_tested = True
                    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
                    text_streamer = TextStreamer(tokenizer)
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        streamer=text_streamer,
                        no_repeat_ngram_size=no_repeat_ngram_size_value,
                    )

                eval_input_str_temp_list.append(prompt)
                eval_inputs_domain_lst.append(test_entry[FEATURE_CLS.domain.name])
                eval_inputs_instance_id_lst.append(test_entry[FEATURE_CLS.instance_id.name])
                eval_inputs_prompt_id_lst.append(test_entry[FEATURE_CLS.prompt_id.name])
                eval_inputs_raw_plan_lst.append(test_entry[FEATURE_CLS.raw_plan.name])
                if len(eval_input_str_temp_list) == eval_batch_size or test_entry == eval_dataset_split[-1]:
                    inputs = tokenizer(eval_input_str_temp_list,
                                       padding=True,
                                       return_tensors="pt").to('cuda')

                    if top_k > 1:
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            top_k=token_top_k,
                            top_p=top_p,
                            num_return_sequences=top_k,
                            no_repeat_ngram_size=no_repeat_ngram_size_value,
                        )
                    else:
                        outputs = model.generate(**inputs, 
                                    max_new_tokens= max_new_tokens,
                                    no_repeat_ngram_size=no_repeat_ngram_size_value,
                                    )
                    input_lengths = inputs["input_ids"].shape[1]
                    # slice the outputs to get the outputs
                    outputs = outputs[:, input_lengths:]

                    decoded_output_lst = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    # * add back the anchor_plan_sec or the action hint
                    temp_decoded_output_lst = [] 
                    # ! if top_k > 1, the need to replicate the test_entry_hint_text_lst
                    if top_k > 1:
                        temp_test_entry_hint_text_lst = []
                        for hint_text in test_entry_hint_text_lst:
                            temp_test_entry_hint_text_lst.extend([hint_text] * top_k)
                        test_entry_hint_text_lst = temp_test_entry_hint_text_lst

                    for i, decoded_output in enumerate(decoded_output_lst):
                        if provide_action_hint and split_name_str not in [SPLIT_NAMES.test_mystery_domain.name, SPLIT_NAMES.test_unseen_domain.name]:
                            temp_decoded_output_lst.append(
                                ANCHOR_PLAN_SEC + test_entry_hint_text_lst[i] + decoded_output
                            )
                        else:
                            temp_decoded_output_lst.append(
                                ANCHOR_PLAN_SEC + decoded_output
                            )
                    decoded_output_lst = temp_decoded_output_lst

                    # construct the eval_preds using OmegaConf
                    eval_preds = OmegaConf.create({
                        "predictions": decoded_output_lst,
                        "inputs": {
                            FEATURE_CLS.domain.name: eval_inputs_domain_lst,
                            FEATURE_CLS.instance_id.name: eval_inputs_instance_id_lst,
                            FEATURE_CLS.prompt_id.name: eval_inputs_prompt_id_lst,
                            FEATURE_CLS.raw_plan.name: eval_inputs_raw_plan_lst,
                        }
                    })
                    eval_input_str_temp_list = [] 
                    eval_inputs_domain_lst = []
                    eval_inputs_instance_id_lst = []
                    eval_inputs_prompt_id_lst = []
                    eval_inputs_raw_plan_lst = []
                    test_entry_hint_text_lst = []
                    # have another process for evaluation
                    evaluation_process_queue.put(eval_preds)

            evaluation_process_queue.put(None)      
            eval_score_avg = evaluation_output_queue.get()
            # join the process to avoid zombie process
            evaluation_process.join()

            eval_df.loc[len(eval_df)] = [split_name.name, eval_score_avg["val_binary_score"], eval_score_avg["lcs_score"], eval_score_avg["lccs_score"]]

    else: # llama factory supports multi-gpu inference
        # ! we are going to use inference batch to speed up the inference
        print("Llama Factory Inferencing")

        message= [f"Hello this is GPU {accelerator.process_index}"]
        messages=gather_object(message)
        accelerator.print(messages)

        if ppo_checkpoint_dir != "" and if_test_ppo_version:
            accelerator.print("Loading PPO model for Evaluation")
            time.sleep(2)
            lora_config = LoraConfig( # Lora Model
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                task_type="CAUSAL_LM",
            )
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_name,
                device_map = {"": accelerator.process_index},
                peft_config= lora_config,
            )
            # load from the ppo checkpoint
            model = load_state_dict_from_zero_checkpoint(model, ppo_checkpoint_dir)
            # move to the device
            model = model.to('cuda')
            # convert to bf16
            model = model.to(dtype=torch.bfloat16)
            model = model.eval()
            # clean cuda cache
            torch.cuda.empty_cache()
            
        else:
            accelerator.print("Loading LLM model for Evaluation")
            time.sleep(2)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": accelerator.process_index},
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # batch, left pad (for inference), and tokenize
        def prepare_prompts(eval_dataset_split, tokenizer, if_one_shot, domain_idx_dict, batch_size=12, ):
            nonlocal split_name_str
            num_samples = len(eval_dataset_split)
            num_batches = (num_samples + batch_size - 1) // batch_size  # Round up division
            batches = []

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                batch = eval_dataset_split.select(range(start_idx, end_idx))
                batches.append(batch)

            tokenizer.padding_side="left"  
            if accelerator.is_main_process:
                batches = tqdm(batches, desc=f"Inference for {split_name_str}")  
            for prompt_batch in batches:
                inputs_lst = [] 
                domain_names = []
                instance_ids = []
                prompt_ids = []
                raw_plan_lst = []
                text_entry_hint_text_lst = []
                for test_entry in prompt_batch:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."}
                    ] # default system prompt
                    if if_one_shot: # we will have one shot example provided
                        domain_name = test_entry[FEATURE_CLS.domain.name]
                        while True:
                            one_shot_domain_idx = random.choice(domain_idx_dict[domain_name])
                            if one_shot_domain_idx != test_entry_idx:
                                break
                        one_shot_test_entry = eval_dataset_split[one_shot_domain_idx]
                        messages.append(
                            {"role": "user", "content": one_shot_test_entry["instruction"] + "\n" + one_shot_test_entry["input"]}
                        )
                        messages.append({
                            "role": "assistant",
                            "content": one_shot_test_entry["output"]
                        })
                    messages.append(
                        {"role": "user", "content": test_entry["instruction"] + "\n" + test_entry["input"]}
                    )
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    # * add the action hint
                    if provide_action_hint and (split_name_str not in [SPLIT_NAMES.test_mystery_domain.name, SPLIT_NAMES.test_unseen_domain.name] or config_type in [CONFIG_TYPES.t0.name, CONFIG_TYPES.accu_t1.name, CONFIG_TYPES.t0_t5.name, CONFIG_TYPES.accu_t1_t4.name, CONFIG_TYPES.accu_t2_t4.name, CONFIG_TYPES.accu_t2.name]):
                        gt_plan_text = test_entry["output"]
                        test_entry_hint_text = obtain_action_hint_text(gt_plan_text, action_hint_size, is_llama_factory, ANCHOR_PLAN_SEC)
                        text += test_entry_hint_text # we automatically add the ANCHOR_PLAN_SEC already
                        text_entry_hint_text_lst.append(test_entry_hint_text)

                    inputs_lst.append(text)
                    domain_names.append(test_entry[FEATURE_CLS.domain.name])
                    instance_ids.append(test_entry[FEATURE_CLS.instance_id.name])
                    prompt_ids.append(test_entry[FEATURE_CLS.prompt_id.name])
                    raw_plan_lst.append(test_entry[FEATURE_CLS.raw_plan.name])

                yield    [
                        tokenizer(
                            inputs_lst,
                            return_tensors="pt",
                            padding="longest",
                            truncation=False,
                            pad_to_multiple_of=8,
                            add_special_tokens=False,
                        ).to('cuda'),
                        
                        domain_names,
                        instance_ids,
                        prompt_ids,
                        raw_plan_lst,
                        text_entry_hint_text_lst,
                    ]
            tokenizer.padding_side="right"

        # - Print Stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        # * Iterate through the test dataset
        for split_name_str in test_split_list:
            split_name = SPLIT_NAMES.__members__[split_name_str]
            
            config_cache_dict = dict() # * cache the config_content for compute_metrics

            compute_metrics_compare_ref = partial(
                compute_planning_metrics_compare_with_ref,
                config_cache_dict=config_cache_dict,
                config_dir=os.path.join(
                    os.environ["WORKING_DIR"], general_cfg["planning_domain_config_dir"]
                )
            )

            compute_metrics_using_val = partial(
                compute_planning_metrics_using_val,
                val_fp=os.path.join(os.environ["WORKING_DIR"], general_cfg["val_fp"]),
                domain_fp_pattern=os.path.join(
                    os.environ["WORKING_DIR"], general_cfg["domain_fp_pattern"]
                ),
                problem_instance_fp_pattern=os.path.join(
                    os.environ["WORKING_DIR"], general_cfg["problem_instance_fp_pattern"]
                ),
                plan_output_dir=plan_output_dir,
                config_dir=os.path.join(
                    os.environ["WORKING_DIR"], general_cfg["planning_domain_config_dir"]
                ),
                config_cache_dict=config_cache_dict,
                split_name=split_name.name,
                only_check_plan_executable = only_check_plan_executable,
                only_check_goal_achievable = only_check_goal_achievable, 
            )
            if only_check_plan_executable:
                print("WARNING! We are only checking the plan executable, will not check if plan reaches the goal state")
            if only_check_goal_achievable:
                print("WARNING! We are only checking if the goal state is achievable, will not check if plan is executable")

            # * -- loop the dataset
            eval_dataset_split_full = eval_dataset_full[test_split_list.index(split_name.name)]
            if config_type != CONFIG_TYPES.t0.name:
                # ! we filter to only have data with column "prompt_id" value == 0
                eval_dataset_split_full = eval_dataset_split_full.filter(lambda x: x[FEATURE_CLS.prompt_id.name] == 0)
            # ! DEBUG: we only test some examples
            if debug_mode:
                eval_dataset_split_full = eval_dataset_split_full.select(list(range(debug_mode_num)))

            # create evaluation process queue
            if accelerator.is_main_process:
                evaluation_process_queue = Queue(maxsize=1024)
                evaluation_output_queue = Queue(maxsize=1)
                my_eval_func_partial = partial(my_eval_func, config_type=config_type)
                evaluation_process = Process(target=my_eval_func_partial, args=(evaluation_process_queue, evaluation_output_queue, compute_metrics_using_val, compute_metrics_compare_ref))

                evaluation_process.start()

            # sync GPUs
            accelerator.wait_for_everyone()  
            try:
                with accelerator.split_between_processes(eval_dataset_split_full) as eval_dataset_split:
                    # first loop to get domains
                    domain_idx_dict = dict()
                    for test_entry_idx, test_entry in enumerate(eval_dataset_split):
                        domain_name = test_entry[FEATURE_CLS.domain.name]
                        if domain_name not in domain_idx_dict:
                            domain_idx_dict[domain_name] = []
                        domain_idx_dict[domain_name].append(test_entry_idx)

                    results = dict(
                        eval_inputs_domain_lst = [],
                        eval_inputs_instance_id_lst = [],
                        eval_inputs_prompt_id_lst = [],
                        eval_inputs_raw_plan_lst = [],
                        outputs=[],
                        counts = 0,
                        
                    )

                    prompt_batches = prepare_prompts(
                        eval_dataset_split,
                        tokenizer,
                        if_one_shot,
                        domain_idx_dict,
                        batch_size=eval_batch_size,
                    )
                    
                        
                    has_tested = False
                    
                    for prompts_tokenized, domain_names, instance_ids, prompt_ids, raw_plan_lst, test_entry_hint_text_lst in prompt_batches:

                        if top_k > 1:
                            outputs_tokenized = model.generate(
                                **prompts_tokenized,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                top_k=token_top_k,
                                top_p=top_p,
                                num_return_sequences=top_k,
                                no_repeat_ngram_size=no_repeat_ngram_size_value,
                            )

                        else:
                            outputs_tokenized = model.generate(
                                **prompts_tokenized,
                                max_new_tokens=max_new_tokens,
                                no_repeat_ngram_size=no_repeat_ngram_size_value,
                            )

                        input_lengths = prompts_tokenized["input_ids"].shape[1]
                        # slice the outputs to get the outputs
                        outputs_tokenized = outputs_tokenized[:, input_lengths:]    

                        decoded_output_lst = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)

                        # * add back the anchor_plan_sec or the action hint
                        # ! if top_k > 1, the need to replicate the test_entry_hint_text_lst
                        if top_k > 1:
                            temp_test_entry_hint_text_lst = []
                            for hint_text in test_entry_hint_text_lst:
                                temp_test_entry_hint_text_lst.extend([hint_text] * top_k)
                            test_entry_hint_text_lst = temp_test_entry_hint_text_lst

                        if provide_action_hint and (split_name_str not in [SPLIT_NAMES.test_mystery_domain.name, SPLIT_NAMES.test_unseen_domain.name] or config_type in [CONFIG_TYPES.t0.name, CONFIG_TYPES.accu_t1.name, CONFIG_TYPES.t0_t5.name, CONFIG_TYPES.accu_t1_t4.name, CONFIG_TYPES.accu_t2_t4.name, CONFIG_TYPES.accu_t2.name]):
                            temp_decoded_output_lst = []
                            for i, decoded_output in enumerate(decoded_output_lst):
                                temp_decoded_output_lst.append(
                                    test_entry_hint_text_lst[i] + decoded_output
                                )
                            decoded_output_lst = temp_decoded_output_lst
                        
                        if accelerator.is_main_process and not has_tested: # we try printing the first batch
                            has_tested = True
                            print("Printing the first batch's first output")
                            print(decoded_output_lst[0])
                            

                        results["outputs"].extend(decoded_output_lst)
                        results["eval_inputs_domain_lst"].extend(domain_names)
                        results["eval_inputs_instance_id_lst"].extend(instance_ids)
                        results["eval_inputs_prompt_id_lst"].extend(prompt_ids)
                        results["eval_inputs_raw_plan_lst"].extend(raw_plan_lst)
                        results["counts"] += len(decoded_output_lst)

                    results = [ results ] # transform to list, otherwise gather_object() will not collect correctly

                # wait for all processes to finish
                print(f"GPU {accelerator.process_index} finished inference.")
                accelerator.wait_for_everyone()
                results_gathered = gather_object(results)

                if accelerator.is_main_process:
                    for results in results_gathered:
                        eval_preds = OmegaConf.create({
                            "predictions": results["outputs"],
                            "inputs": {
                                FEATURE_CLS.domain.name: results["eval_inputs_domain_lst"],
                                FEATURE_CLS.instance_id.name: results["eval_inputs_instance_id_lst"],
                                FEATURE_CLS.prompt_id.name: results["eval_inputs_prompt_id_lst"],
                                FEATURE_CLS.raw_plan.name: results["eval_inputs_raw_plan_lst"],
                            }
                        })
                        evaluation_process_queue.put(eval_preds)

                    evaluation_process_queue.put(None)
                    eval_score_avg = evaluation_output_queue.get()
                    # join the process to avoid zombie process
                    evaluation_process.join()

                    if config_type in [CONFIG_TYPES.t0.name, CONFIG_TYPES.accu_t1_t4.name]:
                        for eval_score_single in eval_score_avg: # split name example test_same_domain
                            eval_df.loc[len(eval_df)] = [split_name.name, eval_score_single["val_binary_score"], eval_score_single["lcs_score"], eval_score_single["lccs_score"], eval_score_single["domain_name"]]
                    else:
                        eval_df.loc[len(eval_df)] = [split_name.name, eval_score_avg["val_binary_score"], eval_score_avg["lcs_score"], eval_score_avg["lccs_score"]]

                    total_counts = sum([results["counts"] for results in results_gathered])
                    print(f"Total tested entry counts: {total_counts}")
                    print("Print Eval_DF")
                    print(eval_df)
            except Exception as e:
                print(f"GPU {accelerator.process_index} failed with error: {e}")
                if accelerator.is_main_process:
                    csv_save_path = os.path.join(os.environ["WORKING_DIR"], f"data/08_reporting/language_model_planning_evaluation/{eval_cfg['config_type_name']}_planning_evaluation_results{'_lla_factory' if is_llama_factory else ''}_top_k_{top_k}{'_ppo' if if_test_ppo_version else ''}{'_debug' if debug_mode else ''}_interrupted.csv")
        
                    # End wandb
                    if general_cfg["wandb_record"]:
                        # upload eval_df to wandb
                        wandb.log({'eval_df': wandb.Table(data=eval_df)})
                        wandb.finish()

                    # Save the eval_df
                    print("Eval DF size:", eval_df.shape)
                    Path(os.path.dirname(csv_save_path)).mkdir(parents=True, exist_ok=True)
                    eval_df.to_csv(csv_save_path, index=False)

    # * Memory Logging
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(
        f"Peak reserved memory for training % of max memory = {lora_percentage} %."
    )

    if not is_llama_factory or accelerator.is_main_process:
        csv_save_path = os.path.join(os.environ["WORKING_DIR"], f"data/08_reporting/language_model_planning_evaluation/{eval_cfg['config_type_name']}_planning_evaluation_results{'_lla_factory' if is_llama_factory else ''}_top_k_{top_k}{'_ppo' if if_test_ppo_version else ''}{'_debug' if debug_mode else ''}.csv")
        
        # End wandb
        if general_cfg["wandb_record"]:
            # upload eval_df to wandb
            if if_test_ppo_version:
                wandb.log({'eval_df_ppo': wandb.Table(data=eval_df)})
            else:
                wandb.log({'eval_df': wandb.Table(data=eval_df)})
            wandb.finish()

        # Save the eval_df
        print("Eval DF size:", eval_df.shape)
        Path(os.path.dirname(csv_save_path)).mkdir(parents=True, exist_ok=True)
        eval_df.to_csv(csv_save_path, index=False)
    
    return None


if __name__ == "__main__":
    import json
    import logging.config
    from kedro.config import OmegaConfigLoader
    from kedro.framework.project import settings
    import os 
    
    conf_path = os.path.join(os.environ["WORKING_DIR"], settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path, env= "base")
    conf_params = conf_loader['parameters']
    config_type_name = os.environ.get("EVAL_CONFIG_TYPE", "t0")
    print(f"config_type_name for planning evaluation: {config_type_name}")
    time.sleep(2)
    eval_cfg = conf_params["language_model_planning_evaluation_params"][f"{config_type_name}_params"]
    general_cfg = conf_params["language_model_planning_evaluation_general_params"]

    print("Please using accelerate launch to run this pipeline")
    
    num_process = len(os.environ.get("CUDA_VISIBLE_DEVICES", "1").split(","))
    print("Num process:", num_process)
    evaluate_language_model_planning(eval_cfg, general_cfg)
