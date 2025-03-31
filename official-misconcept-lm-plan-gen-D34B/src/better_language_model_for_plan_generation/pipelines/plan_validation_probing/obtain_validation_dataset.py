from copy import deepcopy
from icecream import ic
from functools import partial
from better_language_model_for_plan_generation.pipelines.text_prompt_generation.plan_getter import ANCHOR_PLAN_SEC, ANCHOR_PLAN_SEC_END
from better_language_model_for_plan_generation.utils.constant import *
import random
import os
import numpy as np
from glob import glob
from natsort import natsorted
import re
from tqdm.auto import tqdm
from pathlib import Path
from datasets import load_dataset
from datetime import datetime
import wandb
import torch
from functools import partial
from omegaconf import OmegaConf 
import pandas as pd
from datasets import Dataset, load_from_disk

VALID_MODEL_CONFIG_TYPE = [CONFIG_TYPES.accu_t1_t4.name, CONFIG_TYPES.accu_t2_t4.name, CONFIG_TYPES.accu_t4.name, CONFIG_TYPES.accu_t1_t3_t4.name, CONFIG_TYPES.t0_t5.name]

def clean_plan_text(plan_text_full, model_config_type:str):
    """
    Cleans the plan text by removing invalid parts and selecting valid parts based on the model configuration type.
    Args:
        plan_text_full (str): The input plan text.
        model_config_type (str): The model configuration type.
    Returns:
        tuple: A tuple containing the valid plan text and the invalid plan text.
    """
    
    # we shall convert the plan text back to list and then select the invalid part and valid part 
    # ref: parse_generated_plan_text_into_lst in `src/better_language_model_for_plan_generation/pipelines/training_llm_for_plan_generation/eval_generated_plan_text.py`
    plan_start = plan_text_full.find(SPECIAL_TOKENS.PLAN.value)
    if plan_start == -1:
        raise ValueError(f"The plan text does not contain the plan section.\nRAW INPUT\n\n{plan_text_full}")
    
    plan_section_str = plan_text_full[plan_start+len(SPECIAL_TOKENS.PLAN.value):].strip()
    plan_end = plan_section_str.find(SPECIAL_TOKENS.PLAN_END.value)
    if plan_end == -1: # if not found, we shall use the end of the string
        plan_end = len(plan_section_str)
    plan_section_str = plan_section_str[:plan_end].strip()
    plan_text_lst = plan_section_str.split("\n")
    # remove the empty string
    plan_text_lst = [x.strip() for x in plan_text_lst if x != ""]
    
    assert model_config_type in VALID_MODEL_CONFIG_TYPE, f"model_config_type: {model_config_type} is not in the valid list: {VALID_MODEL_CONFIG_TYPE}"
    
    updated_lst = [] # the updated plan text list, we will only keep only one MISTAKE line in the plan text
    # need to find scratchpad (mistake) line(s)
    mistake_line_idxs = [] 
    
    for cur_ind, plan_text in enumerate(plan_text_lst):
        # ! check if the line contains the mistake
        # -- Step 1: check if the line contains the mistake
        if SPECIAL_TOKENS.BACK.value in plan_text:
            # ! further analyze the mistake line, we need to correct the wrong counter mistake 
            # seek correct line 
            seek_idx = cur_ind + 1 # the next line
            while seek_idx < len(plan_text_lst):
                if SPECIAL_TOKENS.BACK.value not in plan_text_lst[seek_idx]:
                    break
                seek_idx += 1
                
            seek_plan_text = plan_text_lst[seek_idx] # seek means correct next line
            # Now we have both the mistake line and any next correct line because t2 contains special COUNT error and we do not want to test that. 
            # we only get the plan part, we do not need the dense CoT, the PRECON and EFFECT part and COUNT part
            # we check if the seek plan text is the same as the current plan text
            # if same, we will skip the current plan text
            count_end_start_for_seek = seek_plan_text.find(SPECIAL_TOKENS.COUNT_END.value)
            temp_seek_plan_text_part = seek_plan_text[count_end_start_for_seek + len(SPECIAL_TOKENS.COUNT_END.value):].strip()
            count_end_start_for_current = plan_text.find(SPECIAL_TOKENS.COUNT_END.value)
            temp_plan_text_part = plan_text[count_end_start_for_current + len(SPECIAL_TOKENS.COUNT_END.value):].strip()
    
                
            if temp_seek_plan_text_part.strip() == temp_plan_text_part.replace(SPECIAL_TOKENS.BACK.value, "").strip():
                # this means the current plan text is actually correct, we should skip it
                continue
            else:
                # we need to keep the current plan text
                mistake_line_idxs.append(len(updated_lst))
                # ! also remove the <BACK> tag
                back_start = plan_text.find(SPECIAL_TOKENS.BACK.value)
                plan_text = plan_text[:back_start]
        
        # -- Step 2: consider the model_config_type and unwrap the plan text
        if model_config_type == CONFIG_TYPES.accu_t4.name:
            # no need to do anything
            pass 
        if model_config_type in [CONFIG_TYPES.accu_t2_t4.name, CONFIG_TYPES.accu_t1_t4.name, CONFIG_TYPES.t0_t5.name]:
            # remove dense CoT, the PRECON and EFFECT part , no t3
            precon_start = plan_text.find(SPECIAL_TOKENS.PRECON.value)
            before_precon_str = plan_text[:precon_start].strip()
            precon_end = plan_text.find(SPECIAL_TOKENS.PRECON_END.value)
            after_precon_str = plan_text[precon_end + len(SPECIAL_TOKENS.PRECON_END.value):].strip()
            effect_start = after_precon_str.find(SPECIAL_TOKENS.EFFECT.value)
            before_effect_str = after_precon_str[:effect_start].strip()
            plan_text = before_precon_str + " "+ before_effect_str 
        if model_config_type in [CONFIG_TYPES.accu_t1_t4.name, CONFIG_TYPES.accu_t1_t3_t4.name, CONFIG_TYPES.t0_t5.name]:
            # remove lite CoT, the GOAL and COUNT part, no t2 
            counter_end = plan_text.find(SPECIAL_TOKENS.COUNT_END.value)
            plan_text = plan_text[counter_end + len(SPECIAL_TOKENS.COUNT_END.value):].strip()
        updated_lst.append(plan_text)
            
    # -- Step 3: generate valid and invalid example 
    # invalid example only keep one mistake line and make sure the mistake line is the last line
    invalid_example = deepcopy(updated_lst)
    # randomly pick one mistake line
    if len(mistake_line_idxs) == 0: # it means this plan text does not contain any mistake
        # special case, we return None, None 
        return None, None, None
    wanted_mistake_idx = random.choice(mistake_line_idxs)
    invalid_example = invalid_example[:wanted_mistake_idx + 1] # only keep the mistake line and the lines before it
    # remove other mistake lines
    invalid_example = [x for idx, x in enumerate(invalid_example) if idx not in mistake_line_idxs or idx == wanted_mistake_idx]
    
    # valid example remove the mistake line and randomly slice the plan text
    valid_example = deepcopy(updated_lst)
    # remove all the mistake lines
    valid_example = [x for idx, x in enumerate(valid_example) if idx not in mistake_line_idxs]
    # randomly slice the plan text  > 3 
    slice_len = random.randint(3, len(valid_example))
    raw_example = [x for x in valid_example]
    valid_example = valid_example[:slice_len]
    
    # -- Step 4: reconstruct back the plan text ref: 
    # Additional, add space before < and after > for invalid example
    for idx in range(len(invalid_example)):
        if invalid_example[idx].startswith("<"):
            invalid_example[idx] = " " + invalid_example[idx]
        if invalid_example[idx].endswith(">"):
            invalid_example[idx] = invalid_example[idx] + " "
    invalid_plan_text = "\n".join(invalid_example)
    invalid_plan_text = f"{ANCHOR_PLAN_SEC}{invalid_plan_text}" # ! DO NOT WANT TO ADD ANCHOR_PLAN_SEC_END as it is not the end of the plan text
    
    # Additional, add space before < and after > for valid example
    for idx in range(len(valid_example)):
        if valid_example[idx].startswith("<"):
            valid_example[idx] = " " + valid_example[idx]
        if valid_example[idx].endswith(">"):
            valid_example[idx] = valid_example[idx] + " "
            
    valid_plan_text = "\n".join(valid_example)
    valid_plan_text = f"{ANCHOR_PLAN_SEC}{valid_plan_text}" # ! DO NOT WANT TO ADD ANCHOR_PLAN_SEC_END as it is not the end of the plan text
    
    # Additional, add space before < and after > for raw example
    for idx in range(len(raw_example)):
        if raw_example[idx].startswith("<"):
            raw_example[idx] = " " + raw_example[idx]
        if raw_example[idx].endswith(">"):
            raw_example[idx] = raw_example[idx] + " "
    raw_plan_text = "\n".join(raw_example)
    raw_plan_text = f"{ANCHOR_PLAN_SEC}{raw_plan_text}" # ! DO NOT WANT TO ADD ANCHOR_PLAN_SEC_END as it is not the end of the plan text
    
    return valid_plan_text, invalid_plan_text, raw_plan_text

def llama_factory_mapping(batch, model_config_type:str, add_back=False):
    # ! add_back = False, later we will add both " WRONG" and "\n" and compare the conditional probability. 
    # add_back so that we can calculate perplexity to see if the model can detect whether the plan is valid or not
    # we need to have system, user, and assistant part, ref: `src/better_language_model_for_plan_generation/pipelines/language_model_planning_evaluation/nodes.py`
    batch_len = len(batch['domain'])
    valid_example_output = []
    invalid_example_output = []
    raw_example_output = []
    system_output = []
    user_output = []
    for i in range(batch_len):
        # -- clean the plan text
        valid_plan_text, invalid_plan_text, raw_plan_text = clean_plan_text(batch['output'][i], model_config_type)
        if valid_plan_text is None:
            continue
        if add_back:
            valid_plan_text += SPECIAL_TOKENS.BACK.value
            invalid_plan_text += SPECIAL_TOKENS.BACK.value
        valid_example_output.append(valid_plan_text)
        invalid_example_output.append(invalid_plan_text)
        raw_example_output.append(raw_plan_text)
        system_output.append("You are a helpful assistant.")
        user_output.append(batch['instruction'][i] + '\n'  + batch['input'][i])
    
    return {
        'system': system_output,
        'user': user_output,
        'valid_example_output': valid_example_output,
        'invalid_example_output': invalid_example_output,
        'raw_example_output': raw_example_output
    } 
    
def unsloth_mapping(batch, model_config_type:str, add_back=False):
    # check the "full_text_prompt" in `src/better_language_model_for_plan_generation/pipelines/text_prompt_generation/nodes.py`
    assert False, "we currently only support llama factory model"
    batch_len = len(batch['domain'])
    valid_example_output = []
    invalid_example_output = []
    for i in range(batch_len):
        # -- clean the plan text
        valid_plan_text, invalid_plan_text = clean_plan_text(batch['plan_text'][i], model_config_type)
        if add_back:
            valid_plan_text += SPECIAL_TOKENS.BACK.value
            invalid_plan_text += SPECIAL_TOKENS.BACK.value
        domain_text = batch['domain_text'][i]
        problem_init_text = batch['problem_init_text'][i]
        problem_goal_text = batch['problem_goal_text'][i]
        # we need to construct the full text prompt and then append the valid and invalid plan text
        valid_example_output.append(domain_text + problem_init_text + problem_goal_text + valid_plan_text)
        invalid_example_output.append(domain_text + problem_init_text + problem_goal_text + invalid_plan_text)


    return {
        'valid_example_output': valid_example_output,
        'invalid_example_output': invalid_example_output
    }


def obtain_validation_dataset(model_cfg, model_config_type:str, previous_dataset:Dataset, cache_path:str):
    """
    Obtain a validation dataset by mapping the previous dataset based on the given model configuration.
    Args:
        model_cfg (dict): The model configuration. refer to language_model_planning_evaluation_params in eval yaml 
        model_config_type (str): The type of model configuration. accu_t1_t4 etc which contain t4 at least
        previous_dataset (Dataset): The previous dataset to be mapped.
    Returns:
        Dataset: The mapped validation dataset., each sentence's end contains <BACK> token
    """
    if not os.path.exists(cache_path):
    
        # -- Step 1: find whether it is llama_factory or unsloth 
        is_llama_factory = model_cfg['is_llama_factory']
        if is_llama_factory:
            mapping_func = llama_factory_mapping
        else:
            mapping_func = unsloth_mapping
            
        
        # -- Step 2: map the previous dataset
        mapped_dataset = previous_dataset.map(
            partial(mapping_func, model_config_type=model_config_type),
            batched=True,
            remove_columns=previous_dataset.column_names,
        )
        # save only 1600 examples
        if len(mapped_dataset) < 1600:
            mapped_dataset = mapped_dataset.select(range(len(mapped_dataset)))
        else:
            mapped_dataset = mapped_dataset.select(random.sample(range(len(mapped_dataset)), 1600))
        # Save the mapped dataset to disk 
        mapped_dataset.save_to_disk(cache_path)
        
        # -- Step 3: save the mapped dataset 
    else:
        mapped_dataset = load_from_disk(cache_path)
    
    return mapped_dataset
        
