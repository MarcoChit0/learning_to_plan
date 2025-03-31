from icecream import ic
from .eval_generated_plan_text import parse_generated_plan_text_into_lst, convert_text_to_plan_lst, compare_two_plan_lst, eval_plan_lst_using_planner
import time 
from .constant import *
import random
import os
import numpy as np
from glob import glob
from natsort import natsorted
import re
from pathlib import Path
import jsonlines
from datasets import load_dataset
from collections import defaultdict
from omegaconf import OmegaConf
from datetime import datetime
from subprocess import PIPE, run
from multiprocessing import Manager, Lock
from multiprocessing import Pool 
# from multiprocessing.dummy import Pool # Debug
import multiprocessing as mp

CPU_THREAD=6

def compute_planning_metrics_compare_with_ref(eval_preds, config_cache_dict, config_dir, if_hard:bool=False, if_lccs: bool = False, if_average=True):
    full_text_prompt_chunk = eval_preds.predictions
    target_plan_lst_chunk = eval_preds.inputs[FEATURE_CLS.raw_plan.name]
    domain_name_chunk = eval_preds.inputs[FEATURE_CLS.domain.name]
    
    if len(full_text_prompt_chunk) != len(domain_name_chunk):
        # it means we have multiple outputs and we want to get the highest score for top-k
        # it means len(full_text_prompt_chunk) > len(domain_name_chunk)
        k_size = len(full_text_prompt_chunk) // len(domain_name_chunk)
        full_text_prompt_chunk = [full_text_prompt_chunk[i*k_size:(i+1)*k_size] for i in range(len(domain_name_chunk))]
    else:
        k_size = False
    
    total_score_lst = []
    for full_text_prompt, target_plan_lst, domain_name in zip(full_text_prompt_chunk, target_plan_lst_chunk, domain_name_chunk):
        # ! Eval step 1, obtain params 
        if domain_name not in config_cache_dict:
            config_path = os.path.join(config_dir, f"{domain_name}.yaml")
            config_content = OmegaConf.load(config_path)
            # get encoded_objects_text_dict and actions_text_dict
            encoded_objects_text_dict = config_content.encoded_objects
            actions_text_dict = config_content.actions
            config_cache_dict[domain_name] = dict()
            config_cache_dict[domain_name]["encoded_objects_text_dict"] = encoded_objects_text_dict
            config_cache_dict[domain_name]["actions_text_dict"] = actions_text_dict
        else:
            encoded_objects_text_dict = config_cache_dict[domain_name]["encoded_objects_text_dict"]
            actions_text_dict = config_cache_dict[domain_name]["actions_text_dict"]
            
        # ! Eval step 2, parse the plan text into a plan list
        if k_size:
            local_score_lst = []
            for indx in range(k_size):
                plan_text_lst = parse_generated_plan_text_into_lst(full_text_prompt[indx])
                if not plan_text_lst:
                    local_score_lst.append(0.0)
                    continue
                compile_dict_cache = config_cache_dict[domain_name].get("compile_dict_cache", None)
                plan_lst, compile_dict_cache = convert_text_to_plan_lst(
                    plan_text_lst = plan_text_lst,
                    domain_name=domain_name,
                    encoded_objects_text_dict = encoded_objects_text_dict,
                    actions_text_dict = actions_text_dict,
                    compile_dict_cache = compile_dict_cache,
                )
                # update the compile_dict_cache
                config_cache_dict[domain_name]["compile_dict_cache"] = compile_dict_cache
                
                score = compare_two_plan_lst(plan_lst, target_plan_lst, domain_name, if_hard=if_hard, if_lccs=if_lccs) # measure the longest common subsequence
                local_score_lst.append(score)
            # select the highest score
            max_score = max(local_score_lst)
            total_score_lst.append(max_score)
        else:
            plan_text_lst = parse_generated_plan_text_into_lst(full_text_prompt)
            # convert the plan text to plan sequence
            if not plan_text_lst:
                total_score_lst.append(0.0)
                continue
            
            compile_dict_cache = config_cache_dict[domain_name].get("compile_dict_cache", None)
            # plan_text_lst = ["" for x in range(len(target_plan_lst))] # debug 
            plan_lst, compile_dict_cache = convert_text_to_plan_lst(
                plan_text_lst = plan_text_lst,
                domain_name=domain_name,
                encoded_objects_text_dict = encoded_objects_text_dict,
                actions_text_dict = actions_text_dict,
                compile_dict_cache = compile_dict_cache,
            )
            # update the compile_dict_cache
            config_cache_dict[domain_name]["compile_dict_cache"] = compile_dict_cache
            
            # ! Eval step 3, Evaluate the score of the plan sequence
            score = compare_two_plan_lst(plan_lst, target_plan_lst, domain_name, if_hard=if_hard, if_lccs=if_lccs) # measure the longest common subsequence
            total_score_lst.append(score)
        
    if if_average:
        mean_score = float(np.mean(total_score_lst))
        return {
            "planning_metric": mean_score
        }
    else:
        return {
            "planning_metric": total_score_lst
        }


def mp_comput_using_val(kwargs):
    full_text_prompt = kwargs['full_text_prompt'] 
    domain_name = kwargs['domain_name'] 
    instance_id = kwargs['instance_id'] 
    prompt_id = kwargs['prompt_id'] 
    val_fp = kwargs['val_fp'] 
    ref_domain_fp = kwargs['ref_domain_fp'] 
    ref_problem_fp = kwargs['ref_problem_fp'] 
    plan_output_dir = kwargs['plan_output_dir'] 
    config_dir = kwargs['config_dir'] 
    config_cache_dict = kwargs['config_cache_dict']  
    only_check_plan_executable = kwargs['only_check_plan_executable']
    only_check_goal_achievable = kwargs['only_check_goal_achievable']
    
    # ! Eval step 1, obtain params 
    if domain_name not in config_cache_dict:
        config_path = os.path.join(config_dir, f"{domain_name}.yaml")
        config_content = OmegaConf.load(config_path)
        # get encoded_objects_text_dict and actions_text_dict
        encoded_objects_text_dict = config_content.encoded_objects
        actions_text_dict = config_content.actions
        
        temp_dict = {
            "encoded_objects_text_dict": encoded_objects_text_dict,
            "actions_text_dict": actions_text_dict
        }
        config_cache_dict[domain_name] = temp_dict
    else:
        encoded_objects_text_dict = config_cache_dict[domain_name]["encoded_objects_text_dict"]
        actions_text_dict = config_cache_dict[domain_name]["actions_text_dict"]
    # ! Eval step 2, parse the plan text into a plan list
    plan_text_lst = parse_generated_plan_text_into_lst(full_text_prompt)
    if not plan_text_lst:
        return 0.0
    # convert the plan text to plan sequence
    compile_dict_cache = config_cache_dict[domain_name].get("compile_dict_cache", None)
    plan_lst, compile_dict_cache = convert_text_to_plan_lst(
        plan_text_lst = plan_text_lst,
        domain_name=domain_name,
        encoded_objects_text_dict = encoded_objects_text_dict,
        actions_text_dict = actions_text_dict,
        compile_dict_cache = compile_dict_cache,
    )
    # update the compile_dict_cache
    domain_cache_dict_temp = config_cache_dict[domain_name]
    domain_cache_dict_temp["compile_dict_cache"] = compile_dict_cache
    config_cache_dict[domain_name] = domain_cache_dict_temp
    # ! Eval step 3, Evaluate the score of the plan sequence
    # ! --- write the plan to a file and using VAL ---
    # write the ref plan lst to a file
    plan_output_fp = os.path.join(plan_output_dir, f"{domain_name}_{instance_id}_{prompt_id}.txt")
    with open(plan_output_fp, "w") as f:
        f.writelines("\n".join(plan_lst))
    plan_length = len(plan_lst)
    # evaluate the plan_lst using the planner
    eval_bool_result = eval_plan_lst_using_planner(ref_domain_fp, ref_problem_fp, plan_output_fp, plan_length, val_fp, only_check_plan_executable=only_check_plan_executable, only_check_goal_achievable=only_check_goal_achievable) 
    
    if eval_bool_result:
        return 1.0
    else:
        return 0.0

def compute_planning_metrics_using_val(eval_preds, val_fp, domain_fp_pattern, problem_instance_fp_pattern, plan_output_dir, config_dir, config_cache_dict, split_name:str, if_average=True, only_check_plan_executable=False, only_check_goal_achievable=False):
    full_text_prompt_chunk = eval_preds.predictions
    domain_name_chunk = eval_preds.inputs[FEATURE_CLS.domain.name]
    instance_id_chunk = eval_preds.inputs[FEATURE_CLS.instance_id.name]
    prompt_id_chunk = eval_preds.inputs[FEATURE_CLS.prompt_id.name]
    
    if len(full_text_prompt_chunk) != len(domain_name_chunk): # it means we have multiple outputs and we want to get the highest score for top-k
        k_size = len(full_text_prompt_chunk) // len(domain_name_chunk)
        full_text_prompt_chunk = [full_text_prompt_chunk[i*k_size:(i+1)*k_size] for i in range(len(domain_name_chunk))]
    else:
        k_size = False
    
    manager = Manager()
    mp_config_cache_dict = manager.dict(config_cache_dict)
    args = []
    score_lst = [] 
    for full_text_prompt, domain_name, instance_id, prompt_id in zip(full_text_prompt_chunk, domain_name_chunk, instance_id_chunk, prompt_id_chunk):
        if k_size:
            args = []
            for indx in range(k_size):
                kwargs = {
                    "full_text_prompt": full_text_prompt[indx],
                    "domain_name": domain_name,
                    "instance_id": instance_id,
                    "prompt_id": prompt_id,
                    "val_fp": val_fp,
                    "ref_domain_fp": domain_fp_pattern.format(domain_name=domain_name),
                    "ref_problem_fp": problem_instance_fp_pattern.format(domain_name=domain_name, subdir_name=SUBDIRS_NAME_DICT[split_name], instance_id=instance_id),
                    "plan_output_dir": plan_output_dir,
                    "config_dir": config_dir,
                    "config_cache_dict": mp_config_cache_dict,
                    "only_check_plan_executable": only_check_plan_executable,
                    "only_check_goal_achievable": only_check_goal_achievable,
                }
                args.append(kwargs)
            with Pool(CPU_THREAD) as p:
                k_size_scores = list(p.imap(mp_comput_using_val, args, chunksize=10))
                # select the highest score
                max_score = max(k_size_scores)
                score_lst.append(max_score)
        else:
            kwargs = {
                "full_text_prompt": full_text_prompt,
                "domain_name": domain_name,
                "instance_id": instance_id,
                "prompt_id": prompt_id,
                "val_fp": val_fp,
                "ref_domain_fp": domain_fp_pattern.format(domain_name=domain_name),
                "ref_problem_fp": problem_instance_fp_pattern.format(domain_name=domain_name, subdir_name=SUBDIRS_NAME_DICT[split_name], instance_id=instance_id),
                "plan_output_dir": plan_output_dir,
                "config_dir": config_dir,
                "config_cache_dict": mp_config_cache_dict,
                "only_check_plan_executable": only_check_plan_executable,
                "only_check_goal_achievable": only_check_goal_achievable,
            }
            args.append(kwargs)
        
    if not k_size:
        with Pool(CPU_THREAD) as p:
            score_lst = list(p.imap(mp_comput_using_val, args, chunksize=10))
    mean_score = float(np.mean(score_lst))
    
    # reupdate the config_cache_dict
    config_cache_dict_temp = dict(mp_config_cache_dict)
    for domain_name, temp_dict in config_cache_dict_temp.items():
        config_cache_dict[domain_name] = temp_dict
    
    if if_average:
        return {
            "planning_metric": mean_score
        }
    else:
        return {
            "planning_metric": score_lst
        }