
from typing import List
import pickle
import sys
from icecream import ic 
from .constant import *
import random
import os
import numpy as np
from glob import glob
from natsort import natsorted
import re
from tqdm.auto import tqdm
from pathlib import Path
import jsonlines
from datasets import load_dataset
from collections import defaultdict
from omegaconf import OmegaConf
from datetime import datetime
from subprocess import PIPE, run

SUPPORT_SPECIAL_CASE = False # Set this to False, otherwise we cannot guarantee the correctness of the evaluation 

def longest_common_contiguous_subsequence(str1, str2):
    m, n = len(str1), len(str2)
    # Initialize the DP table
    lcs_con = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    result = 0
    end_index = 0

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                lcs_con[i][j] = lcs_con[i-1][j-1] + 1
                if lcs_con[i][j] > result:
                    result = lcs_con[i][j]
                    end_index = i
            else:
                lcs_con[i][j] = 0

    # If no common subsequence found
    if result == 0:
        return ""

    # Extract the longest common contiguous subsequence
    start_index = end_index - result
    return str1[start_index:end_index]

def longest_common_subsequence(list1, list2):
    m, n = len(list1), len(list2)
    # Create a table to store lengths of LCS for all subproblems
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Build the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list1[i-1] == list2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Reconstruct the LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if list1[i-1] == list2[j-1]:
            lcs.append(list1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return list(reversed(lcs))

def encode_action_text(pattern, action_text):
    # Step 1: Convert {} to {0}, {1}, ... based on their order of appearance
    def replace_braces(match):
        # Increment the counter for each placeholder
        replace_braces.counter += 1
        return f"{{{replace_braces.counter}}}"

    # Initialize a counter for the placeholders
    replace_braces.counter = -1
    numbered_pattern = re.sub(r'\{\}', replace_braces, pattern)
    
    # recount how many placeholders in the pattern
    total_num_placeholders = len(re.findall(r'\{\d+\}', numbered_pattern))

    # Step 2: find match 
    # split the template and action text 
    
    index_object_dict = {}
    
    template_words = numbered_pattern.split()
    action_text_words = action_text.split()
    template_pointer = 0 
    action_text_pointer = 0
 
    while template_pointer < len(template_words) and action_text_pointer < len(action_text_words):
        # case 1: same word 
        if template_words[template_pointer] == action_text_words[action_text_pointer]:
            template_pointer += 1
            action_text_pointer += 1
        # case 2: encounter a placeholder
        elif template_words[template_pointer].startswith("{") and template_words[template_pointer].endswith("}"):
            # get the index 
            index = int(template_words[template_pointer][1:-1])
            # get the next word in the template
            if template_pointer + 1 < len(template_words):
                next_template_word = template_words[template_pointer + 1]
                # find the next word in the action text
                next_action_pointer_temp = action_text_pointer
                while next_action_pointer_temp < len(action_text_words):
                    if next_template_word == action_text_words[next_action_pointer_temp]:
                        break
                    next_action_pointer_temp += 1
                # once break, we have the pointer settled 
                template_pointer += 1
                # store the match in the index_object_dict
                index_object_dict[index] = " ".join(action_text_words[action_text_pointer:next_action_pointer_temp])
                # update the action_text_pointer
                action_text_pointer = next_action_pointer_temp
                 
            else:
                match_action_text_word = " ".join(action_text_words[action_text_pointer:])
                index_object_dict[index] = match_action_text_word
                template_pointer += 1
                action_text_pointer = len(action_text_words)
                
        else:
            return False # meaning the template and action text do not match
    if len(index_object_dict) == total_num_placeholders:
        return index_object_dict
    else:
        return False
    
    
def encode_object_text(encoded_name, encoded_objects, reverse_regex_dict_cache=None, reverse_simple_dict_cache=None, reverse_direct_dict_cache=None):
    # divide into two types of encoded objects, one is regex, the other is simple
            
    # reverse the dict 
    if reverse_regex_dict_cache is None:
        reverse_regex_dict = dict()
        reverse_simple_dict = dict()
        reverse_direct_dict = dict()
        for origin, decode_pattern in encoded_objects.items():
            # decode_pattern replace {} with (.+)
            if '{}' in decode_pattern:
                updated_pattern = re.sub(r'\{\}', r'(.+)', decode_pattern)
                if origin.startswith("^"): # regex case 
                    # update this origin
                    origin = origin.replace("^", "").replace("$", "").replace("\\", "")     
                    reverse_regex_dict[re.compile(updated_pattern)] = origin
                else: # simple case
                    reverse_simple_dict[re.compile(updated_pattern)] = origin 
            else:
                reverse_direct_dict[decode_pattern] = origin
    else:
        reverse_regex_dict = reverse_regex_dict_cache
        reverse_simple_dict = reverse_simple_dict_cache
        reverse_direct_dict = reverse_direct_dict_cache
    
    # loop over the reverse dict to find the match
    for pattern, origin in reverse_regex_dict.items():
        match = pattern.search(encoded_name)
        if match:
            groups = match.groups()
            output = re.sub(r'\(.+?\)', '{}', origin).format(*groups)
            return output, reverse_regex_dict, reverse_simple_dict, reverse_direct_dict
        else:
            continue
        
    for pattern, origin in reverse_simple_dict.items():
        match = pattern.search(encoded_name)
        if match:
            assert len(match.groups()) == 1 # only one group
            output = origin+match.group(1)
            return output, reverse_regex_dict, reverse_simple_dict, reverse_direct_dict
        
    for pattern, origin in reverse_direct_dict.items():
        if pattern == encoded_name:
            return origin, reverse_regex_dict, reverse_simple_dict, reverse_direct_dict
        
    return False, reverse_regex_dict, reverse_simple_dict, reverse_direct_dict # if not found, return False
    
# ! MAIN FUNCTION
def convert_text_to_plan_lst(
    plan_text_lst,
    domain_name,
    encoded_objects_text_dict,
    actions_text_dict,
    compile_dict_cache=None,
):
    """
    Convert a list of plan text into a list of plan sequences.
    Args:
        plan_text_lst (list): A list of plan text.
        encoded_objects_text_dict (dict): A dictionary mapping encoded object text to original object name.
        actions_text_dict (dict): A dictionary mapping action name to action text template.
        compile_dict_cache (dict, optional): A dictionary caching compiled dictionaries. Defaults to None.
    Returns:
        tuple: A tuple containing the plan sequence list and the compile dictionary cache.
    """
    if compile_dict_cache is None:
        compile_dict_cache = dict()
    encoded_objects_text_dict_str = str(encoded_objects_text_dict)
    plan_sequence = []
    for action_text in plan_text_lst:
        # Step 1. find the action name and args from the action text 
        found_action = False 
        index_object_dict = None 
        for action_name, action_text_template in actions_text_dict.items():
            # use regex to handle potential formatting in the template 
            # * Handle fuzzy matching for driverlog and logistics domain
            if domain_name in [PLANNING_DOMAIN.driverlog.name, PLANNING_DOMAIN.logistics.name]:
                if action_name in ['load-truck', 'unload-truck'] and 'truck' not in action_text:
                    continue
                elif action_name in ['load-airplane', 'unload-airplane'] and 'airplane' not in action_text:
                    continue
       
            index_object_dict = encode_action_text(action_text_template, action_text)
            if index_object_dict:
                found_action = action_name
                break
        if not found_action:
            plan_sequence.append("BAD") # if not found, problematic
        else:
            # continue convert the object text to original object name
            output = "(" + found_action
            
            indices = list(index_object_dict.keys())
            indices.sort()
            # check what lack
            
            work_smooth = True 
            local_ind = 0
            for cur_indx in indices:
                if SUPPORT_SPECIAL_CASE and local_ind != cur_indx: # * SPECIAL CASE
                    if domain_name == PLANNING_DOMAIN.barman.name:
                        while local_ind < cur_indx:
                            output += " left right"
                            local_ind += 2
                
                object_text = index_object_dict[cur_indx]
                reverse_regex_dict_cache, reverse_simple_dict_cache, reverse_direct_dict_cache = compile_dict_cache.get(encoded_objects_text_dict_str, (None, None, None))
                
                object_name, reverse_regex_dict_cache, reverse_simple_dict_cache, reverse_direct_dict_cache = encode_object_text(object_text, encoded_objects_text_dict, reverse_regex_dict_cache, reverse_simple_dict_cache, reverse_direct_dict_cache)
                
                if encoded_objects_text_dict_str not in compile_dict_cache:
                    compile_dict_cache[encoded_objects_text_dict_str] = (reverse_regex_dict_cache, reverse_simple_dict_cache, reverse_direct_dict_cache)
                
                if object_name:
                    output += " " + object_name
                    local_ind += 1
                else:
                    work_smooth = False
                    break
            if not work_smooth:
                plan_sequence.append("BAD")
            else:
                if SUPPORT_SPECIAL_CASE and domain_name == PLANNING_DOMAIN.barman.name: # * SPECIAL CASE 
                    if found_action in ['shake', 'clean-shot']: # need to add left and right 
                        output += " left right"
                output += ")"
                plan_sequence.append(output)
                
    return plan_sequence, compile_dict_cache
        
# ! MAIN FUNCTION        
def parse_generated_plan_text_into_lst(
    generated_output: str,
):
    plan_start = generated_output.find(SPECIAL_TOKENS.PLAN.value)
    if plan_start == -1:
        return False
    plan_section_str = generated_output[plan_start+len(SPECIAL_TOKENS.PLAN.value):].strip()
    plan_end = plan_section_str.find(SPECIAL_TOKENS.PLAN_END.value)
    plan_section_str = plan_section_str[:plan_end].strip()
    plan_text_lst = plan_section_str.split("\n")
    # remove the empty string
    plan_text_lst = [x.strip() for x in plan_text_lst if x != ""]
    # further remove lite cot, dense cot, scratchpad, etc.
    updated_lst = []
    for plan_text in plan_text_lst:
        # ! skip the scratchpad
        if SPECIAL_TOKENS.BACK.value in plan_text:
            continue
        # ! unwrap lite cot 
        if SPECIAL_TOKENS.GOAL.value in plan_text:
            # find the place of </COUNT>
            count_end = plan_text.find(SPECIAL_TOKENS.COUNT_END.value)
            if count_end == -1:
                return False
            # only want the content after </COUNT>
            plan_text = plan_text[count_end+len(SPECIAL_TOKENS.COUNT_END.value):]
        # ! unwrap dense cot
        if SPECIAL_TOKENS.PRECON.value in plan_text:
            # we want the content between </PRECON> and <EFFECT>
            precon_end = plan_text.find(SPECIAL_TOKENS.PRECON_END.value)
            effect_start = plan_text.find(SPECIAL_TOKENS.EFFECT.value, precon_end)
            plan_text = plan_text[precon_end+len(SPECIAL_TOKENS.PRECON_END.value):effect_start]
            plan_text = plan_text.strip()
        
        updated_lst.append(plan_text)
    return updated_lst
        
# still need to provide the planning VAL case to evaluate the plan
def eval_plan_lst_using_planner(
    ref_domain_fp,
    ref_problem_fp,
    plan_output_fp,
    plan_length,
    val_fp,
    remove_plan_output=True,
    only_check_plan_executable=False,
    only_check_goal_achievable=False,
):
    assert not (only_check_plan_executable and only_check_goal_achievable) # cannot check both
    
    if only_check_plan_executable:
        if plan_length < 3:
            succeed_phrase = "XXXXX" # if the plan length less than 3, we deem it as not executable directly
        else:
            succeed_phrase = 'Plan executed successfully'
    elif only_check_goal_achievable:
        succeed_phrase = 'Goal not satisfied'
    else:
        succeed_phrase = 'Successful plans'
    if only_check_goal_achievable:
        cmd_lst = [val_fp, '-c', ref_domain_fp, ref_problem_fp, plan_output_fp]
    else:
        cmd_lst = [val_fp, ref_domain_fp, ref_problem_fp, plan_output_fp]
    result = run(cmd_lst, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    # print(result.stdout)
    if remove_plan_output:
        Path(plan_output_fp).unlink()
    if not only_check_goal_achievable:
        if succeed_phrase in result.stdout:
            return True
        else:
            return False
    else:
        if succeed_phrase in result.stdout: # means the goal is not satisfied, so return False
            return False
        else:
            return True
    
# ! MAIN FUNCTION
def compare_two_plan_lst(predicted_plan_lst, target_plan_lst, domain_name=None, if_hard=True, if_lccs=False):
    
    # if if_hard is True, then we require the exact match
    if if_hard:
        if len(predicted_plan_lst) != len(target_plan_lst):
            return 0.0
        for i in range(len(predicted_plan_lst)):
            if predicted_plan_lst[i] != target_plan_lst[i]:
                if SUPPORT_SPECIAL_CASE and domain_name == PLANNING_DOMAIN.barman.name: # * SPECIAL CASE for barman 
                    if 'left right' in predicted_plan_lst[i]:
                        if predicted_plan_lst[i].replace('left right', 'right left') == target_plan_lst[i]:
                            continue
                    elif 'right left' in predicted_plan_lst[i]:
                        if predicted_plan_lst[i].replace('right left', 'left right') == target_plan_lst[i]:
                            continue
                return 0.0
        return 1.0
    # if if_hard is False, then we calculate the longest common subsequence   
    else:
        # calculate the longest common subsequence
        if if_lccs: # longest common contiguous subsequence
            longest_sub = longest_common_contiguous_subsequence(predicted_plan_lst, target_plan_lst)
        else:
            longest_sub = longest_common_subsequence(predicted_plan_lst, target_plan_lst)
        return len(longest_sub) / len(target_plan_lst)
    

# if __name__ == "__main__":
        
#     # load dataset 
#     huggingface_dataset_dir = os.path.join(os.environ['WORKING_DIR'], "data/03_primary/llm_plan_generation_dataset")
#     val_fp = os.path.join(os.environ['WORKING_DIR'], 'opt/VAL/validate')
    
#     domain_fp_pattern = os.path.join(huggingface_dataset_dir, "zraw_data/instances/{domain_name}/generated_domain.pddl")
#     problem_instance_fp_pattern = os.path.join(huggingface_dataset_dir, "zraw_data/instances/{domain_name}/{subdir_name}/instance-{instance_id}.pddl")
    
    
#     plan_output_dir = os.path.join(os.environ['WORKING_DIR'], f"data/07_model_output/llm_plan_generation/plan_output-{datetime.now().strftime('%m-%d-%H-%M-%S')}")
#     Path(plan_output_dir).mkdir(parents=True, exist_ok=True)
    
#     debug_config_name = CONFIG_TYPES.accu_t4.name
#     split_name = SPLIT_NAMES.test_same_domain.name
#     dataset = load_dataset(huggingface_dataset_dir, debug_config_name, split=split_name)
    
#     test_chunk = dataset
#     total_size = len(test_chunk)
#     # preparation 
#     config_dir = os.path.join(os.environ['WORKING_DIR'], "data/03_primary/llm_plan_generation_dataset/zraw_data/configs")
#     config_cache_dict = dict()
    
    
#     # extract the plan text sequence 
#     full_text_prompt_datachunk = test_chunk["full_text_prompt"]
#     raw_plan_datachunk = test_chunk["raw_plan"]
#     domain_name_datachunk = test_chunk["domain"]
#     instance_id_datachunk = test_chunk["instance_id"]
#     prompt_id_datachunk = test_chunk["prompt_id"]
    
#     # zip the data
#     for idx, test_data in enumerate(tqdm(test_chunk, desc="Evaluating generated plan text", total=total_size)):
        
#         # ! Eval step 1, obtain params 
#         full_text_prompt = test_data["full_text_prompt"]
#         target_plan_lst = test_data["raw_plan"]
#         domain_name = test_data["domain"]
#         instance_id = test_data["instance_id"]
#         prompt_id = test_data["prompt_id"]
        
#         ref_domain_fp = domain_fp_pattern.format(domain_name=domain_name)
#         ref_problem_fp = problem_instance_fp_pattern.format(domain_name=domain_name, subdir_name=SUBDIRS_NAME_DICT[split_name], instance_id=instance_id)
        
        
#         if domain_name not in config_cache_dict:
#             config_path = os.path.join(config_dir, f"{domain_name}.yaml")
#             config_content = OmegaConf.load(config_path)
#             # get encoded_objects_text_dict and actions_text_dict
#             encoded_objects_text_dict = config_content.encoded_objects
#             actions_text_dict = config_content.actions
#             config_cache_dict[domain_name] = dict()
#             config_cache_dict[domain_name]["encoded_objects_text_dict"] = encoded_objects_text_dict
#             config_cache_dict[domain_name]["actions_text_dict"] = actions_text_dict
#         else:
#             encoded_objects_text_dict = config_cache_dict[domain_name]["encoded_objects_text_dict"]
#             actions_text_dict = config_cache_dict[domain_name]["actions_text_dict"]
            
#         # ! Eval step 2, parse the plan text into a plan list
#         plan_text_lst = parse_generated_plan_text_into_lst(full_text_prompt)
#         # convert the plan text to plan sequence
#         compile_dict_cache = config_cache_dict[domain_name].get("compile_dict_cache", None)
#         # plan_text_lst = ["" for x in range(len(target_plan_lst))] # debug 
#         plan_lst, compile_dict_cache = convert_text_to_plan_lst(
#             plan_text_lst = plan_text_lst,
#             domain_name=domain_name,
#             encoded_objects_text_dict = encoded_objects_text_dict,
#             actions_text_dict = actions_text_dict,
#             compile_dict_cache = compile_dict_cache,
#         )
#         # update the compile_dict_cache
#         config_cache_dict[domain_name]["compile_dict_cache"] = compile_dict_cache
        
#         # ! Eval step 3, Evaluate the score of the plan sequence
#         # type 1: compare the plan_lst with target_plan_lst without calling VAL
#         score = compare_two_plan_lst(plan_lst, target_plan_lst, domain_name, if_hard=True)
#         if score != 1.0:
#             ic(idx, score)
#             ic(plan_text_lst)
#             ic(plan_lst)
#             ic(target_plan_lst)
#             for target_plan_idx in range(len(target_plan_lst)):
#                 if target_plan_lst[target_plan_idx] != plan_lst[target_plan_idx]:
#                     print(f"Incorrect at index {target_plan_idx}")
#                     print(f"Target: {target_plan_lst[target_plan_idx]}")
#                     print(f"Predicted: {plan_lst[target_plan_idx]}")
#                     print(f'Raw text: {plan_text_lst[target_plan_idx]}')
#                     print("---\n")
#             raise ValueError("Error in the plan sequence")
        
#         # type 2: eval the plan_lst using VAL
#         # ! --- write the plan to a file and using VAL ---
#         # write the ref plan lst to a file
#         ref_plan_output_fp = os.path.join(plan_output_dir, f"ref_{domain_name}_{instance_id}_{prompt_id}.txt")
#         with open(ref_plan_output_fp, "w") as f:
#             f.writelines("\n".join(target_plan_lst))
            
#         ref_eval_bool_result = eval_plan_lst_using_planner(ref_domain_fp, ref_problem_fp, ref_plan_output_fp, val_fp)
#         assert ref_eval_bool_result
            
#         # write the generated plan lst to a file 
#         plan_output_fp = os.path.join(plan_output_dir, f"{domain_name}_{instance_id}_{prompt_id}.txt")
#         with open(plan_output_fp, "w") as f:
#             f.writelines("\n".join(plan_lst))
        
#         # evaluate the plan_lst using the planner
#         eval_bool_result = eval_plan_lst_using_planner(ref_domain_fp, ref_problem_fp, plan_output_fp, val_fp)
#         assert eval_bool_result
        
        
if __name__ == "__main__":
    full_text_prompt = """
My plan is as follows:
<PLAN>
<PRECON>We can grasp shaker_119 only if shaker_119 is on the table. We can grasp shaker_119 only if left_hand is empty.</PRECON>grasp the shaker_119 using left_hand<EFFECT>Once we grasp the shaker_119, we are holding the shaker_119 and left_hand is no longer empty.</EFFECT>

<PRECON>We can fill the shot_189 with ingredient_348 only if one hand is holding the shot and the other hand is free to manipulate dispenser_149. We can fill the shot_189 with ingredient_348 only if shot_189 is empty and clean.</PRECON>fill the shot_189 on right_hand with ingredient_348 using dispenser_149 when left_hand holds shaker_119<EFFECT>Once we fill the shot_189 with ingredient_348, shot_189 is no longer empty and clean. Once we fill the shot_189 with ingredient_348, shot_189 contains ingredient_348 and is used to store ingredient_348.</EFFECT>

<PRECON>We can pour the content of shot_189 into shaker_119 only if we are holding both the shot_189 and shaker_119. We can pour the content of shot_189 into shaker_119 only if shot_189 contains some ingredient.</PRECON>pour from shot_189 containing ingredient_348 into shaker_119 using right_hand<EFFECT>Once we pour the ingredient_348 of shot_189 into shaker_119, shaker_119 contains ingredient_348 and shaker_119 is no longer empty. Once we pour, shot_189 is empty.</EFFECT>

<PRECON>We can fill the shot_189 with ingredient_24 only if one hand is holding the shot and the other hand is free to manipulate dispenser_366. We can fill the shot_189 with ingredient_24 only if shot_189 is empty and clean.</PRECON>fill the shot_189 on right_hand with ingredient_24 using dispenser_366 when left_hand holds shaker_119<EFFECT>Once we fill the shot_189 with ingredient_24, shot_189 is no longer empty and clean. Once we fill the shot_189 with ingredient_24, shot_189 contains ingredient_24 and is used to store ingredient_24.</EFFECT>

<PRECON>We can pour the content of shot_189 into shaker_119 only if we are holding both the shot_189 and shaker_119. We can pour the content of shot_189 into shaker_119 only if shot_189 contains some ingredient.</PRECON>pour from shot_189 containing ingredient_24 into shaker_119 using right_hand<EFFECT>Once we pour the ingredient_24 of shot_189 into shaker_119, shaker_119 contains ingredient_24 and shaker_119 is no longer empty. Once we pour, shot_189 is empty.</EFFECT>

<PRECON>We can shake shaker_119 only if we are holding shaker_119 and shaker_119 contains two types of ingredients. We can shake shaker_119 only if shaker_119 has not yet shaken.</PRECON>shake shaker_119 on left_hand to mix into cocktail_107 when right_hand is empty<EFFECT>Once we shake shaker_119, shaker_119 is considered shaken. Once we shake shaker_119, shaker_119 contains cocktail_107, which is a mixture of ingredient_348 and ingredient_24.</EFFECT>

<PRECON>We can pour the cocktail_107 from shaker_119 into shot_234 only if we are holding shaker_119 and shaker_119 contains cocktail_107. We can pour the cocktail_107 into shot_234 only if shot_234 is empty and clean.</PRECON>pour from shaker_119 containing cocktail_107 into shot_234 using left_hand<EFFECT>Once we pour the cocktail_107 from shaker_119 into shot_234, shot_234 contains cocktail_107 and is not empty and not clean. shaker_119's fill level will decrement one level level_-1.</EFFECT>
</PLAN>
"""


    plan_text_lst = parse_generated_plan_text_into_lst(full_text_prompt)
    # convert the plan text to plan sequence
    print(plan_text_lst)