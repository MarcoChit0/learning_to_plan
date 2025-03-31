# through the plan generation, there are two types of errors:

# 1. the successor state is not valid, i.e., $succ(s) \notin \{\langle a, s'\rangle\mid f(s,a) =s'\}$
# 2. the goal state checker  $isgoal(s)$ error. 

# Thus the negative example should cover these two cases.

# For 1, we randomly move plan action at the end to front
# For 2, we randomly change the how many steps left counter (make it smaller)

# ! func in this file will be used in plan_getter, do not directly call funcs in this file outside of plan_getter

from copy import deepcopy
from typing import List
import random
import re 
from enum import Enum
from better_language_model_for_plan_generation.utils.constant import CONFIG_TYPES, SPECIAL_TOKENS

class MISTAKE_TYPE(Enum):
    wrong_counter_only = 0 
    dumb_mistake_only = 1 # dumb mistake means placing plan action at the end to front
    wrong_counter_and_dumb_mistake = 2
    
class ADVANCED_MISTAKE_TYPE(Enum):
    wrong_objects_only = 3 
    wrong_action_only = 4
    
def get_group_name_of_object(obj_name):
    obj_split = obj_name.split("_")
    # check if last part is not a number, then it is a group name
    # otherwise, the first part is the group name
    if not obj_split[-1].isdigit():
        group_name = obj_split[-1]
    else:
        group_name = obj_split[0]
    return group_name

def advanced_scratchpad_getter(action_text_lst: List[str], mistake_ratio: float, plan_interm_tuple_lst: List[tuple], actions_text_dict, config_type):
    
    # ! --- formulate action_lst_dict and object_lst_dict
    config_type_str = config_type.name 
    action_lst_dict = dict() # key is action name, value is a tuple of object candidates as list
    object_lst_dict = dict() # key is object name, value is a list of same group objects
    
    
    object_group_dict = dict()
    
    for tup in plan_interm_tuple_lst:
        local_action_data = tup[0]
    
        local_object_data_lst = tup[1:]
        action_lst_dict[local_action_data] = []
        # analyze object_data_lst
        for idx_o, obj_name in enumerate(local_object_data_lst):
            object_lst_dict[obj_name] = None 
            # get the group name
            group_name = get_group_name_of_object(obj_name)
            # add to object_group_dict
            if group_name not in object_group_dict:
                object_group_dict[group_name] = []
            if obj_name not in object_group_dict[group_name]:
                object_group_dict[group_name].append(obj_name)
                
            # update action_lst_dict
            if len(action_lst_dict[local_action_data]) < idx_o + 1:
                action_lst_dict[local_action_data].append([])
            
            if obj_name not in action_lst_dict[local_action_data][idx_o]:
                action_lst_dict[local_action_data][idx_o].append(obj_name)
            
        
    # update object_lst_dict
    for obj_name in object_lst_dict:
        group_name = get_group_name_of_object(obj_name)
        assert group_name in object_group_dict
        object_lst_dict[obj_name] = object_group_dict[group_name]
    
    # grouping object by getting their 
    
    # ! --- end of formulating, they will be used to generate mistakes
    
    nowrong_index_lst = [] 
    for idx, action_text in enumerate(action_text_lst):
        if SPECIAL_TOKENS.BACK.value not in action_text:
            nowrong_index_lst.append(idx)
            
    assert len(nowrong_index_lst) == len(plan_interm_tuple_lst) # sanity check
            
    # based on the mistake_ratio, we further decide some index to make mistakes
    further_mistake_index_lst = random.sample(nowrong_index_lst, int(len(nowrong_index_lst) * mistake_ratio))
    
    further_mistake_action_text_lst = []
    for further_mis_idx in further_mistake_index_lst:
        # get the action text
        further_mis_action_text = deepcopy(action_text_lst[further_mis_idx])
        # get action name and object name from the plan_interm_tuple_lst
        original_further_mist_action_name = plan_interm_tuple_lst[nowrong_index_lst.index(further_mis_idx)][0]
        original_further_mist_object_name_lst = deepcopy(plan_interm_tuple_lst[nowrong_index_lst.index(further_mis_idx)][1:])
        
        # randomly pick a mistake type
        mistake_type = random.choice(list(ADVANCED_MISTAKE_TYPE))
        if mistake_type == ADVANCED_MISTAKE_TYPE.wrong_objects_only:
            # randomly decide how many objects to change
            num_obj_to_change = random.randint(1, len(original_further_mist_object_name_lst)) 
            obj_change_idx_lst = random.sample(range(len(original_further_mist_object_name_lst)), num_obj_to_change)
            for obj_change_idx in obj_change_idx_lst:
                obj_name = original_further_mist_object_name_lst[obj_change_idx]
                # get the group name    
                group_name = get_group_name_of_object(obj_name)
                # get the object list
                obj_lst = object_lst_dict[obj_name]
                # randomly pick a new object
                if len(obj_lst) > 1:
                    while True:
                        new_obj_name = random.choice(obj_lst)
                        if new_obj_name != obj_name:
                            break
                else:
                    new_obj_name = original_further_mist_object_name_lst[(obj_change_idx-1) % len(original_further_mist_object_name_lst)]
                # replace the object name
                further_mis_action_text = further_mis_action_text.replace(obj_name, new_obj_name)
                
        elif mistake_type == ADVANCED_MISTAKE_TYPE.wrong_action_only:
            # randomly pick a new action
            while True:
                new_action_name = random.choice(list(action_lst_dict.keys()))
                if new_action_name != original_further_mist_action_name:
                    break
            # see if any objects overlap 
            new_action_obj_lst = action_lst_dict[new_action_name]
            updated_tuple = [new_action_name] # ! not finish yet, but will be used to construct the new action text
            for tup in new_action_obj_lst:
                random.shuffle(tup) 
                find_overlap = False 
                for tup_obj in tup:
                    if len(original_further_mist_object_name_lst) == 0: # if the original object list is empty, then we can just add the object
                        break
                    if tup_obj in original_further_mist_object_name_lst:
                        find_overlap = True
                        # keep the object 
                        updated_tuple.append(tup_obj)
                        # pop the object from the original_further_mist_object_name_lst
                        original_further_mist_object_name_lst.remove(tup_obj)
                        break
                if not find_overlap:
                    updated_tuple.append(random.choice(tup))
            # now we have the updated_tuple, we can construct the new action text
            updated_tuple_action = updated_tuple[0]
            updated_tuple_obj = updated_tuple[1:]
            
            further_mis_action_text_temp = ""
            if SPECIAL_TOKENS.COUNT_END.value in further_mis_action_text:
                further_mis_action_text_temp = further_mis_action_text[:further_mis_action_text.find(SPECIAL_TOKENS.COUNT_END.value) + len(SPECIAL_TOKENS.COUNT_END.value)] + " "
                
            if SPECIAL_TOKENS.EFFECT_END.value in further_mis_action_text:
                # quite complex 
                # find a line that contains the same action name
                find_line_idx = -1 
                find_line_objs = None 
                for tup_idx, tup in enumerate(plan_interm_tuple_lst):
                    if tup[0] == updated_tuple_action:
                        find_line_idx = tup_idx
                        find_line_objs = tup[1:]
                        break
                assert find_line_idx != -1
                # get the action text of the find line 
                find_line_action_text = action_text_lst[nowrong_index_lst[find_line_idx]]
                # ! get the precon part 
                precon_part = find_line_action_text[find_line_action_text.find(SPECIAL_TOKENS.PRECON.value):find_line_action_text.find(SPECIAL_TOKENS.PRECON_END.value) + len(SPECIAL_TOKENS.PRECON_END.value)]
                precon_part = " "+ precon_part + " "
                # replace the object name
                for find_line_obj_idx, find_line_obj in enumerate(find_line_objs):
                    precon_part = precon_part.replace(find_line_obj, updated_tuple_obj[find_line_obj_idx])
                # ! get the effect part
                effect_part = find_line_action_text[find_line_action_text.find(SPECIAL_TOKENS.EFFECT.value):find_line_action_text.find(SPECIAL_TOKENS.EFFECT_END.value) + len(SPECIAL_TOKENS.EFFECT_END.value)]
                effect_part = " "+ effect_part + " "
                # replace the object name
                for find_line_obj_idx, find_line_obj in enumerate(find_line_objs):
                    effect_part = effect_part.replace(find_line_obj, updated_tuple_obj[find_line_obj_idx])
                    
                further_mis_action_text = further_mis_action_text_temp + precon_part + actions_text_dict[updated_tuple_action].format(*updated_tuple_obj) + effect_part
                
                
            else:        
                further_mis_action_text = further_mis_action_text_temp + actions_text_dict[updated_tuple_action].format(*updated_tuple_obj)
            
        else:
            raise ValueError(f"Invalid mistake type {mistake_type}")
        
        further_mistake_action_text_lst.append(further_mis_action_text + f" {SPECIAL_TOKENS.BACK.value} ")
                   
        
    # form the final action_text_lst
    updated_output = []
    for idx, action_text in enumerate(action_text_lst):
        if idx in further_mistake_index_lst:
            updated_output.append(further_mistake_action_text_lst[further_mistake_index_lst.index(idx)])
            
        updated_output.append(action_text) # again add back the correct action text
    
    return updated_output 
    
    


def scratchpad_getter(action_text_lst: List[str], plan_length: int, mistake_ratio: float, if_lite_cot: bool):
    def increment_idx():
        nonlocal idx
        nonlocal local_mistake_count
        idx = idx + 1
        local_mistake_count = 0
        return idx, local_mistake_count
    
    count_part_pattern = re.compile(r"""<COUNT>(.*)</COUNT>""")
    
    updated_action_text_lst = []
    idx = 0 
    local_mistake_count = 0 
    
    while idx < plan_length:
        # get random number
        rand_num = random.random()
        # scale the mistake_ratio, reduce the chance of having multiple consecutive mistakes
        mistake_ratio_local = mistake_ratio * (0.5**local_mistake_count)
        if rand_num < mistake_ratio_local:
            # make mistake 
            # randomly pick a mistake type
            if if_lite_cot:
                mistake_type = random.choice(list(MISTAKE_TYPE))
            else: # will only have dumb mistake
                mistake_type = MISTAKE_TYPE.dumb_mistake_only
            if mistake_type == MISTAKE_TYPE.dumb_mistake_only:
                # if idx is the last index, then we do not need to do anything
                if idx == plan_length - 1:
                    updated_action_text_lst.append(action_text_lst[idx])
                    idx, local_mistake_count = increment_idx()
                    continue
                
                else:
                    # find a mistake_idx ranging from idx+1 to plan_length - 1
                    mistake_idx = random.randint(idx + 1, plan_length - 1)
                    mistake_action_text = action_text_lst[mistake_idx]
                    # modify the action_text_lst
                    if if_lite_cot:
                        # need to sub the <COUNT> part 
                        count_part_actual = re.search(count_part_pattern, action_text_lst[idx]).group()
                        mistake_action_text = re.sub(count_part_pattern, count_part_actual, mistake_action_text)
                    # add WRONG tag
                    mistake_action_text = mistake_action_text.rstrip() # remove the trailing space
                    mistake_action_text += f" {SPECIAL_TOKENS.BACK.value} "
                    updated_action_text_lst.append(mistake_action_text) 
                    local_mistake_count += 1
            elif mistake_type == MISTAKE_TYPE.wrong_counter_only:
                mistake_action_text = deepcopy(action_text_lst[idx])
                count_part_match = re.search(count_part_pattern, mistake_action_text)
                if count_part_match is None:
                    raise ValueError(f"Cannot find the count part in the action text {mistake_action_text}")
                count_os = count_part_match.group(1)
                # get partial count
                if int(count_os) == 0:
                    # rewrite count_os to have o
                    count_mistake = random.randint(1, 4)
                else:    
                    count_mistake = random.randint(0, int(count_os) - 1)
                
                count_mistake = f"{SPECIAL_TOKENS.COUNT.value} " + str(count_mistake) + f" {SPECIAL_TOKENS.COUNT_END.value}"
                # sub 
                mistake_action_text = re.sub(count_part_pattern, count_mistake, mistake_action_text)
                
                # add WRONG tag
                mistake_action_text = mistake_action_text.rstrip() # remove the trailing space
                mistake_action_text += f" {SPECIAL_TOKENS.BACK.value} "
                updated_action_text_lst.append(mistake_action_text)
                local_mistake_count += 1
                
            elif mistake_type == MISTAKE_TYPE.wrong_counter_and_dumb_mistake:
                # if idx is the last index, then we do not need to do anything
                if idx == plan_length - 1:
                    updated_action_text_lst.append(action_text_lst[idx])
                    idx, local_mistake_count = increment_idx()
                    continue
                else:
                    # find a mistake_idx ranging from idx+1 to plan_length - 1
                    mistake_idx = random.randint(idx + 1, plan_length - 1)
                    mistake_action_text = action_text_lst[mistake_idx]
                    # modify the action_text_lst
                    
                    # add WRONG tag
                    mistake_action_text = mistake_action_text.rstrip() # remove the trailing space
                    mistake_action_text += f" {SPECIAL_TOKENS.BACK.value} "
                    updated_action_text_lst.append(mistake_action_text) 
                    local_mistake_count += 1
                    
            else:
                raise ValueError(f"Invalid mistake type {mistake_type}")
                
        else:
            # no mistake 
            updated_action_text_lst.append(action_text_lst[idx])
            idx, local_mistake_count = increment_idx()
            
    return updated_action_text_lst