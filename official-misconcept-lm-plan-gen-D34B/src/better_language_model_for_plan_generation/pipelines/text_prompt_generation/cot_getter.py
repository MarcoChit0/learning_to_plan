import re
import random

# ! func in this file will be used in plan_getter, do not directly call funcs in this file outside of plan_getter

def lite_cot_getter(action_text_lst, problem_goal_text):
    updated_action_text_lst = []
    s = len(action_text_lst)
    for re_count, action_text in enumerate(action_text_lst):
        updated_text = " <GOAL> " + problem_goal_text + " <COUNT> " + str(s - re_count - 1)  + " </COUNT> " + action_text
        updated_action_text_lst.append(updated_text) 
    return updated_action_text_lst


def parse_domain_intro_variable(domain_intro_variable, action_size, action_text_dict, if_shuffle=True):
    pattern = re.compile(r"""\[['"](.+)['"]\]""")
    # find all the matches and index 
    matches = list(pattern.finditer(domain_intro_variable))
    assert len(matches) == action_size, f"Number of matches {len(matches)} does not match the action size {action_size}"
    # get the index
    index_lst = [(m.start(), m.end() ) for m in matches]
    action_name_lst = [m.group(1) for m in matches]
    output_action_variable_info_dict = {}
    for idx, action_name in enumerate(action_name_lst):
        output_action_variable_info_dict[action_name] = {
            "precondition": None,
            "effect": None,
        }
        start_ind = index_lst[idx][1]
        end_ind = index_lst[idx + 1][0] if idx != len(index_lst) - 1 else len(domain_intro_variable)
        action_variable_str = domain_intro_variable[start_ind:end_ind].strip()
        # split by "\n"
        action_variable_lines = action_variable_str.split("\n")
        action_variable_lines = [x.strip() for x in action_variable_lines if x.strip() != ""]
        # effects are those start with 'Once'
        effects_lst = []
        precon_lst = [] 
        for line in action_variable_lines:
            if line.startswith("Once"):
                effects_lst.append(line)
            else:
                precon_lst.append(line)
        if if_shuffle:
            random.shuffle(effects_lst)
            random.shuffle(precon_lst)
        output_action_variable_info_dict[action_name]["precondition"] = precon_lst
        output_action_variable_info_dict[action_name]["effect"] = effects_lst
        
    # check if all the action names are in the action_text_dict
    action_key_output_set = set(output_action_variable_info_dict.keys())
    action_key_dict_set = set(action_text_dict.keys())
    assert action_key_output_set == action_key_dict_set, f"Action key set in the domain_intro_variable does not match the action_text_dict"
    
    return output_action_variable_info_dict


def dense_cot_getter(action_text_lst, plan_interm_tuple_lst, domain_intro_variable, action_size, action_text_dict):
    if domain_intro_variable is None:
        return action_text_lst
    action_variable_info_dict = parse_domain_intro_variable(domain_intro_variable, action_size, action_text_dict)
    updated_action_text_lst = []
    
    for idx, action_text in enumerate(action_text_lst):
        action_name = plan_interm_tuple_lst[idx][0]
        action_args = plan_interm_tuple_lst[idx][1:]
        # get precon text representations 
        precon_text_rep_lst = []
        for precon in action_variable_info_dict[action_name]["precondition"]:
            precon_text_rep = precon.format(*action_args)
            precon_text_rep_lst.append(precon_text_rep)
        # get effect text representations
        effect_text_rep_lst = []
        for effect in action_variable_info_dict[action_name]["effect"]:
            effect_text_rep = effect.format(*action_args)
            effect_text_rep_lst.append(effect_text_rep)
            
        updated_text = " <PRECON> " + " ".join(precon_text_rep_lst) + " </PRECON> " + action_text + " <EFFECT> " + " ".join(effect_text_rep_lst) + " </EFFECT> "
        updated_action_text_lst.append(updated_text)
    
    return updated_action_text_lst 