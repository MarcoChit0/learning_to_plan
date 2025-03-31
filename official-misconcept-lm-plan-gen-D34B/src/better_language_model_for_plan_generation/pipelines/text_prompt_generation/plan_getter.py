import os
import re
from icecream import ic
from omegaconf import OmegaConf
from better_language_model_for_plan_generation.utils.constant import *
from better_language_model_for_plan_generation.utils.call_planning_as_service import call_lama_first_pddl_planner
from better_language_model_for_plan_generation.pipelines.text_prompt_generation.cot_getter import lite_cot_getter, dense_cot_getter
from better_language_model_for_plan_generation.pipelines.text_prompt_generation.scratchpad_getter import scratchpad_getter, advanced_scratchpad_getter
from pddl.core import Domain
from pddl.parser.domain import DomainParser

ANCHOR_PLAN_SEC = "My plan is as follows:\n<PLAN>\n"
ANCHOR_PLAN_SEC_END = "\n</PLAN>"

def convert_actions_to_text(
    plan_sequence, 
    encoded_objects_text_dict,
    actions_text_dict,
):
    # return a list of text representation of the actions
    # Step 1. convert the plan sequence element into (action_name, arg1, arg2, ...) tuple
    # seq element example '(clean-shot shot0 ingredient124 left right)'
    plan_interm_tuple_lst = []
    for seq_element in plan_sequence:
        # unwrap () 
        seq_element = seq_element[1:-1]
        # split by space
        seq_element_lst = seq_element.split(" ")
        # ! for args, we further convert them into text representation
        act_name = seq_element_lst[0]
        terms = seq_element_lst[1:]
        text_ver_terms = [] 
        for term_name in terms:
            found_flag = False
            for key, val in encoded_objects_text_dict.items():
                if key.startswith("^") and key.endswith("$"):
                    match = re.search(key, term_name)
                    if match:
                        group_lst = match.groups()
                        output = val.format(*group_lst)
                        text_ver_terms.append(output)
                        found_flag = True
                        break # when found, break the loop
                    else:
                        continue
                else: # if not regex check if the key is in the term_name
                    if key in term_name:
                        # two cases, if val has {} or not, if yes, we need to find the remaining part 
                        # if not, we just need to replace the key with the val
                        if "{}" in val:
                            match_index = term_name.index(key)
                            # get the remaining part
                            remaining_part = term_name[match_index + len(key):]
                            output = val.format(remaining_part)
                            text_ver_terms.append(output)
                        else: # if not, just replace the key with the val
                            text_ver_terms.append(val)
                        found_flag = True
                        break # when found, break the loop
                    else:
                        continue
            if not found_flag:
                raise ValueError(f"Cannot find the text representation for the term {term_name}")
            
        updated_tuple = [act_name] + text_ver_terms
        plan_interm_tuple_lst.append(updated_tuple)
    
    # Step 2. convert tuple into text representation
    plan_text_lst = []
    for seq_element in plan_interm_tuple_lst:
        action_name = seq_element[0]
        other_args = seq_element[1:]
        # get the action text
        action_val = actions_text_dict[action_name]
        # format the action text with the other_args
        action_text = action_val.format(*other_args)
        plan_text_lst.append(action_text)
        
    return plan_text_lst, plan_interm_tuple_lst


def plan_info_getter(
    plan_lst,
    problem_goal_text: str, # collected from problem_info_getter in problem_getter.py
    config_type: CONFIG_TYPES,
    config_content: OmegaConf,
    mistake_rate: float, # for scratchpad generation, we need to know the mistake rate
    actions_text_dict, # for advanced scratchpad generation
):

    # return (
    # "raw_plan": [action_1, ..., action_n],
    # "plan_text": plan_text, # part of the prompt
    # "plan_length": m # number of actions in the plan
    # )

    # * get encoded_objects dict
    encoded_objects_dict = config_content.encoded_objects
    action_text_dict = config_content.actions
    action_size = len(action_text_dict)
    if 'domain_intro_variable' in config_content:
        domain_intro_variable = config_content.domain_intro_variable
    else:
        domain_intro_variable = None

    action_text_lst, plan_interm_tuple_lst = convert_actions_to_text(
        plan_lst,
        encoded_objects_dict,
        action_text_dict,
    )

    # ! --- Depend on the Config Type, we may need to modify the Action Text List ---
    if config_type in [CONFIG_TYPES.t0, CONFIG_TYPES.accu_t1]:
        pass # normal plan output 
    elif config_type == CONFIG_TYPES.accu_t2: # two components 1) goal state, 2) how many steps left
        # add lite cot
        action_text_lst = lite_cot_getter(action_text_lst, problem_goal_text)
    elif config_type == CONFIG_TYPES.accu_t3:
        # add dense first
        action_text_lst = dense_cot_getter(action_text_lst, plan_interm_tuple_lst, domain_intro_variable, action_size, action_text_dict)
        # add lite second
        action_text_lst = lite_cot_getter(action_text_lst, problem_goal_text)
    elif config_type == CONFIG_TYPES.accu_t4:
        # add dense first
        action_text_lst = dense_cot_getter(action_text_lst, plan_interm_tuple_lst, domain_intro_variable, action_size, action_text_dict)

        # add lite second
        action_text_lst = lite_cot_getter(action_text_lst, problem_goal_text)

        # add synthetic mistake
        action_text_lst = scratchpad_getter(
            action_text_lst, 
            len(plan_lst),
            mistake_rate, 
            if_lite_cot=True
        )
    elif config_type == CONFIG_TYPES.accu_t1_t4:
        pass
        # add synthetic mistake only
        action_text_lst = scratchpad_getter(
            action_text_lst, 
            len(plan_lst),
            mistake_rate, 
            if_lite_cot=False
        )
    elif config_type == CONFIG_TYPES.accu_t2_t4:
        # add lite cot
        action_text_lst = lite_cot_getter(action_text_lst, problem_goal_text)
        # add synthetic mistake
        action_text_lst = scratchpad_getter(
            action_text_lst, 
            len(plan_lst),
            mistake_rate, 
            if_lite_cot=True
        )
    elif config_type == CONFIG_TYPES.accu_t1_t3:
        # add dense first
        action_text_lst = dense_cot_getter(action_text_lst, plan_interm_tuple_lst, domain_intro_variable, action_size, action_text_dict)
    elif config_type == CONFIG_TYPES.accu_t1_t3_t4:
        # add dense first
        action_text_lst = dense_cot_getter(action_text_lst, plan_interm_tuple_lst, domain_intro_variable, action_size, action_text_dict)

        # add synthetic mistake
        action_text_lst = scratchpad_getter(
            action_text_lst, 
            len(plan_lst),
            mistake_rate, 
            if_lite_cot=False
        )
    
    elif config_type == CONFIG_TYPES.t0_t5:
        # add synthetic mistake only
        action_text_lst = scratchpad_getter(
            action_text_lst, 
            len(plan_lst),
            mistake_rate, 
            if_lite_cot=False
        )    
        # apply advanced scratchpad
        action_text_lst = advanced_scratchpad_getter(action_text_lst, mistake_rate, plan_interm_tuple_lst, actions_text_dict, config_type)
    
    else:
        raise ValueError(f"Invalid Config Type: {config_type}")

    # ! --- End ---

    plan_text = "\n".join(action_text_lst)
    plan_text = f"{ANCHOR_PLAN_SEC}{plan_text}{ANCHOR_PLAN_SEC_END}"
    return {
        "raw_plan": plan_lst,
        "plan_text": plan_text,
        "plan_length": len(plan_lst)
    }


if __name__ == "__main__":
    # * general setup
    problem_instances_dir= 'data/01_raw/LLMs-Planning/plan-bench/instances'
    domain_filename = "generated_domain.pddl"
    problem_instance_dir = "generated_basic"
    CWD = '/home/sukaih/Extrastorage/PhDProject_Ext/BetterLMforPlanGen/better-language-model-for-plan-generation'
    
    problem_goal_text_example = "My goal is to have that crate_0 is in depot_48."
    
    mistake_rate = 0.5 
    
    for domain_name in PLANNING_DOMAIN:
        domain_name = domain_name.name
        ic(domain_name)
        # * get domain model
        domain_path = os.path.join(CWD, problem_instances_dir, domain_name, domain_filename)
        with open(domain_path, 'r') as f:
            domain_data = f.read()
        
        # * get problem instance
        instance_example_path = os.path.join(CWD, problem_instances_dir, domain_name, problem_instance_dir, "instance-56.pddl")
        with open(instance_example_path, 'r') as f:
            problem_data = f.read()
            
        # * get config content
        domain_config_dir = 'data/01_raw/LLMs-Planning/plan-bench/configs'

        domain_config_fp = os.path.join(CWD, domain_config_dir, f"{domain_name}.yaml")
        config_content = OmegaConf.load(domain_config_fp)
            
        for config_type in [CONFIG_TYPES.accu_t2_t4]:
            ic(config_type)
            # domain_data and problem_data can be str or file path
            plan_result = call_lama_first_pddl_planner(domain_data, problem_data)
            if not plan_result["is_ok"]:
                raise ValueError(f"Error in finding PDDL plan: {plan_result['error']}")

            plan_lst = plan_result["plan_lst"]
            # output = plan_info_getter(plan_lst, problem_goal_text_example, config_type, config_content, mistake_rate, actions_text_dict)
            # print(output['plan_text'])
            # print("Plan Length", output['plan_length'])
        # break # for testing purpose
