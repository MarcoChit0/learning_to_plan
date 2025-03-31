import os
from typing import List
from better_language_model_for_plan_generation.utils.constant import *
from omegaconf import OmegaConf
from pddl.core import Problem, Domain
from pddl.parser.problem import ProblemParser
from pddl.parser.domain import DomainParser
from pddl.logic.predicates import Predicate
import re
from icecream import ic
import random
from pddl.logic.base import And
from natsort import natsorted
import pickle

INIT_STATE_START_STR = "As initial conditions I have that, "
GOAL_STATE_START_STR = "My goal is to have that "


def convert_predicates_to_text(grounded_predicate_list: List[Predicate], 
                               object_text_dict, 
                               predicate_text_dict,):
    """
    Converts a list of grounded predicates to their corresponding text representation.
    Args:
        grounded_predicate_list (List[Predicate]): A list of grounded predicates.
        object_text_dict: A dictionary mapping object names to their corresponding text representation.
        predicate_text_dict: A dictionary mapping predicate names to their corresponding text representation.
    Returns:
        List[str]: A list of text representations of the grounded predicates.
    """
    
    temp_lst = []
    for predicate in grounded_predicate_list:
        # get predicate name
        predicate_name = predicate.name
        # get predicate terms 
        predicate_terms = predicate.terms
        a = 1 
        # convert terms to text
        predicate_terms_texts = [] 
        for term in predicate_terms:
            term_name = term.name
            # loop over the object_text_dict to get the text of the term
            found_flag = False 
            for key, val in object_text_dict.items():
                # check if the key has ^ and $, if yes, use regex 
                if key.startswith("^") and key.endswith("$"):
                    match = re.search(key, term_name)
                    if match:
                        group_lst = match.groups()
                        output = val.format(*group_lst) # format the text with the groups
                        predicate_terms_texts.append(output) 
                        found_flag = True
                        break # when found, break the loop
                    else:
                        continue 
                else: # if not regex check if the key is in the term_name
                    if key in term_name:
                        # two cases, if val has {} or not, if yes, we need to find the remaining part 
                        # if not, we just need to replace the key with the val
                        if "{}" in val:
                            # get the match index
                            match_index = term_name.index(key)
                            # get the remaining part
                            remaining_part = term_name[match_index + len(key):]
                            output = val.format(remaining_part)
                            predicate_terms_texts.append(output) 
                        else: # if not, just use the val 
                            predicate_terms_texts.append(val)
                        found_flag = True
                        break # when found, break the loop
                    else:
                        continue
            if not found_flag:
                raise ValueError(f"Term {term_name} not found in object_text_dict")
            
        # convert the predicate to text
        # get val from the predicate_text_dict
        predicate_declaration_val = predicate_text_dict[predicate_name]
        # format the predicate_declaration_val with the predicate_terms_texts
        predicate_text = predicate_declaration_val.format(*predicate_terms_texts)
        temp_lst.append(predicate_text)
        
    return temp_lst
                    
        

def problem_info_getter(problem_model: Problem, config_type: CONFIG_TYPES, config_content: OmegaConf,):
    # return {
        # "problem_init_text"
        # "problem_goal_text"
    #}
    
    # get problem model 
    # get init predicate list 
    # convert to text lines 
    
    # get goal predicate list
    # convert to text lines
    # check wether need to shuffle the order of the predicates
    
    # * get predicates dict and encoded_objects dict
    
    predicate_dict = config_content.predicates
    encoded_objects = config_content.encoded_objects
    
    # * get init state and goal state
    init_state_list = problem_model.init
    if isinstance(init_state_list, frozenset):
        init_state_list = list(init_state_list)
    else:
        init_state_list = [init_state_list]
    goal_state_list = problem_model.goal
    if isinstance(goal_state_list, frozenset):
        goal_state_list = list(goal_state_list)
    elif isinstance(goal_state_list, And):
        goal_state_list = list(goal_state_list.operands)
    else:
        goal_state_list = [goal_state_list]
        
    # natsort 
    init_state_list = natsorted(init_state_list)
    goal_state_list = natsorted(goal_state_list)
        
    # * convert init state and goal state to text
    init_state_text_list = convert_predicates_to_text(init_state_list, encoded_objects, predicate_dict)
    
    goal_state_text_list = convert_predicates_to_text(goal_state_list, encoded_objects, predicate_dict)
    
    # * check if need to shuffle 
    if config_type not in [CONFIG_TYPES.t0, CONFIG_TYPES.t0_t5]:
        random.shuffle(init_state_text_list)
        random.shuffle(goal_state_text_list)
        
    # * convert to text
    init_state_text = INIT_STATE_START_STR + ', '.join(init_state_text_list) + "."
    goal_state_text = GOAL_STATE_START_STR + ', '.join(goal_state_text_list) + "."
    return {
        "problem_init_text": init_state_text,
        "problem_goal_text": goal_state_text
    }


if __name__ == "__main__":
    # * general setup
    problem_instances_dir= 'data/01_raw/LLMs-Planning/plan-bench/instances'
    domain_filename = "generated_domain.pddl"
    problem_instance_dir = "generated_basic"
    CWD = '/home/sukaih/Extrastorage/PhDProject_Ext/BetterLMforPlanGen/better-language-model-for-plan-generation'
    
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
            
        problem_model = ProblemParser()(problem_data)
        
        # check if we can pickle the problem model (Yes, we can)
        # with open("problem_model.pkl", "wb") as f:
        #     pickle.dump(problem_model, f)
           
        # * get config content
        domain_config_dir = 'data/01_raw/LLMs-Planning/plan-bench/configs'

        domain_config_fp = os.path.join(CWD, domain_config_dir, f"{domain_name}.yaml")
        config_content = OmegaConf.load(domain_config_fp)
        
        output = problem_info_getter(problem_model, CONFIG_TYPES.accu_t2, config_content)
        print(output)