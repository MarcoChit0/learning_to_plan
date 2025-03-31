from better_language_model_for_plan_generation.utils.constant import *
import os 
from pathlib import Path
import random 
from omegaconf import OmegaConf
from icecream import ic

ANCHOR_ACTION_SEC = "Here are the actions that can be performed:"
ANCHOR_CONDITION_SEC = "The following are the restrictions on the actions"

def extract_info_from_domain_intro(cfg_content:OmegaConf, config_type, split_type_name):
    """
    Extracts information from the domain introduction in the given configuration content.
    Args:
        cfg_content (OmegaConf): The configuration content containing the domain introduction.
        config_type: The type of configuration.
    Returns:
        str: The extracted information from the domain introduction.
    Raises:
        ValueError: If the anchor action section or condition section cannot be found in the domain.
    """
    
    domain_intro_str = cfg_content.domain_intro
    # split into lines 
    lines = domain_intro_str.split("\n")
    lines = [x.strip() for x in lines if x.strip() != ""]  
    
    # know which line index contains the anchor action section
    action_sec_idx = -1
    condition_sec_idx = -1
    for idx, line in enumerate(lines):
        if ANCHOR_ACTION_SEC in line:
            action_sec_idx = idx
        if ANCHOR_CONDITION_SEC in line:
            condition_sec_idx = idx
        if action_sec_idx != -1 and condition_sec_idx != -1:
            break
        
    if action_sec_idx == -1 or condition_sec_idx == -1:
        assert ValueError(f"Cannot find the anchor action section or condition section in the domain {cfg_content.domain_name}")
        
    if config_type in [CONFIG_TYPES.t0, CONFIG_TYPES.t0_t5] or "test" in split_type_name: # raw prompt, so no need to shuffle the information
        output = "\n".join(lines)
    else: # need to shuffle the information
        init_output = "\n".join(lines[:action_sec_idx])
        action_sec_lst = lines[action_sec_idx + 1:condition_sec_idx]
        # shuffle it 
        random.shuffle(action_sec_lst)
        action_sec_str = "\n".join(action_sec_lst)
        
        cond_sec_lst = lines[condition_sec_idx + 1:]
        # shuffle it
        random.shuffle(cond_sec_lst)
        cond_sec_str = "\n".join(cond_sec_lst)
        output = f"{init_output}\n\n{ANCHOR_ACTION_SEC}\n{action_sec_str}\n\n{ANCHOR_CONDITION_SEC}\n{cond_sec_str}\n"
    return output
    

def domain_info_getter(planning_domain: PLANNING_DOMAIN, config_dir, config_type: CONFIG_TYPES, split_type_name:str):
    """
    Retrieves domain information from a configuration file.
    Args:
        planning_domain (PLANNING_DOMAIN): The planning domain.
        config_dir (str): The absolute path to the configuration directory.
        config_type (CONFIG_TYPES): The type of configuration.
    Returns:
        str: The domain text extracted from the configuration file.
    """
    
    # assert config_dir is absolute path
    assert os.path.isabs(config_dir), f"config_dir {config_dir} is not an absolute path."
    
    # config file 
    config_file = os.path.join(config_dir, f"{planning_domain.name}.yaml")
    # load yaml using OmegaConf
    config_content = OmegaConf.load(config_file)
    # get domain text 
    domain_text = extract_info_from_domain_intro(config_content, config_type, split_type_name)
    return domain_text, config_content

if __name__ == "__main__":
    domain_config_dir = 'data/01_raw/LLMs-Planning/plan-bench/configs'
    CWD = '/home/sukaih/Extrastorage/PhDProject_Ext/BetterLMforPlanGen/better-language-model-for-plan-generation'
    
    for domain in PLANNING_DOMAIN:
        ic(domain.name)
        domain_text, config_content = domain_info_getter(domain, os.path.join(CWD, domain_config_dir), CONFIG_TYPES.accu_t2)
        print(domain_text)
        print(config_content)
        break