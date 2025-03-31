"""
This is a boilerplate pipeline 'text_prompt_generation'
generated using Kedro 0.19.6
"""

# * --- INFO ---
# ! we have the following hierarchy for the dataset:
# - planning domain -> 'normal' | 'mystery' | 'novel'
    # - planning horizon  -> 'basic' | 'longer'
        # - Training_Corpora_Type_ID -> 't0' | 'accu-t1' | 'accu-t2' | 'accu-t3' | 'accu-t4' | 'accu-t1+t4' | 'accu-t2+t4'
            # - diverse number of prompts -> 1 | 5 | 25

import pickle
import sys
from icecream import ic 
from better_language_model_for_plan_generation.utils.constant import *
from better_language_model_for_plan_generation.utils.call_planning_as_service import call_lama_first_pddl_planner
from .domain_getter import domain_info_getter
from .problem_getter import problem_info_getter
from .plan_getter import plan_info_getter
import random
import os
import numpy as np
from glob import glob
from natsort import natsorted
from pddl.parser.problem import ProblemParser
import re
from tqdm.auto import tqdm
from pathlib import Path
from multiprocessing import Pool, Lock, Manager
from multiprocessing.dummy import Pool as DummyPool
import multiprocessing as mp
import jsonlines
from datasets import load_dataset
import torch
from multiprocessing import Pool, current_process

def mp_create_jsonl(kwargs):
    # * get the kwargs
    planning_domain = kwargs['planning_domain']
    domain_getter_config_dir = kwargs['domain_getter_config_dir']
    config_type = kwargs['config_type']
    problem_model = kwargs['problem_model']
    domain_data = kwargs['domain_data']
    problem_instance_fp = kwargs['problem_instance_fp']
    mistake_rate = kwargs['mistake_rate']
    instance_id = kwargs['instance_id']
    prompt_id = kwargs['prompt_id']
    split_type = kwargs['split_type']
    seed = kwargs['seed']
    huggingface_dataset_dir = kwargs['huggingface_dataset_dir']
    mp_lock = kwargs['mp_lock']
    mp_plan_dict = kwargs['mp_plan_dict']
    huggingface_chunk_size = kwargs['huggingface_chunk_size']
    if_qwen = kwargs['if_qwen']
    
    # # ! DEBUG
    # if split_type == SPLIT_NAMES.train:
    #     return True # skip training data
    
    # * Set Seed
    random.seed(int(seed))
    np.random.seed(int(seed))
    
    domain_text, config_content = domain_info_getter(planning_domain, domain_getter_config_dir, config_type, split_type.name)
    problem_getter_output = problem_info_getter(problem_model, config_type, config_content)
    problem_init_text = problem_getter_output["problem_init_text"]
    problem_goal_text = problem_getter_output["problem_goal_text"]
    
    # get plan 
    plan_lst_key = f"{planning_domain.name}{problem_instance_fp}"
    if plan_lst_key not in mp_plan_dict:
        plan_result = call_lama_first_pddl_planner(domain_data, problem_instance_fp)
        try_limit = 2 
        try_count = 0
        while try_count < try_limit:
            try: 
                if not plan_result["is_ok"]:
                    raise ValueError(f"Error in finding PDDL plan in domain {planning_domain.name}, problem instance {instance_id} : {plan_result['error']}")
                plan_lst = plan_result["plan_lst"]
                mp_plan_dict[plan_lst_key] = plan_lst
                break
            except Exception as e:
                try_count += 1
                print("Going to retry")
                
        if try_count == try_limit:
            return False
            
    else:
        plan_lst = mp_plan_dict[plan_lst_key]
    actions_text_dict = config_content.actions
    plan_getter_output = plan_info_getter(plan_lst, problem_goal_text, config_type, config_content, mistake_rate, actions_text_dict)
    
    raw_plan = plan_getter_output["raw_plan"]
    plan_text = plan_getter_output["plan_text"]
    plan_length = plan_getter_output["plan_length"]
    
    # construct the output dict 
    if not if_qwen:
        final_output_dict = {
            "domain": planning_domain.name,
            "instance_id": instance_id,
            "prompt_id": prompt_id,
            "mistake_rate": mistake_rate,
            "domain_text": domain_text,
            "problem_init_text": problem_init_text,
            "problem_goal_text": problem_goal_text,
            "plan_text": plan_text,
            "plan_length": plan_length,
            "raw_plan": raw_plan,
            "full_text_prompt": domain_text + problem_init_text + problem_goal_text + plan_text 
        }
    else:
        final_output_dict = {
            "domain": planning_domain.name,
            "instance_id": instance_id,
            "prompt_id": prompt_id,
            "mistake_rate": mistake_rate,
            "instruction": domain_text, # follow llama_factory format
            "input": problem_init_text + "\n" + problem_goal_text,
            "output": plan_text,
            "plan_length": plan_length,
            "raw_plan": raw_plan,
            "full_text_prompt": domain_text + problem_init_text + problem_goal_text + plan_text 
        }
    if "instruction" not in final_output_dict: # it means we are using the old format
        assert set(final_output_dict.keys()) == set(FEATURE_CLS.__members__.keys())     
    
    # write to jsonl   
    chunk_id = int(instance_id) % huggingface_chunk_size
    json_fp = os.path.join(huggingface_dataset_dir, config_type.name, split_type.name, f"chunk_{chunk_id}.jsonl")
    
    with mp_lock:
        with jsonlines.open(json_fp, 'a') as writer:
            writer.write(final_output_dict)
            
    return True 
        
    

# ! NODE
def construct_huggingface_dataset(text_prompt_generation_params):
    from transformers import AutoTokenizer
    ic(text_prompt_generation_params) 
    PWD = os.environ['PWD']
    mistake_rate = text_prompt_generation_params["mistake_rate"]
    domain_config_dir = text_prompt_generation_params["domain_config_dir"]
    problem_instances_dir = text_prompt_generation_params["problem_instances_dir"]
    if_qwen = text_prompt_generation_params["if_qwen"]
    
    creation_kwargs_temp_save_fp = os.path.join(PWD, text_prompt_generation_params["creation_kwargs_temp_save_fp"])
    
    train_size = text_prompt_generation_params["train_size"]
    test_size = text_prompt_generation_params["test_size"]
    mp_worker = text_prompt_generation_params["mp_worker"]
    
    huggingface_dataset_dir = os.path.join(PWD, text_prompt_generation_params["huggingface_dataset_dir"])

    seed = text_prompt_generation_params["seed"]
    huggingface_chunk_size = text_prompt_generation_params["huggingface_chunk_size"]
    
    has_debug_huggingface_dataset = text_prompt_generation_params["has_debug_huggingface_dataset"]
    has_create_huggingface_dataset = text_prompt_generation_params["has_create_huggingface_dataset"]
    
    
    # * Set Seed 
    random.seed(seed)
    np.random.seed(seed)
    
    
    
    if not has_create_huggingface_dataset and not os.path.exists(creation_kwargs_temp_save_fp): # it means we need to create the dataset
        construction_args_collection = [] 
        # loop the config, then loop the domain, then loop the split, then create multiple prompts for each instance
        for planning_domain in tqdm(PLANNING_DOMAIN, desc="Planning Domain"):
            # ! also get domain data 
            domain_getter_config_dir = os.path.join(PWD, domain_config_dir)
            domain_fp = os.path.join(PWD, problem_instances_dir, planning_domain.name, "generated_domain.pddl")
            with open(domain_fp, 'r') as f:
                domain_data = f.read()
                
            # get splits 
            split_lst = planning_domain.get_splits()
            for split_type in tqdm(split_lst, desc="Split Type", leave=False):
                # ! get the problem instance here 
                if split_type == SPLIT_NAMES.train:
                    problem_instances_fps = glob(os.path.join(PWD, problem_instances_dir, planning_domain.name, "generated_basic", "instance-*.pddl"))
                    # natsort
                    problem_instances_fps = natsorted(problem_instances_fps)
                    # slice
                    problem_instances_fps = problem_instances_fps[:train_size]
                    assert len(problem_instances_fps) == train_size
                elif split_type == SPLIT_NAMES.test_same_domain:
                    problem_instances_fps = glob(os.path.join(PWD, problem_instances_dir, planning_domain.name, "generated_basic", "instance-*.pddl"))
                    # natsort
                    problem_instances_fps = natsorted(problem_instances_fps)
                    # slice
                    problem_instances_fps = problem_instances_fps[-test_size:]
                    assert len(problem_instances_fps) == test_size
                elif split_type == SPLIT_NAMES.test_unseen_domain:
                    problem_instances_fps = glob(os.path.join(PWD, problem_instances_dir, planning_domain.name, "generated_basic", "instance-*.pddl"))
                    # slice 
                    problem_instances_fps = problem_instances_fps[:test_size]
                    assert len(problem_instances_fps) == test_size
                elif split_type == SPLIT_NAMES.test_longer_horizon:
                    problem_instances_fps = glob(os.path.join(PWD, problem_instances_dir, planning_domain.name, "generated_basic_longer_plan_len", "instance-*.pddl"))
                    # slice
                    problem_instances_fps = problem_instances_fps[:test_size]
                    assert len(problem_instances_fps) == test_size
                elif split_type == SPLIT_NAMES.test_mystery_domain:
                    problem_instances_fps = glob(os.path.join(PWD, problem_instances_dir, planning_domain.name, "generated_basic", "instance-*.pddl"))
                    # slice 
                    problem_instances_fps = problem_instances_fps[:test_size]
                    assert len(problem_instances_fps) == test_size
                else:
                    raise ValueError(f"Unknown split type {split_type}")    
                
                    
                for problem_instance_fp in tqdm(problem_instances_fps, desc="Problem Instance", leave=False):
                    # get instance_id from the problem_instance_fp
                    instance_id = re.search(r'instance-(\d+).pddl', problem_instance_fp).group(1)
                    # get problem data 
                    with open(problem_instance_fp, 'r') as f:
                        problem_data = f.read()
                    problem_model = ProblemParser()(problem_data)
                        
                    # ! DEBUG, we can only loop over one config type
                    # for config_type in tqdm([CONFIG_TYPES.t0_t5], desc="Config Type", leave=False):
                    for config_type in tqdm(CONFIG_TYPES, desc="Config Type", leave=False):
                        # get prompt_num
                        prompt_num = PROMPT_NUM_DICT[config_type]
                        if split_type != SPLIT_NAMES.train:
                            prompt_num = 1 # only one prompt for testing
                    
                        for prompt_idx in tqdm(range(prompt_num), desc="Prompt", leave=False):
                            # * domain_getter needs planning_domain, domain_getter_config_dir, config_type
                            # * problem_getter needs problem_model, config_type, ~config_content
                            # * plan_getter needs (~domain_data, ~problem_instance_fp), ~problem_goal_text, config_type, ~config_content,  mistake_rate
                            # we thus need planning_domain, domain_getter_config_dir, config_type, problem_model, domain_data, problem_instance_fp, mistake_rate
                            # * later we need to add seed to the args too 
                            kwargs = {
                                "planning_domain": planning_domain,
                                "domain_getter_config_dir": domain_getter_config_dir,
                                "config_type": config_type,
                                "problem_model": problem_model,
                                "domain_data": domain_data,
                                "problem_instance_fp": problem_instance_fp,
                                "mistake_rate": mistake_rate,
                                # we need to add enum data
                                "instance_id": instance_id,
                                "prompt_id": prompt_idx,
                                "split_type": split_type,
                                
                            }
                            construction_args_collection.append(kwargs)
        
        # ! add seed to the args
        print(f"Total number of prompts: {len(construction_args_collection)}")
        total_len = len(construction_args_collection)
        # make random seed for total_len times
        rand_seeds = list(np.random.randint(0, 999999, total_len))     
        # add to the args
        for idx, kwargs in enumerate(construction_args_collection):
            kwargs["seed"] = rand_seeds[idx]
            
        # pickle dump temp save 
        
        Path(os.path.dirname(creation_kwargs_temp_save_fp)).mkdir(parents=True, exist_ok=True)
        with open(creation_kwargs_temp_save_fp, "wb") as f:
            pickle.dump(construction_args_collection, f)
            
    if not has_create_huggingface_dataset and os.path.exists(creation_kwargs_temp_save_fp):
        # continue from the temp save
        mp.set_start_method('fork', force=True)
        print("Continue from the temp save")    
        with open(creation_kwargs_temp_save_fp, "rb") as f:
            construction_args_collection = pickle.load(f)
            
        # add huggingface_dataset_dir to the args
        for kwargs in tqdm(construction_args_collection, desc="Adding Huggingface Dataset Dir"):
            kwargs["huggingface_dataset_dir"] = huggingface_dataset_dir
            kwargs['if_qwen'] = if_qwen
            
        # create lock within same config_type, split_type, planning_domain and instance_id
        mp_manager = Manager()
        mp_plan_dict = mp_manager.dict()
        # lock_dict = dict()
        lock_lst = [mp_manager.Lock() for _ in range(huggingface_chunk_size)]
        for kwargs in tqdm(construction_args_collection, desc="Creating Lock"):
            # assign_lock_key = f"{kwargs['config_type'].name}_{kwargs['split_type'].name}_{kwargs['planning_domain'].name}_{kwargs['instance_id']}"
            # if assign_lock_key not in lock_dict:
            #     lock_dict[assign_lock_key] = mp_manager.Lock()
            # kwargs["mp_lock"] = lock_dict[assign_lock_key]
            assign_lock_key = int(kwargs['instance_id']) % huggingface_chunk_size
            kwargs["mp_lock"] = lock_lst[assign_lock_key]
            kwargs["mp_plan_dict"] = mp_plan_dict
            kwargs['huggingface_chunk_size'] = huggingface_chunk_size
            
        
        # multiprocess 
        # ! need to run `data/03_primary/llm_plan_generation_dataset/debug.py` before running this
        # ! debug, we can set Pool to be DummyPool to debug
        # with DummyPool(1) as p:
        with Pool(mp_worker) as p:
            r = list(tqdm(p.imap_unordered(mp_create_jsonl, construction_args_collection, chunksize=20), total=len(construction_args_collection), desc="Creating JSONL"))
        
                 
    if not has_debug_huggingface_dataset:
        # debug jsonl dataset 
        print("HuggingFace Dataset Dir")
        print(huggingface_dataset_dir)
        dataset = None
        from datasets import disable_progress_bar
        disable_progress_bar()       
        # for config_type in CONFIG_TYPES:
        for config_type in [CONFIG_TYPES.t0_t5]:
            debug_config_name = config_type.name
            if dataset is not None:
                del dataset
            debug_split_name = SPLIT_NAMES.test_longer_horizon.name
            dataset_debug = load_dataset(huggingface_dataset_dir, debug_config_name, split=debug_split_name)
            # upload to huggingface hub 
            
            
            # print(dataset)
            print(f"Debugging {debug_config_name} {debug_split_name}")
            
            
            # init a tokenizer 
            if if_qwen:
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
            else:
                tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")
            def compute_lengths(examples):
                if if_qwen:
                    tokens = None 
                else:
                    tokens = tokenizer(examples['full_text_prompt'], truncation=False, padding=False)
                if if_qwen:
                    temp = examples['output']
                else:
                    temp = examples['plan_text']
                plan_text_tokens = tokenizer(temp, truncation=False, padding=False)
                if tokens is not None:
                    return {'length': [len(t) for t in tokens['input_ids']],
                        'plan_length': [len(t) for t in plan_text_tokens['input_ids']]
                        }
                else:
                    return {'plan_length': [len(t) for t in plan_text_tokens['input_ids']]}
                    
            
            # Map the function over the dataset
            dataset_debug = dataset_debug.map(compute_lengths, batched=True)
            
            # Get length 
            if 'length' in dataset_debug.column_names:
                lengths = dataset_debug['length']
                # determine the max length
                max_length = max(lengths)
                print(f"Recommended max_seq_length: {max_length}") # 6264 -> 8192
            
            max_plan_length = max(dataset_debug['plan_length']) # 5830
            print(f"Recommended response_length for {'qwen' if if_qwen else 'unsloth'}: {max_plan_length}")
            
            # t4 
            # Recommended max_seq_length: 9021
            # Recommended max_plan_length: 7930
            
        
    if has_create_huggingface_dataset and has_debug_huggingface_dataset:
        print("Great! You have created the dataset and debug jsonl dataset")
        print("Please remember to put the raw data into the huggingface dataset folder and zip it")
        
        dataset = None
        for config_type in [CONFIG_TYPES.accu_t4]:
            debug_config_name = config_type.name
            if dataset is not None:
                del dataset
            dataset = load_dataset(huggingface_dataset_dir, debug_config_name)
            # shuffle the dataset
            dataset = dataset.shuffle(seed=42)
            dataset = dataset.flatten_indices()
            # check the size of the testing data
            debug_split_name = SPLIT_NAMES.train.name
            print(f"Size of {debug_config_name} {debug_split_name}: {len(dataset[debug_split_name])}")

            # if greater than 1600, randomly sample 1600
            if len(dataset[debug_split_name]) > 1600:
                dataset = dataset[debug_split_name].shuffle(seed=seed).select(range(1600))
                print(f"Size of {debug_config_name} {debug_split_name} after sampling: {len(dataset)}")
            