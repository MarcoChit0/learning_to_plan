"""
This is a boilerplate pipeline 'plan_validation_probing'
generated using Kedro 0.19.6
"""
import pickle
import torch
import time
from pathlib import Path

import wandb
from better_language_model_for_plan_generation.pipelines.plan_validation_probing.cond_prob import compare_phrase_probabilities

from better_language_model_for_plan_generation.pipelines.plan_validation_probing.obtain_validation_dataset import obtain_validation_dataset
from icecream import ic
from accelerate import Accelerator
from accelerate.utils import gather_object 
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np


from better_language_model_for_plan_generation.utils.constant import SPLIT_NAMES
if torch.cuda.is_available():
    from transformers import AutoModelForCausalLM, AutoTokenizer


#! calculate the conditional probability of the phrases
#! output to the reporting folder as csv file for further analysis
#! we need to test all the cases, including the reinforcement learning cases


def examine_dataset(dataset, freeze=False):
    print("---*** Examine the dataset ---")    
    # get the first 4 examples
    for i in range(4):
        print("--- Valid Example --- ")
        print(dataset[i]['valid_example_output'])
        print("--- Invalid Example --- ")
        print(dataset[i]['invalid_example_output'])
        print("--- Raw Example --- ")
        print(dataset[i]['raw_example_output'])
        time.sleep(5)
    if freeze:
        time.sleep(1000000)


# ! NODE
def probing_language_model_to_identify_mistakes(probing_cfg, model_info_cfg):
    
    # ! when we use llama_factory's model, we need to call `accelerate launch` rather than `kedro run`, so we need to save the config cfg to a local file and then re run the pipeline

    # HF_HOME backup 
    HF_HOME = os.environ.get("HF_HOME")
    if HF_HOME is None: # in case the env var is not set
        os.environ["HF_HOME"] = os.path.join(
            os.environ["WORKING_DIR"], "data/01_raw/huggingface"
        )
        print(os.environ["HF_HOME"])
        Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    
    probing_data_cache_dir = os.path.join(os.environ["WORKING_DIR"], probing_cfg["probing_data_cache_dir"])
    Path(probing_data_cache_dir).mkdir(parents=True, exist_ok=True)
    
    config_type_name = model_info_cfg["config_type_name"]
    model_config_type = config_type_name
    probing_dataset_cache_path = os.path.join(probing_data_cache_dir, f"{model_config_type}_probing_dataset_dir")
    
    
    checkpoint_dir = model_info_cfg["checkpoint_dir"]
    if checkpoint_dir == "":
        raise ValueError(f"checkpoint_dir is empty for config_type_name: {config_type_name}")
    checkpoint_dir = os.path.join(os.environ["WORKING_DIR"], checkpoint_dir)
    
    is_llama_factory = model_info_cfg["is_llama_factory"]
    assert is_llama_factory, "Only support llama factory model for now"
    
    # init accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=3))
    accelerator = Accelerator(
        kwargs_handlers=[kwargs],
    )
    
    
    model_name = checkpoint_dir # for llama factory, the model name is the checkpoint dir
    
    
    # Load dataset (raw)
    test_split_name = probing_cfg['test_split_name']
    if isinstance(test_split_name, list):
        # TODO we need to handle multiple splits
        # TODO handle multiple splits save to disk with different names
        multiple_split_flag = True
    else:
        multiple_split_flag = False
        
    raw_dataset_config = probing_cfg['probing_data_config_type']
 
    dataset_path = os.path.join(os.environ["WORKING_DIR"], probing_cfg["huggingface_dataset_dir"])
    
    # * we need to do it twice, one for only main process to prepare the dataset, the other for all processes to load the dataset
    if accelerator.is_main_process:
            
        probing_eval_dataset_raw = load_dataset(
            dataset_path,
            raw_dataset_config,
            split=test_split_name,
        )
        
        accelerator.print("Loaded Raw Dataset")
        
        freeze_for_debug = probing_cfg['freeze_for_debug']
        
        # prepare the probing dataset
        if multiple_split_flag:
            for split_idx, split_name in enumerate(test_split_name):
                probing_eval_dataset = obtain_validation_dataset(
                    model_cfg = model_info_cfg,
                    model_config_type = model_config_type,
                    previous_dataset = probing_eval_dataset_raw[split_idx],
                    cache_path = probing_dataset_cache_path+ f"_{split_name}", 
                )
                assert os.path.exists(probing_dataset_cache_path+ f"_{split_name}"), f"Probing dataset cache path not found: {probing_dataset_cache_path+ f'_{split_name}'}"
                examine_dataset(probing_eval_dataset, freeze=freeze_for_debug)
        
        else:
            probing_eval_dataset = obtain_validation_dataset(
                model_cfg = model_info_cfg,
                model_config_type = model_config_type,
                previous_dataset = probing_eval_dataset_raw,
                cache_path = probing_dataset_cache_path, 
            )
            assert os.path.exists(probing_dataset_cache_path), f"Probing dataset cache path not found: {probing_dataset_cache_path}"
            examine_dataset(probing_eval_dataset, freeze=freeze_for_debug)
            
        
        accelerator.print("Prepared Probing Dataset")
        del probing_eval_dataset_raw
        del probing_eval_dataset
        
        # -- Wandb Init 
        if probing_cfg['wandb_record']:
            job_name = f"Probing Eval LLM on Plan Gen - Config: {config_type_name} {'PPO Ver' if 'ppo' in checkpoint_dir else ''}"
            merge_dict = {**probing_cfg, **model_info_cfg}
            wandb_kwargs = {
                "project": "ACCE LM for Plan Generation",
                "name": job_name,
                "config": merge_dict,
                "tags": ["probing_eval", f"llama_factory_{is_llama_factory}"],
            }
            wandb.init(**wandb_kwargs)
            
            # create wandb table 
            wandb_table = wandb.Table(columns=["mistake_identification_precision_val", "mistake_identification_recall_val", "data_split_name"])
            wandb_image_lst = []
        
        
    accelerator.wait_for_everyone()
    # Load dataset (cached)
    if multiple_split_flag:
        probing_dataset_cache_path_lst = [probing_dataset_cache_path+ f"_{split_name}" for split_name in test_split_name]
    else:
        probing_dataset_cache_path_lst = [probing_dataset_cache_path]
        
    for probing_dataset_cache_path in probing_dataset_cache_path_lst:
        assert os.path.exists(probing_dataset_cache_path), f"Probing dataset cache path not found: {probing_dataset_cache_path}"
        probing_eval_dataset = load_from_disk(probing_dataset_cache_path)
        
        # ! -- Probing the model -- 
        message= [f"Hello this is GPU {accelerator.process_index}"]
        messages=gather_object(message)
        accelerator.print(messages)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": accelerator.process_index},
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # - Print Stats
        gpu_stats = torch.cuda.get_device_properties(accelerator.device)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        accelerator.print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        accelerator.print(f"{start_gpu_memory} GB of memory reserved.")
        
        # sync GPUs
        accelerator.wait_for_everyone()  
        
        with accelerator.split_between_processes(probing_eval_dataset) as split_probing_eval_dataset:

            # ! --- Init output results 
            results = dict(
                token_probability_contrast_ratio_mistake_case = [],
                token_probability_contrast_ratio_valid_case = [],
                mistake_identification_precision = None, # identify true mistake / all identified mistake
                mistake_identification_recall = None, # identify true mistake / all true mistake
                true_identified_mistake_count = 0, # TP
                all_actual_mistake_count = 0, # TP + FN, used for recall
                all_identified_mistake_count = 0, # TP + FP, used for precision
            )
            # ! --- End init output results
            # ! for loop and do the probing
            if accelerator.is_main_process:
                my_iter = tqdm(split_probing_eval_dataset, desc="Probing the model", total=len(split_probing_eval_dataset))
            else:
                my_iter = split_probing_eval_dataset
                
            for data_entry in my_iter:
                # get system, user, valid_example_output and invalid_example_output
                system = data_entry["system"]
                user = data_entry["user"]
                valid_example_output = data_entry["valid_example_output"]
                invalid_example_output = data_entry["invalid_example_output"]
                raw_example_output = data_entry["raw_example_output"]
                # generate valid and invalid context 
                valid_massages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": valid_example_output},
                ]
                valid_text = tokenizer.apply_chat_template(
                    valid_massages,
                    tokenize=False,
                    add_generation_prompt=False, # we do not need the generation prompt
                )
                invalid_massages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": invalid_example_output},
                ]
                invalid_text = tokenizer.apply_chat_template(
                    invalid_massages,
                    tokenize=False,
                    add_generation_prompt=False, # we do not need the generation prompt
                )
                # breakpoint()
                valid_text = valid_text.replace(tokenizer.eos_token, "")
                invalid_text = invalid_text.replace(tokenizer.eos_token, "") # ! remove the eos token as we want to predict the next token
                
                
                back_token_prob_for_valid, newline_token_prob_for_valid = compare_phrase_probabilities(
                    model,
                    tokenizer,
                    valid_text,
                )
                back_token_prob_for_invalid, newline_token_prob_for_invalid = compare_phrase_probabilities(
                    model,
                    tokenizer,
                    invalid_text,
                )
                # breakpoint()
                # feed to the results
                results["token_probability_contrast_ratio_mistake_case"].append(
                    back_token_prob_for_invalid / newline_token_prob_for_invalid
                ) # > 1 better
                results["token_probability_contrast_ratio_valid_case"].append(
                    back_token_prob_for_valid / newline_token_prob_for_valid
                ) # < 1 better 
                
                results['true_identified_mistake_count'] += 1 if back_token_prob_for_invalid > newline_token_prob_for_invalid else 0 # TP
                results['all_actual_mistake_count'] += 1 # always increment because we have invalid case
                
                
                results['all_identified_mistake_count'] += 1 if back_token_prob_for_invalid > newline_token_prob_for_invalid else 0 # FP
                
                results['all_identified_mistake_count'] += 1 if back_token_prob_for_valid > newline_token_prob_for_valid else 0 # TP
                
            # -- Calculate the precision and recall
            # calculate the precision
            results['mistake_identification_precision'] = results['true_identified_mistake_count'] / results['all_identified_mistake_count'] if results['all_identified_mistake_count'] > 0 else 0
            
            # calculate the recall
            results['mistake_identification_recall'] = results['true_identified_mistake_count'] / results['all_actual_mistake_count'] if results['all_actual_mistake_count'] > 0 else 0
            
            results = [ results ] # transform to list, otherwise gather_object() will not collect correctly
            
        # wait for all processes to finish
        accelerator.print(f"GPU {accelerator.process_index} finished probing")
        accelerator.wait_for_everyone()
        
        results_gathered = gather_object(results)
        
        # -- Post processing the results
        if accelerator.is_main_process:
            gathered_mistake_identification_precision = [] 
            gathered_mistake_identification_recall = []
            gathered_token_probability_contrast_ratio_mistake_case = []
            gathered_token_probability_contrast_ratio_valid_case = []
            for results in results_gathered:
                gathered_mistake_identification_precision.append(results['mistake_identification_precision'])
                gathered_mistake_identification_recall.append(results['mistake_identification_recall'])
                gathered_token_probability_contrast_ratio_mistake_case.extend(results['token_probability_contrast_ratio_mistake_case'])
                gathered_token_probability_contrast_ratio_valid_case.extend(results['token_probability_contrast_ratio_valid_case'])
                
            # average the precision and recall
            gathered_mistake_identification_precision = sum(gathered_mistake_identification_precision) / len(gathered_mistake_identification_precision)
            gathered_mistake_identification_recall = sum(gathered_mistake_identification_recall) / len(gathered_mistake_identification_recall)
            
        # -- End post processing the results
        if accelerator.is_main_process:
            # Save the results
            output_dict = {
                "mistake_identification_precision_val": gathered_mistake_identification_precision,
                "mistake_identification_recall_val": gathered_mistake_identification_recall,
                "token_probability_contrast_ratio_mistake_case_lst": gathered_token_probability_contrast_ratio_mistake_case,
                "token_probability_contrast_ratio_valid_case_lst": gathered_token_probability_contrast_ratio_valid_case,
            }
            pkl_save_path = os.path.join(os.environ["WORKING_DIR"], f"data/08_reporting/language_model_token_probing/{model_config_type}_probing_results{'_ppo' if 'ppo' in checkpoint_dir else ''}.pkl")
            Path(pkl_save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(pkl_save_path, "wb") as f:
                pickle.dump(output_dict, f)
                
            # End Wandb 
            if probing_cfg['wandb_record']:
                # save mistake_identification_precision_val and mistake_identification_recall_val to wandb
                # save lst as boxplot to wandb
                # TODO box plot is not available in wandb, need to check the latest version
                prefix = probing_dataset_cache_path.split("/")[-1]
             
                wandb_table.add_data(gathered_mistake_identification_precision, gathered_mistake_identification_recall, prefix)
                
                # draw bot plot and save to PILImage
                wandb_image_lst.append(
                    {
                        'col_name': f"{prefix}.token_probability_contrast_ratio_mistake_case",
                        'lst': gathered_token_probability_contrast_ratio_mistake_case
                        }
                )
                
                wandb_image_lst.append(
                    {
                        'col_name': f"{prefix}.token_probability_contrast_ratio_valid_case",
                        'lst': gathered_token_probability_contrast_ratio_valid_case
                        }
                )
                
   
    if accelerator.is_main_process:
        if probing_cfg['wandb_record']:
            # save the table to wandb
            wandb.log({"mistake_identification_precision_recall_table": wandb_table})
            # save the box plot to wandb
            data = []
            cols = [] 
            for image_dict in wandb_image_lst:
                data.append(image_dict['lst'])
                cols.append(image_dict['col_name'])
            fig, ax = plt.subplots()
            ax.boxplot(data)
            ax.set_title("Token Probability Contrast Ratio")
            ax.set_xticklabels(cols)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            img = Image.open(buf)
            
            wandb.log({"token_probability_contrast_ratio_box_plot": wandb.Image(img)})
            
            wandb.finish()


if __name__ == "__main__":
    import json 
    import logging.config 
    from kedro.config import OmegaConfigLoader
    from kedro.framework.project import settings
    import os  
    
    conf_path = os.path.join(os.environ["WORKING_DIR"], settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path, env= "base")
    conf_params = conf_loader['parameters']
    config_type_name = os.environ.get("EVAL_CONFIG_TYPE", "accu_t1_t4")
    print(f"config_type_name for probing: {config_type_name}")
    time.sleep(2)
    
    probing_cfg = conf_params['plan_validation_probing_params']
    model_info_cfg = conf_params['plan_validation_probing_testing_model_info'][f'{config_type_name}_params']
    
    print("Please using accelerate launch to run this pipeline")
    
    num_process = len(os.environ.get("CUDA_VISIBLE_DEVICES", "1").split(","))
    print("Num process:", num_process)
    probing_language_model_to_identify_mistakes(probing_cfg, model_info_cfg)
    