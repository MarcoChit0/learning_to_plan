# generate.py 

import datetime
import os
import torch
from tqdm import tqdm
from learning_to_plan import models
import google.generativeai as genai

# Import project modules
import learning_to_plan.config as config
from learning_to_plan import task # Import task module
import numpy as np

logger = config.get_logger(__name__)
from typing import Union, Optional
# --- Batch Generation from File (Modified) ---

def generate_batch(
        model_name: str, 
        domain:str, 
        number_of_instances:Union[str, int] = "all", 
        few_shot:int = 0, 
        random_seed:int = 42,
        prompt_type:task.Task.PromptType = task.Task.PromptType.IO,
        **generation_kwargs):
    start_time = datetime.datetime.now()
    rng = np.random.RandomState(random_seed)

    logger.info(
        f"Starting generation batch with model '{model_name}' – time: {start_time}" # Use logger
    )
    generation_kwargs['is_trainable'] = False
    model = models.get_model(model_name=model_name, **generation_kwargs)
    # --- Get tasks from dataset ---
    try:
        if number_of_instances == "all":
            tasks = task.get_tasks(filter_by_domain=domain, filter_by_type=task.Task.Type.TEST)
        elif number_of_instances == "basic":
            tasks = task.get_tasks(filter_by_domain=domain, filter_by_type=task.Task.Type.TEST, is_longer_plan=False)
        elif number_of_instances == "long":
            tasks = task.get_tasks(filter_by_domain=domain, filter_by_type=task.Task.Type.TEST, is_longer_plan=True)
        elif isinstance(number_of_instances, int):
            tasks = task.get_tasks(filter_by_domain=domain, filter_by_type=task.Task.Type.TEST, number_of_instances=number_of_instances)
        else:
            raise ValueError(f"Invalid value for number_of_instances: {number_of_instances}. Must be 'all', 'basic', 'long', or a positive integer.")
        assert len(tasks) > 0, f"No tasks found for generation."
        logger.info(f"Getting {len(tasks)} tasks for generation.") # Use logger
    except Exception as e:
        logger.error(f"Error getting tasks for generation: {e}", exc_info=True)
        raise e
    
    try:
        possible_few_shot_examples = set()
        if few_shot > 0:
            val = task.get_tasks(domain=domain, type=task.Task.Type.VALIDATION, is_longer_plan=True)
            train = task.get_tasks(domain=domain, type=task.Task.Type.TRAIN, is_longer_plan=True)
            possible_few_shot_examples = set(val + train)
    except Exception as e:
        logger.error(f"Error getting possible CoT examples: {e}", exc_info=True)
        raise e

    # --- Generate Plans ---
    logger.info("Starting plan generation loop...") # Use logger
    for t in tqdm(tasks, total=len(tasks), desc="Generating plans"):
        few_shot_examples = set()
        if few_shot > 0:
            few_shot_examples = set(
                rng.choice(
                    list(possible_few_shot_examples),
                    size=min(few_shot, len(possible_few_shot_examples)),
                    replace=False
                )
            )
        try:
            model.generate(
                task=t,
                few_shot_examples=few_shot_examples,
                **generation_kwargs
            )
        except Exception as e:
            logger.error(f"Error generating plan for task {t._id} with model {model_name}: {e}", exc_info=True)
    logger.info(f"Plan generation loop completed for {len(tasks)} instances.")
    try:
        logger.info(f"Saving generated plans to {model._model_dir_path}")
        model.save_generated_plans()
        logger.info(f"Generated plans saved successfully.")
    except Exception as e:
        logger.error(f"Error saving generated plans: {e}", exc_info=True)

    end_time = datetime.datetime.now()
    logger.info(f"Generation batch finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}. Total time: {end_time - start_time}") # Use logger