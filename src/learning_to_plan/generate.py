# generate.py 

import datetime
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
        number_of_cot_examples:int = 0, 
        random_seed:int = 42,
        checkpoint_dir:Optional[str] = None, 
        **generation_kwargs):
    start_time = datetime.datetime.now()
    rng = np.random.RandomState(random_seed)

    logger.info(
        f"Starting generation batch with model '{model_name}' – time: {start_time}" # Use logger
    )

    model = models.get_model(model_name=model_name, checkpoint_dir=checkpoint_dir)
    

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
        possible_cot_examples = set()
        if number_of_cot_examples > 0:
            # Get possible CoT examples
            val = task.get_tasks(domain=domain, type=task.Task.Type.VALIDATION, is_longer_plan=True)
            train = task.get_tasks(domain=domain, type=task.Task.Type.TRAIN, is_longer_plan=True)
            possible_cot_examples = set(val + train)
    except Exception as e:
        logger.error(f"Error getting possible CoT examples: {e}", exc_info=True)
        raise e

    # --- Generate Plans ---
    logger.info("Starting plan generation loop...") # Use logger
    for t in tqdm(tasks, total=len(tasks), desc="Generating plans"):
        cot_examples = set()
        if number_of_cot_examples > 0:
            cot_examples = set(
                rng.choice(
                    list(possible_cot_examples),
                    size=min(number_of_cot_examples, len(possible_cot_examples)),
                    replace=False
                )
            )
        try:
            model.generate(
                task=t,
                cot_examples=cot_examples,
                **generation_kwargs
            )
            logger.info(f"Task prompt:\n{t.get_prompt(with_plan=False, cot_examples=cot_examples)}")
            logger.info(f"Model plans:\n{model._generated_plans}")
        except Exception as e:
            logger.error(f"Error generating plan for task {t._id} with model {model_name}: {e}", exc_info=True) # Use logger

    logger.info(f"Plan generation loop completed for {len(tasks)} instances.")

    end_time = datetime.datetime.now()
    logger.info(f"Generation batch finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}. Total time: {end_time - start_time}") # Use logger