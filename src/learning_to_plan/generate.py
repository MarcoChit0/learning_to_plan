# generate.py 

import datetime
from tqdm import tqdm
from learning_to_plan import models

# Import project modules
import learning_to_plan.config as config
from learning_to_plan import task # Import task module
logger = config.get_logger(__name__)
from typing import Union
# --- Batch Generation from File (Modified) ---

def generate_batch(
        model_name: str, 
        domain:str, 
        number_of_instances:Union[str, int] = "all", 
        num_samples:int = 1,
        overwrite_generated_plans:bool = False,
        **generation_kwargs):
    start_time = datetime.datetime.now()

    logger.info(
        f"Starting generation batch with model '{model_name}' – time: {start_time}" # Use logger
    )
    prompt_type = generation_kwargs.get("prompt_type", None)
    assert prompt_type is not None, "Prompt type must be specified in generation_kwargs."
    try:
        model = models.get_model(model_name=model_name)
        generation_kwargs['is_trainable'] = False
        model.setup(**generation_kwargs)
    except Exception as e:
        logger.error(f"Error initializing model '{model_name}': {e}", exc_info=True)
        raise e
    # --- Get tasks from dataset ---
    try:
        if number_of_instances == "all":
            tasks = task.get_tasks(filter_by_domain=domain, filter_by_type=task.Task.TYPE.TEST)
        elif number_of_instances == "basic":
            tasks = task.get_tasks(filter_by_domain=domain, filter_by_type=task.Task.TYPE.TEST, is_longer_plan=False)
        elif number_of_instances == "long":
            tasks = task.get_tasks(filter_by_domain=domain, filter_by_type=task.Task.TYPE.TEST, is_longer_plan=True)
        elif isinstance(number_of_instances, int):
            tasks = task.get_tasks(filter_by_domain=domain, filter_by_type=task.Task.TYPE.TEST, number_of_instances=number_of_instances)
        else:
            raise ValueError(f"Invalid value for number_of_instances: {number_of_instances}. Must be 'all', 'basic', 'long', or a positive integer.")
        assert len(tasks) > 0, f"No tasks found for generation."
        logger.info(f"Getting {len(tasks)} tasks for generation.") # Use logger
    except Exception as e:
        logger.error(f"Error getting tasks for generation: {e}", exc_info=True)
        raise e

    # --- Generate Plans ---
    logger.info("Starting plan generation loop...") # Use logger
    for t in tqdm(tasks, total=len(tasks), desc="Generating plans"):
        prompt_metadata = t.get_prompt_metadata(**generation_kwargs)
        model_metadata = model.get_metadata()
        if overwrite_generated_plans:
            logger.info(f"Overwriting existing generated plans for task {t._id} with prompt type {prompt_type}.")
            model.overwrite_generated_plans(
                t=t,
                prompt_type=prompt_type,
                model_metadata=model_metadata,
                prompt_metadata=prompt_metadata
            )
        generated_plans_for_task_with_prompt_type = model.get_generated_plans(
            t=t,
            prompt_type=prompt_type,
            model_metadata=model_metadata,
            prompt_metadata=prompt_metadata
        )
        # Determine how many new samples we need
        existing_count = len(generated_plans_for_task_with_prompt_type)
        if num_samples <= existing_count:
            logger.info(
                f"Skipping generation: requested {num_samples} samples but already have {existing_count} for task {t._id} with prompt {prompt_type}."
            )
            continue
        to_generate = num_samples - existing_count
        if existing_count > 0:
            logger.info(
                f"Task {t._id} with prompt {prompt_type} has {existing_count}/{num_samples} plans; "
                f"generating {to_generate} more."
            )
        else:
            logger.info(
                f"Task {t._id} with prompt {prompt_type} has no plans; generating {to_generate} samples."
            )
        for _ in tqdm(
            range(to_generate),
            desc=f"Generating samples for task {t._id}",
            unit="sample",
            leave=False
        ):
            chat = t.get_chat(with_plan=False, **generation_kwargs)
            try:
                status = model.Content.STATUS.ERROR
                error_message = None
                pddl_plan = None
                raw_plan = None

                response = model.generate_single_sample(
                    chat=chat,
                    **generation_kwargs,
                )
                print("----.")
                print(response)
                print(".----")
                if response == "":
                    error_message = "Empty response from model."
                elif config.TOKENS.PLAN_START.value not in response:
                    error_message = f"Plan start token '{config.TOKENS.PLAN_START.value}' not found in response."
                elif config.TOKENS.PLAN_END.value not in response:
                    error_message = f"Plan end token '{config.TOKENS.PLAN_END.value}' not found in response."
                else:
                    plan_start_idx = response.index(config.TOKENS.PLAN_START.value)
                    plan_end_idx = response.index(config.TOKENS.PLAN_END.value)
                    if plan_start_idx >= plan_end_idx:
                        error_message = "Plan start token is after the end token in the response."
                    else:
                        raw_plan = response[plan_start_idx + len(config.TOKENS.PLAN_START.value):plan_end_idx].strip()
                        if prompt_type == config.PROMPT_TYPE.PDDL:
                            # The raw plan is already in PDDL format
                            pddl_plan = raw_plan
                            status = model.Content.STATUS.OK
                        else:
                            try:
                                pddl_plan = t._domain_translator.translate_natural_language_plan_to_pddl(raw_plan)
                                if pddl_plan.replace(" ", "").replace("\n", "") == "":
                                    error_message = "Translated PDDL plan is empty."
                                else:
                                    status = model.Content.STATUS.OK
                            except Exception as e:
                                error_message = "Error translating plan to PDDL: " + str(e)
                print(f"Status for task {t._id} with prompt {prompt_type} : {status}")
                print(f"Raw plan for task {t._id} with prompt {prompt_type} : {raw_plan}")
                print(f"PDDL plan for task {t._id} with prompt {prompt_type} : {pddl_plan}")
                print(f"Error message for task {t._id} with prompt {prompt_type} : {error_message}")
            except Exception as e:
                error_message = "Error generating sample : " + str(object=e)
            finally:
                if status == model.Content.STATUS.ERROR:
                    logger.warning(
                        f"Failed to generate valid plan for task {t._id} with prompt {prompt_type}: {error_message}",
                        exc_info=True
                    )
                    pddl_plan = None
                    raw_plan = None
                    validity = model.Content.VALIDITY.INVALID
                else:
                    logger.info(f"Generated valid plan for task {t._id} with prompt {prompt_type}.")
                    error_message = None
                    validity = model.Content.VALIDITY.UNCHECKED
                content = model.Content(
                        t=t,
                        prompt_type=prompt_type,
                        status=status,
                        pddl_plan=pddl_plan,
                        raw_plan=raw_plan,
                        validity=validity,
                        error_message=error_message,
                        model_metadata=model_metadata,
                        prompt_metadata=prompt_metadata
                    )
                model.add_generated_plan(content=content)
    logger.info("Plan generation loop completed.") # Use logger
    try:
        logger.info(f"Saving generated plans to {model._model_dir_path}")
        model.save_generated_plans()
        logger.info(f"Generated plans saved successfully.")
    except Exception as e:
        logger.error(f"Error saving generated plans: {e}", exc_info=True)

    end_time = datetime.datetime.now()
    logger.info(f"Generation batch finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}. Total time: {end_time - start_time}") # Use logger