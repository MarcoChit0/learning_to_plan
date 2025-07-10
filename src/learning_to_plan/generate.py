# generate.py 

import datetime
from typing import Optional
from tqdm import tqdm
from learning_to_plan.models import utils as model_utils

# Import project modules
import learning_to_plan.config as config
from learning_to_plan.data import task
logger = config.get_logger(__name__)
from typing import Union
from learning_to_plan.data import generated_plans
from learning_to_plan.prompt_builder import utils as prompt_builder_utils
from learning_to_plan.domain_translators import utils as domain_translator_utils
# --- Batch Generation from File (Modified) ---

def generate_batch(
        model_name: str, 
        domain:str, 
        number_of_instances:Union[str, int] = "all",
        task_type: Optional[task.Task.TYPE] = None,
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
        model = model_utils.get_model(model_name=model_name)
        generation_kwargs['is_trainable'] = False
        model.setup(**generation_kwargs)
    except Exception as e:
        logger.error(f"Error initializing model '{model_name}': {e}", exc_info=True)
        raise e
    try:
        task_selection_kwargs = {
            'filter_by_domain': domain,
            'filter_by_pourpose': task.Task.POURPOSE.TEST
        }
        if task_type:
            assert isinstance(task_type, (str, task.Task.TYPE)), "task_type must be a string or a Task.TYPE enum."
            task_selection_kwargs['filter_by_type'] = task_type

        tasks : set[task.Task] = task.task_database.get(
            **task_selection_kwargs
        )
        assert len(tasks) > 0, f"No tasks found for domain '{domain}', whose selection kwargs where {task_selection_kwargs}."
        assert isinstance(tasks, set), "Tasks must be a set of Task objects."
        assert all(isinstance(t, task.Task) for t in tasks), "All items in tasks must be Task objects."
        logger.info(f"Found {len(tasks)} tasks for domain '{domain}', whose selection kwargs where {task_selection_kwargs}.")

        if number_of_instances != "all":
            assert isinstance(number_of_instances, int) and number_of_instances > 0, "number_of_instances must be 'all' or a positive integer."
            l = min(len(tasks), number_of_instances)

            # select l tasks from the set evenly distributed
            import numpy as np
            indices = np.linspace(0, len(tasks) - 1, num=l, dtype=int)
            tasks = set(sorted(tasks)[i] for i in indices)
            assert len(tasks) == l, f"Expected {l} tasks, but got {len(tasks)} after selection."
            
        logger.info(f"Generating plans for {len(tasks)} tasks in domain '{domain}' with prompt type='{prompt_type}',num_samples={num_samples}, and type='{task_type}'.")
    
    except Exception as e:
        raise ValueError(
            f"Error retrieving tasks for domain '{domain}', whose selection kwargs where {task_selection_kwargs}: {e}"
        ) from e

    # --- Generate Plans ---
    logger.info("Starting plan generation loop...") # Use logger
    for t in tqdm(tasks, total=len(tasks), desc="Generating plans"):
        prompt_metadata = prompt_builder_utils.get_metadata(**generation_kwargs)
        model_metadata = model.get_metadata()
        _generated_plans = generated_plans.generated_plan_database.get(
            filter_by_task_id=t.id,
            filter_by_prompt_type=prompt_type,
            filter_by_model_metadata=model_metadata,
            filter_by_prompt_metadata=prompt_metadata
        )
        if overwrite_generated_plans:
            logger.info(
                f"Overwriting {len(_generated_plans)} existing generated plans for task {t.id} with prompt {prompt_type}."
            )
            try:
                generated_plans.generated_plan_database.delete(obj=_generated_plans)
            except Exception as e:
                logger.error(f"Error deleting existing generated plans for task {t.id}: {e}", exc_info=True)
                raise e
            logger.info(
                f"Generated plans for task {t.id} with prompt {prompt_type} were deleted successfully."
            )

        # Determine how many new samples we need
        existing_count = len(_generated_plans)
        if num_samples <= existing_count:
            logger.info(
                f"Skipping generation: requested {num_samples} samples but already have {existing_count} for task {t.id} with prompt {prompt_type}."
            )
            continue
        to_generate = num_samples - existing_count
        if existing_count > 0:
            logger.info(
                f"Task {t.id} with prompt {prompt_type} has {existing_count}/{num_samples} plans; "
                f"generating {to_generate} more."
            )
        else:
            logger.info(
                f"Task {t.id} with prompt {prompt_type} has no plans; generating {to_generate} samples."
            )
        for _ in tqdm(
            range(to_generate),
            desc=f"Generating samples for task {t.id}",
            unit="sample",
            leave=False
        ):
            chat = prompt_builder_utils.get_chat(t=t, with_plan=False, **generation_kwargs)
            try:
                status = config.STATUS.ERROR
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
                            pddl_plan = raw_plan
                        else:
                            try:
                                pddl_plan = domain_translator_utils.translate_natural_language_plan_to_pddl(t, raw_plan)
                            except Exception as e:
                                error_message = "Error translating plan to PDDL: " + str(e)
                    if error_message is None:
                        if pddl_plan.replace(" ", "").replace("\n", "") == "":
                            error_message = "Generated plan is empty after translation."
                        else:
                            status = config.STATUS.OK
            except Exception as e:
                error_message = "Error generating sample : " + str(object=e)
            finally:
                if status == config.STATUS.ERROR:
                    logger.warning(
                        f"Failed to generate valid plan for task {t.id} with prompt {prompt_type}: {error_message}",
                        exc_info=True
                    )
                    pddl_plan = None
                    raw_plan = None
                    validity = generated_plans.GeneratedPlan.VALIDITY.INVALID
                else:
                    logger.info(f"Generated valid plan for task {t.id} with prompt {prompt_type}.")
                    error_message = None
                    validity = generated_plans.GeneratedPlan.VALIDITY.UNCHECKED
                new_generated_plan = generated_plans.GeneratedPlan(
                    task_id=t.id,
                    raw_plan=raw_plan,
                    pddl_plan=pddl_plan,
                    model_metadata=model_metadata,
                    prompt_metadata=prompt_metadata,
                    status=status,
                    validity=validity,
                    error_message=error_message
                )
                try:
                    generated_plans.generated_plan_database.add(obj=new_generated_plan)
                    logger.info(f"Added new generated plan for task {t.id} with prompt {prompt_type} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
                except Exception as e:
                    logger.error(
                        f"Error adding new generated plan for task {t.id} with prompt {prompt_type}: {e}",
                        exc_info=True
                    )
                    raise e

    end_time = datetime.datetime.now()
    logger.info(f"Generation batch finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}. Total time: {end_time - start_time}") # Use logger