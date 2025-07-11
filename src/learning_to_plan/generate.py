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
from learning_to_plan.data import generated_plan
from learning_to_plan.prompt_builder import utils as prompt_builder_utils
from learning_to_plan import database
# --- Batch Generation from File (Modified) ---

def generate_batch(
        model_name: str,
        domain: str,
        prompt_type: config.PROMPT_TYPE,
        number_of_instances: Union[str, int] = "all",
        task_type: Optional[task.Task.TYPE] = None,
        num_samples: int = 1,
        overwrite_generated_plans: bool = False,
        **generation_kwargs):
    start_time = datetime.datetime.now()

    logger.info(
        f"Starting generation batch with model '{model_name}' – time: {start_time}" # Use logger
    )
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
            'filter_by_purpose': task.Task.purpose.TEST
        }
        if task_type:
            assert isinstance(task_type, (str, task.Task.TYPE)), "task_type must be a string or a Task.TYPE enum."
            task_selection_kwargs['filter_by_type'] = task_type

        tasks : set[task.Task] = database.task_database.get(
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

    prompt_builder = prompt_builder_utils.get_prompt_builder(prompt_type=prompt_type, **generation_kwargs)

    # --- Generate Plans ---
    logger.info("Starting plan generation loop...") # Use logger
    for t in tqdm(tasks, total=len(tasks), desc="Generating plans"):
        prompt_metadata = prompt_builder.get_metadata()
        model_metadata = model.get_metadata()
        _generated_plans = database.generated_plan_database.get(
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
                database.generated_plan_database.delete(obj=_generated_plans)
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
            chat = prompt_builder.get_chat(t=t, with_plan=False, **generation_kwargs)
            try:
                response = model.generate_single_sample(
                    chat=chat,
                    **generation_kwargs,
                )
                try:
                    pddl_plan = prompt_builder.process_response(response=response)
                    error_message = None
                except Exception as e:
                    pddl_plan = None
                    error_message = f"Error processing response: {e}"
            except Exception as e:
                error_message = f"Could not generate plan for task {t.id} with prompt {prompt_type}: {e}"
                pddl_plan = None
            finally:
                gen_plan = generated_plan.GeneratedPlan(
                    task_id=t.id,
                    pddl_plan=pddl_plan,
                    model_metadata=model_metadata,
                    prompt_metadata=prompt_metadata,
                    validity=generated_plan.GeneratedPlan.VALIDITY.UNCHECKED,
                    error_message=error_message
                )
            try:
                database.generated_plan_database.add(obj=gen_plan)
                logger.info(
                    f"Generated plan for task {t.id} with prompt {prompt_type} saved successfully."
                )
            except Exception as e:
                logger.error(
                    f"Error saving generated plan for task {t.id} with prompt {prompt_type}: {e}",
                    exc_info=True
                )
                raise e

    end_time = datetime.datetime.now()
    logger.info(f"Generation batch finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}. Total time: {end_time - start_time}") # Use logger