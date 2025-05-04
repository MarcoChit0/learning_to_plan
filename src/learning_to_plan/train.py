import os
import datetime
import datasets
from learning_to_plan import models
import learning_to_plan.config as config
logger = config.get_logger(__name__)

def run_training_procedure(model_name, domain,  **train_kwargs):
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting training at {start_time_str}")

    model_checkpoint_dir = config.get_checkpoint_dir(domain, model_name)
    config.create_necessary_dirs(model_checkpoint_dir)
    logger.info(f"Checkpoints will be saved to: {model_checkpoint_dir}")
    try:
        model = models.get_model(model_name=model_name, checkpoint_dir=model_checkpoint_dir)
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
        raise e

    # --- Load and Prepare Dataset ---
    try:
        from learning_to_plan import task
        logger.info(f"Loading training and validation datasets for domain: {domain}.")
        train_tasks:set[task.Task] = task.get_tasks(domain=domain, type=task.Task.Type.TRAIN)
        validation_tasks:set[task.Task] = task.get_tasks(domain=domain, type=task.Task.Type.VALIDATION)

        # TODO: REMOVE THIS LATER
        train_tasks = set(sorted(train_tasks)[:100])

        # Make the prompts that will be used for training and validation
        eos_token = None
        if hasattr(model, "tokenizer"):
            try:
                eos_token = model.tokenizer.eos_token
            except AttributeError:
                logger.warning(f"Model {model_name} does not have an EOS token. Using None.")
        
        # Convert the tasks to prompts
        training_prompts : list[str]  = [t.get_prompt(eos_token=eos_token, with_plan=True) for t in train_tasks]
        logger.info(f"Training prompts created with {len(training_prompts)} examples.")
        validation_prompts : list[str]  = [t.get_prompt(eos_token=eos_token, with_plan=True) for t in validation_tasks]
        logger.info(f"Validation prompts created with {len(validation_prompts)} examples.")

        # Create datasets.Dataset objects
        train_dataset = datasets.Dataset.from_dict({"text": training_prompts})
        logger.info(f"Training dataset created with {len(train_dataset)} examples.")
        validation_dataset = datasets.Dataset.from_dict({"text": validation_prompts})
        logger.info(f"Validation dataset created with {len(validation_dataset)} examples.")
        
        # Remove the tasks and prompts from memory
        del train_tasks, validation_tasks, training_prompts, validation_prompts


        # Create DatasetDict
        # TODO: UNCOMMENT THIS LATER
        # dataset = datasets.DatasetDict({
        #     'train': train_dataset,
        #     'validation': validation_dataset
        # })
        # logger.info(f"Number of validation examples: {len(dataset['validation'])}")
        dataset = datasets.DatasetDict({
            'train': train_dataset,
            'validation': train_dataset # TODO: REMOVE THIS LATER
        })
        logger.info(f"Dataset converted to DatasetDict successfully: {dataset}")
        logger.info(f"Number of training examples: {len(dataset['train'])}")
        logger.info(f"Number of validation examples: {len(dataset['validation'])}")

    except Exception as e:
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        raise e

    # --- Train ---
    try:
        model.train(checkpoint_dir=model_checkpoint_dir, dataset=dataset, **train_kwargs)
        end_time = datetime.datetime.now()
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Training completed at {end_time_str}")
    except Exception as e:
        logger.error(f"Error when training model {model_name}: {e}", exc_info=True)
        raise e