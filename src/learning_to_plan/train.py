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
        model = models.get_model(model_name=model_name, **train_kwargs)
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
        raise e

    # --- Load and Prepare Dataset ---
    try:
        from learning_to_plan import task
        logger.info(f"Loading training and validation datasets for domain: {domain}.")
        train_tasks:set[task.Task] = task.get_tasks(filter_by_domain=domain, filter_by_type=task.Task.Type.TRAIN)
        validation_tasks:set[task.Task] = task.get_tasks(filter_by_domain=domain, filter_by_type=task.Task.Type.VALIDATION)

        # TODO: REMOVE THIS LATER
        train_tasks = set(sorted(train_tasks)[:10])

        # Convert the tasks to prompts
        training_prompts : list[str]  = [t.get_prompt(with_plan=True) for t in train_tasks]
        with open(os.path.join(model_checkpoint_dir, "training_prompts.txt"), "w") as f:
            for prompt in training_prompts:
                f.write(prompt + "\n")
        logger.info(f"Training prompts created with {len(training_prompts)} examples.")
        validation_prompts : list[str]  = [t.get_prompt(with_plan=True) for t in validation_tasks]
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

    # --- Tokenize and Process Dataset ---
    try:
        tokenized_train_dataset = dataset['train'].map(
            model.tokenize,
            batched=True,
            remove_columns=["text"],
            desc="Processing training dataset",
        )
        tokenized_eval_dataset = dataset['validation'].map(
            model.tokenize,
            batched=True,
            remove_columns=["text"],
            desc="Processing validation dataset",
        )
        logger.info("Dataset tokenization and processing complete.")
        logger.info(f"Processed Training Dataset Features: {tokenized_train_dataset.features}")
        logger.info(f"Processed Validation Dataset Features: {tokenized_eval_dataset.features}")
        if len(tokenized_train_dataset) > 0:
            sample_idx = 0
            logger.debug(f"--- Sample {sample_idx} (Processed) ---")
            logger.debug(f"Input IDs: {tokenized_train_dataset[sample_idx]['input_ids'][:50]}...") # Log a snippet
            logger.debug(f"Labels:    {tokenized_train_dataset[sample_idx]['labels'][:50]}...")    # Log a snippet
            decoded_input = model.decode(tokenized_train_dataset[sample_idx]['input_ids'], skip_special_tokens=False)
            labeled_tokens = [
                model.decode([tok_id]) if label_id != -100 else "[-]"
                for tok_id, label_id in zip(tokenized_train_dataset[sample_idx]['input_ids'], tokenized_train_dataset[sample_idx]['labels'])
            ]
            logger.debug(f"Decoded Input (first 300 chars): {decoded_input[:300]}...")
            logger.debug(f"Labeled Tokens (first 50): {' '.join(labeled_tokens[:50])}...")
        else: 
            raise ValueError("Tokenized training dataset is empty despite non-empty input. Check processing logic and data.")
    except Exception as e:
        logger.error(f"Error during dataset processing: {e}", exc_info=True)
        raise e


    # --- Train ---
    try:
        logger.info("Calling model.train() at %s", datetime.datetime.now())
        model.train(checkpoint_dir=model_checkpoint_dir, tokenized_train_dataset=tokenized_train_dataset, tokenized_eval_dataset=tokenized_eval_dataset, **train_kwargs)
        end_time = datetime.datetime.now()
        logger.info(f"Training completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        logger.error(f"Error when training model {model_name}: {e}", exc_info=True)
        raise e