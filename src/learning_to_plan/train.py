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
        model = models.get_model(model_name=model_name, checkpoint_dir=model_checkpoint_dir, is_trainable=True, **train_kwargs)
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
        raise e

    # --- Load and Prepare Dataset ---
    try:
        from learning_to_plan import task
        logger.info(f"Loading training and validation datasets for domain: {domain}.")
        train_tasks:set[task.Task] = task.get_tasks(filter_by_domain=domain, filter_by_type=task.Task.Type.TRAIN)
        validation_tasks:set[task.Task] = task.get_tasks(filter_by_domain=domain, filter_by_type=task.Task.Type.VALIDATION)

        # Convert the tasks to prompts
        training_chats = [t.get_prompt_componenets() for t in train_tasks]
        logger.info(f"Training dataset loaded with {len(training_chats)} tasks.")
        validation_chats = [t.get_prompt_componenets() for t in validation_tasks]
        logger.info(f"Validation dataset loaded with {len(validation_chats)} tasks.")

        # Create datasets.Dataset objects
        train_dataset = datasets.Dataset.from_list(training_chats)
        logger.info(f"Training dataset created with {len(train_dataset)} examples.")
        validation_dataset = datasets.Dataset.from_list(validation_chats)
        logger.info(f"Validation dataset created with {len(validation_dataset)} examples.")
        
        # Remove the tasks and prompts from memory
        del train_tasks, validation_tasks, training_chats, validation_chats

        # Create DatasetDict
        dataset = datasets.DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset
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
            model.tokenize_chat,
            batched=True,
            remove_columns=["instruction", "input", "output"],
            desc="Processing training dataset",
        )
        tokenized_eval_dataset = dataset['validation'].map(
            model.tokenize_chat,
            batched=True,
            remove_columns=["instruction", "input", "output"],
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

            def save_dataset_samples(dataset, model, checkpoint_dir, dataset_name, num_samples=5):
                if len(dataset) == 0:
                    logger.warning(f"Cannot save samples from empty {dataset_name} dataset.")
                    return
                
                actual_num_samples = min(num_samples, len(dataset))
                logger.info(f"Selecting {actual_num_samples} random samples from {dataset_name} dataset.")
                # Ensure reproducibility if dataset is shuffled
                random_samples_dataset = dataset.shuffle(seed=42).select(range(actual_num_samples))
                
                samples_to_save = []
                for i in range(actual_num_samples):
                    sample = random_samples_dataset[i]
                    # Create a new dictionary to avoid modifying the original dataset structure directly
                    # and to ensure all data is serializable.
                    processed_sample = {
                        'input_ids': sample['input_ids'],
                        'labels': sample['labels'],
                        'attention_mask': sample.get('attention_mask', []), # Include if present
                        'decoded_input': model.decode(sample['input_ids'], skip_special_tokens=False),
                        'decoded_labels': model.decode(sample['labels'], skip_special_tokens=False),
                    }
                    samples_to_save.append(processed_sample)

                sample_file_path = os.path.join(checkpoint_dir, f"sample_{dataset_name}_data.jsonl")
                try:
                    with open(sample_file_path, "w") as f:
                        for sample_dict in samples_to_save:
                            import json # Ensure json is imported
                            f.write(json.dumps(sample_dict) + "\n")
                    logger.info(f"Sample {dataset_name} data saved to {sample_file_path}")
                except Exception as e:
                    logger.error(f"Error saving sample {dataset_name} data to {sample_file_path}: {e}", exc_info=True)

            # Save samples for training dataset
            save_dataset_samples(tokenized_train_dataset, model, model_checkpoint_dir, "training")
            # Save samples for validation dataset
            save_dataset_samples(tokenized_eval_dataset, model, model_checkpoint_dir, "validation")



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