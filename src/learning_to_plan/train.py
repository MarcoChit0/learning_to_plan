import os
import datetime
import datasets
from learning_to_plan.models import base
import learning_to_plan.config as config
logger = config.get_logger(__name__)
from learning_to_plan.data import task
from learning_to_plan import prompt_building
from learning_to_plan.models import utils

def get_tokenized_dataset(model: base.Model, tasks:set[task.Task], max_seq_length:int=1024, **kwargs):
    """
    Create a dataset from the tasks and model.
    """
    data: dict = {
        'input_ids': [],
        'labels': [],
        'attention_mask': [],
    }
    for t in tasks:
        chat = prompt_building.get_chat(t, with_plan=True, **kwargs)
        tokenized_chat = model.tokenize_chat(chat, max_seq_length=max_seq_length)
        data['input_ids'].append(tokenized_chat['input_ids'])
        data['labels'].append(tokenized_chat['labels'])
        data['attention_mask'].append(tokenized_chat['attention_mask'])

    # Convert to Dataset
    dataset = datasets.Dataset.from_dict(data)
    # Set the format for the dataset
    dataset.set_format(type='torch', columns=['input_ids', 'labels', 'attention_mask'])
    return dataset
    
def save_dataset_samples(dataset:datasets.Dataset, model:base.Model, checkpoint_dir:str, dataset_name:str, num_samples:int=5):
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
            'input_ids': sample['input_ids'].tolist(), # Convert to list for JSON serialization
            'labels': sample['labels'].tolist(), # Convert to list for JSON serialization
            'attention_mask': sample['attention_mask'].tolist(), # Convert to list for JSON serialization
            'decoded_input': model.decode(sample['input_ids'], skip_special_tokens=False),
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


def run_training_procedure(model_name: str, domain: str, **train_kwargs):
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting training at {start_time_str}")

    model_checkpoint_dir = config.get_checkpoint_dir(domain, model_name)
    config.create_necessary_dirs(model_checkpoint_dir)
    logger.info(f"Checkpoints will be saved to: {model_checkpoint_dir}")
    try:
        model = utils.get_model(model_name=model_name)
        train_kwargs['is_trainable'] = True
        train_kwargs['checkpoint_dir'] = model_checkpoint_dir
        model.setup(**train_kwargs)
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
        raise e

    # --- Load and Prepare Dataset ---
    try:
        logger.info(f"Loading training and validation datasets for domain: {domain}.")
        train_tasks:set[task.Task] = task.task_database.get(filter_by_domain=domain, filter_by_pourpose=task.Task.POURPOSE.TRAIN)
        validation_tasks:set[task.Task] = task.task_database.get(filter_by_domain=domain, filter_by_pourpose=task.Task.POURPOSE.VALIDATION)

        logger.info(f"Tokenizing datasets for training and validation.")
        tokenized_train_dataset = get_tokenized_dataset(model, train_tasks, **train_kwargs)
        tokenized_eval_dataset = get_tokenized_dataset(model, validation_tasks, **train_kwargs)

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

            # --- Save samples from training and validation datasets ---
            save_dataset_samples(tokenized_train_dataset, model, model_checkpoint_dir, "training")
            save_dataset_samples(tokenized_eval_dataset, model, model_checkpoint_dir, "validation")

        else: 
            raise ValueError("Tokenized training dataset is empty despite non-empty input. Check processing logic and data.")
    except Exception as e:
        logger.error(f"Error during dataset processing: {e}", exc_info=True)
        raise e


    # --- Train ---
    try:
        logger.info("Calling model.train() at %s", datetime.datetime.now())
        model.train(tokenized_train_dataset=tokenized_train_dataset, tokenized_eval_dataset=tokenized_eval_dataset, **train_kwargs)
        end_time = datetime.datetime.now()
        logger.info(f"Training completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        logger.error(f"Error when training model {model_name}: {e}", exc_info=True)
        raise e