from __future__ import annotations
from typing import Optional, List, Dict, Any
import os
from learning_to_plan import config, task
import datasets
import json
logger = config.get_logger(__name__)

class Model:
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._generated_plans : set[Model.Content] = set()
        self._model_dir_path = os.path.join(config.MODELS_DIR, model_name)
        os.makedirs(self._model_dir_path, exist_ok=True)
        self._metadata = {
            "model_name": model_name,
        }
        self.load_generated_plans()
        logger.info(f"Initialized model {self._model_name} with directory {self._model_dir_path}.")
        logger.info(f"Loaded {len(self._generated_plans)} generated plans from {config.GENERATED_PLANS_FILE_NAME}.")
        logger.warning(f"If you want to generate content or train the model {self._model_name}, please call the setup method first.")

    def clear_model_dir(self) -> None:
        if os.path.exists(self._model_dir_path):
            logger.info(f"Deleting existing model directory: {self._model_dir_path}")
            os.rmdir(self._model_dir_path)
        os.makedirs(self._model_dir_path, exist_ok=True)
        self._generated_plans.clear()
        logger.info(f"Model directory {self._model_dir_path} has been reset.")
    
    def setup(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        return

    def add_generated_plan(self, content:Content) -> None:
        self._generated_plans.add(content)
        logger.info(f"Added new plan with ID {content._id} for task {content._task} and prompt type {content._prompt_type} to model {self._model_name}.")

    def get_generated_plans(self, t: Optional[task.Task]=None, prompt_type: Optional[config.PROMPT_TYPE]=None, model_metadata:Optional[dict[str, any]] = None, prompt_metadata:Optional[dict[str, any]] = None) -> set[Content]:
        plans = set()
        for content in self._generated_plans:
            if t is not None and content._task != t:
                continue
            if prompt_type is not None and content._prompt_type != prompt_type:
                continue
            if model_metadata is not None and content._model_metadata != model_metadata:
                continue
            if prompt_metadata is not None and content._prompt_metadata != prompt_metadata:
                continue
            else:
                plans.add(content)
        return plans

    def overwrite_generated_plans(self, t: task.Task, prompt_type: config.PROMPT_TYPE, model_metadata:Optional[dict[str, any]] = None, prompt_metadata:Optional[dict[str, any]] = None) -> None:
        counter = 0
        for content in self._generated_plans:
            if content._task == t and content._prompt_type == prompt_type:
                if model_metadata is not None and content._model_metadata != model_metadata:
                    continue
                if prompt_metadata is not None and content._prompt_metadata != prompt_metadata:
                    continue
                Model.Content.remove_id(content._id)
                counter += 1
        logger.info(f"Overwrote {counter} plans for task {t} and prompt type {prompt_type} in model {self._model_name}.")

    def generate_single_sample(self, chat:list[dict[str, str]], **generation_kwargs) -> str:
        """
        Generates a plan based on the provided prompt.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def train(self, train_dataset: datasets.DatasetDict, eval_dataset: datasets.DatasetDict,  **train_kwargs) -> None: # Changed type hint
        """
        Trains the model on the provided data.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def tokenize_chat(self, examples, max_seq_length: int = 1024) -> Dict[str, Any]:
        """
        Tokenizes the input examples for chat-based models.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def decode(self, input_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decodes the input IDs to a string.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns metadata about the model.
        This is a placeholder method and should be implemented in subclasses.
        """
        return self._metadata
    
    def save_generated_plans(self) -> None:
        """
        Saves generated plans to a JSONL file in the model directory.
        Each line is one JSON object corresponding to a Model.Content.
        """
        file_path = os.path.join(self._model_dir_path, config.GENERATED_PLANS_FILE_NAME)
        logger.info(f"Saving generated plans to {file_path}.")
        try:
            with open(file_path, "w", encoding="utf-8") as fout:
                for content in sorted(self._generated_plans):
                    row = content.write_to_json_row()
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info(f"Successfully saved {len(self._generated_plans)} plans to {file_path}.")
        except Exception as e:
            logger.error(f"Error saving plans to {file_path}: {e}", exc_info=True)
            raise

    def load_generated_plans(self) -> None:
        """
        Loads generated plans from a JSONL file in the model directory.
        """
        file_path = os.path.join(self._model_dir_path, config.GENERATED_PLANS_FILE_NAME)
        logger.info(f"Loading generated plans from {file_path}.")
        if not os.path.exists(file_path):
            logger.warning(f"No file at {file_path}, skipping load.")
            return

        loaded_count = 0
        try:
            with open(file_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        content = Model.Content.read_from_json_row(obj)
                        self._generated_plans.add(content)
                        loaded_count += 1
                    except Exception as e:
                        logger.error(f"Failed to parse line as Content JSON: {line}\n{e}", exc_info=True)
            logger.info(f"Successfully loaded {loaded_count} plans from {file_path}.")
        except Exception as e:
            logger.error(f"Error reading plans from {file_path}: {e}", exc_info=True)
            raise