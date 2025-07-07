from __future__ import annotations
from typing import Optional, List, Dict, Any
import os
from learning_to_plan import config, task
import datasets
import json
from learning_to_plan import generated_plans
logger = config.get_logger(__name__)

class Model:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_dir_path = os.path.join(config.MODELS_DIR, model_name)
        os.makedirs(self.model_dir_path, exist_ok=True)
        self.metadata = {
            "model_name": model_name,
        }
        logger.info(f"Initialized model {self.model_name} with directory {self.model_dir_path}.")
        logger.warning(f"If you want to generate content or train the model {self.model_name}, please call the setup method first.")
    
    def setup(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        return

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
        return self.metadata