from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os
from learning_to_plan import config
from learning_to_plan.data import metadata
import datasets
logger = config.get_logger(__name__)

class Model:
    DEFAULT_GENERATION_CONFIG : dict[str, any] = {}

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metadata : dict[str, Any] = {
            "model_name": model_name,
        }
        self.model_dir_path = os.path.join(config.MODELS_DIR, model_name)
        os.makedirs(self.model_dir_path, exist_ok=True)
        logger.info(f"Initialized model {self.model_name} with directory {self.model_dir_path}.")
        logger.warning(f"If you want to generate content or train the model {self.model_name}, please call the setup method first.")
    
    def setup(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        return

    def generate(self, chat:list[dict[str, str]], **generation_kwargs) -> Tuple[str, dict[str, Any]]:
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
    
    @classmethod
    def get_generation_config(cls, **gen_kwargs) -> Dict[str, Any]:
        """
        Returns the generation configuration for the model.
        This is a placeholder method and should be implemented in subclasses.
        """
        _gen_config = cls.DEFAULT_GENERATION_CONFIG.copy()
        for key in _gen_config:
            if key in gen_kwargs:
                _gen_config[key] = gen_kwargs[key]
        return _gen_config
    
    def get_metadata(self, **gen_kwargs) -> metadata.Metadata:
        """
        Returns the metadata of the model. This method should only be used when generating plans. There is no need to call this method when training the model, because it will create a lot of useless metadata entries.
        """
        return metadata.create_metadata(
            **self.metadata,
            **self.get_generation_config(**gen_kwargs)
        )
    
    