import abc
from learning_to_plan.data import task
from learning_to_plan import config
import os
from learning_to_plan.domain_translators import utils

def get_prompt_metadata(prompt_type: config.PROMPT_TYPE = config.PROMPT_TYPE.IO, **kwargs) -> dict[str, any]:
    if prompt_type == config.PROMPT_TYPE.IO:
        return {}
    elif prompt_type == config.PROMPT_TYPE.FEW_SHOT:
        few_shot = kwargs.get("few_shot", 1)
        return {"few_shot": few_shot}
    elif prompt_type == config.PROMPT_TYPE.PDDL:
        return {}
    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}. Supported types are: {list(config.PROMPT_TYPE)}.")

class PromptBuilder(abc.ABC):
    def __init__(self, prompt_type: config.PROMPT_TYPE, **kwargs):
        self.prompt_type = prompt_type
        self.__dict__.update(kwargs)
        self.prompt_metadata = {
            "prompt_type": self.prompt_type.value,
        }
    
    @abc.abstractmethod
    def get_chat(self, t: task.Task, with_plan: bool = True, **kwargs) -> list[dict[str, str]]:
        """
        Builds the chat prompt for the given task.
        :param t: The task for which to build the prompt.
        :param with_plan: Whether to include the plan in the prompt.
        :return: A list of dictionaries representing the chat prompt.
        """
        raise NotImplementedError("Subclasses must implement the get_chat method.")