import abc
from learning_to_plan.data import task
from learning_to_plan import config
from learning_to_plan.data import metadata

class PromptBuilder(abc.ABC):
    def __init__(self, prompt_type: config.PROMPT_TYPE, **kwargs):
        self.prompt_type : config.PROMPT_TYPE = prompt_type
        self.metadata = {
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

    @abc.abstractmethod
    def process_response(self, response: str) -> str:
        """
        Processes the response from the model.
        :param response: The response from the model.
        :param task: The task for which the response is being processed.
        :return: The processed response.
        """
        raise NotImplementedError("Subclasses must implement the process_response method.")
    
    def get_metadata(self, **gen_kwargs) -> dict[str, any]:
        """
        Returns the metadata for the prompt builder.
        :return: A dictionary containing the metadata.
        """
        return metadata.create_metadata(
            class_name=self.__class__.__name__,
            **self.metadata,
        )