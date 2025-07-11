from learning_to_plan.prompt_builder import natural_language
from learning_to_plan import config
from learning_to_plan.data import task
from learning_to_plan.domain_translators import utils as domain_translator_utils

logger = config.get_logger(__name__)

class IOPromptBuilder(natural_language.NaturalLanguagePromptBuilder):
    def __init__(self, **kwargs):
        super().__init__(prompt_type=config.PROMPT_TYPE.IO, **kwargs)

    def get_chat(self, t : task.Task, with_plan: bool = True, **kwargs) -> list[dict[str, str]]:
        """
        Returns a chat object for the given task.
        """
        task_components_in_nl = domain_translator_utils.get_task_components_in_natural_language(t=t, with_plan=with_plan)

        domain = task_components_in_nl['domain_description']
        initial_state = "As initial conditions I have that: " + (", ".join(task_components_in_nl['initial_state_facts']))
        goal_facts = "My goal is to have that: " + (", ".join(task_components_in_nl['goal_facts']))

        chat: list[dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{domain}\n{initial_state}\n{goal_facts}"},
        ]   

        if with_plan:
            plan = f"My plan is as follows:\n{config.TOKENS.PLAN_START.value}\n{task_components_in_nl['plan']}\n{config.TOKENS.PLAN_END.value}"
            chat.append({"role": "assistant", "content": plan})
        
        return chat
