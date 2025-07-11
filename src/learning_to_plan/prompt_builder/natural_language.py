from learning_to_plan.prompt_builder import base
from learning_to_plan import config
from learning_to_plan.domain_translators import utils as domain_translator_utils

logger = config.get_logger(__name__)

class NaturalLanguagePromptBuilder(base.PromptBuilder):
    def __init__(self, prompt_type, **kwargs):
        super().__init__(prompt_type=prompt_type, **kwargs)

    def process_response(self, response: str) -> str:
        if response == "":
            raise ValueError("Empty response from model.")
        if config.TOKENS.PLAN_START.value not in response:
            raise ValueError(f"Plan start token '{config.TOKENS.PLAN_START.value}' not found in response.")
        if config.TOKENS.PLAN_END.value not in response:
            raise ValueError(f"Plan end token '{config.TOKENS.PLAN_END.value}' not found in response.")
        plan_start_idx = response.index(config.TOKENS.PLAN_START.value)
        plan_end_idx = response.index(config.TOKENS.PLAN_END.value)
        if plan_start_idx >= plan_end_idx:
            raise ValueError("Plan start token is after the end token in the response.")
        raw_plan = response[plan_start_idx + len(config.TOKENS.PLAN_START.value):plan_end_idx].strip()
        if raw_plan.replace(" ", "").replace("\n", "") == "":
            raise ValueError("Generated plan is empty after removing whitespace and start/end tokens.")
        try:
            pddl_plan = domain_translator_utils.translate_natural_language_plan_to_pddl(
                t=None,  # Task is not provided here, but can be passed if needed
                raw_plan=raw_plan
            )
        except Exception as e:
            raise ValueError("Error translating plan to PDDL: " + str(e))
        return pddl_plan
