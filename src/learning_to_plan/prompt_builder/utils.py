from learning_to_plan.prompt_builder import base
from learning_to_plan.prompt_builder.natural_language import io, few_shot
from learning_to_plan import config
from learning_to_plan.prompt_builder.pddl import pddl, landmarks

def get_prompt_builder(prompt_type: config.PROMPT_TYPE, **kwargs) -> base.PromptBuilder:
    mapping = {
        config.PROMPT_TYPE.IO: io.IOPromptBuilder,
        config.PROMPT_TYPE.FEW_SHOT: few_shot.FewShotPromptBuilder,
        config.PROMPT_TYPE.PDDL: pddl.PDDLPromptBuilder,
        config.PROMPT_TYPE.LANDMARKS: landmarks.LandmarksPromptBuilder
    }
    if prompt_type in mapping:
        return mapping[prompt_type](**kwargs)
    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}. Supported types are: {list(mapping.keys())}.")