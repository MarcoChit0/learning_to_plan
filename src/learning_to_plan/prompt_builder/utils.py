from learning_to_plan.prompt_builder import base, few_shot, io, pddl
from learning_to_plan import config
from learning_to_plan.data import task

def get_prompt_builder(prompt_type: config.PROMPT_TYPE, **kwargs) -> base.PromptBuilder:
    if prompt_type == config.PROMPT_TYPE.IO:
        return io.IOPromptBuilder(**kwargs)
    elif prompt_type == config.PROMPT_TYPE.FEW_SHOT:
        return few_shot.FewShotPromptBuilder(**kwargs)
    elif prompt_type == config.PROMPT_TYPE.PDDL:
        return pddl.PDDLPromptBuilder(**kwargs)
    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}. Supported types are: {list(config.PROMPT_TYPE)}.")