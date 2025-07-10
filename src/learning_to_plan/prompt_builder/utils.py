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

def get_chat(t: task.Task, with_plan: bool = True, prompt_type: config.PROMPT_TYPE = config.PROMPT_TYPE.IO, **kwargs) -> list[dict[str, str]]:
    """
    Builds the chat prompt for the given task.
    :param t: The task for which to build the prompt.
    :param with_plan: Whether to include the plan in the prompt.
    :param prompt_type: The type of prompt to build.
    :return: A list of dictionaries representing the chat prompt.
    """
    prompt_builder = get_prompt_builder(prompt_type, **kwargs)
    return prompt_builder.get_chat(t, with_plan=with_plan, **kwargs)

def get_metadata(prompt_type: config.PROMPT_TYPE = config.PROMPT_TYPE.IO, **kwargs) -> dict[str, any]:
    """
    Returns metadata for the prompt.
    :param prompt_type: The type of prompt.
    :return: A dictionary containing metadata for the prompt.
    """
    prompt_builder = get_prompt_builder(prompt_type, **kwargs)
    return prompt_builder.prompt_metadata