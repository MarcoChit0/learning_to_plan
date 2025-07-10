from learning_to_plan.domain_translators.base import DomainTranslator
from learning_to_plan.data import task

def get_domain_traslator(domain: str) ->DomainTranslator:
    """
    Returns the appropriate domain translator based on the given domain name.
    
    Args:
        domain (str): The name of the domain for which to get the translator.
    
    Returns:
        DomainTranslator: An instance of the appropriate domain translator.
    
    Raises:
        ValueError: If the domain is not recognized.
    """
    if domain == "blocksworld":
        from learning_to_plan.domain_translators.blocksworld import BlocksworldTranslator
        return BlocksworldTranslator()
    else:
        raise ValueError(f"Domain '{domain}' is not recognized or supported.")

def get_task_components_in_natural_language(t : task.Task, with_plan:bool = True) -> dict[str, str]:
    domain_translator = get_domain_traslator(domain=t._domain)
    domain_description = domain_translator._domain_description_in_natural_language.strip()
    pddl_instance = t.read_instance()
    initial_state_facts_nl = domain_translator.get_initial_state_facts_from_pddl_instance(pddl_instance=pddl_instance)
    goal_facts_nl = domain_translator.get_goal_facts_from_pddl_instance(pddl_instance=pddl_instance)
    if t._pddl_plan:
        plan_nl = domain_translator.translate_pddl_plan_to_natural_language(pddl_plan=t._pddl_plan)
    else:
        if with_plan:
            raise ValueError("PDDL plan is not available, but 'with_plan' is set to True. Please ensure the task has a PDDL plan before requesting it.")
        else:
            plan_nl = ""
    return {
        "domain_description": domain_description,
        "initial_state_facts": initial_state_facts_nl,
        "goal_facts": goal_facts_nl,
        "plan": plan_nl
    }

def translate_natural_language_plan_to_pddl(t: task.Task, nl_plan: str) -> str:
    """
    Translates a natural language plan to PDDL format using the domain translator.
    
    Args:
        t (task.Task): The task containing the domain translator.
        nl_plan (str): The natural language plan to translate.
    
    Returns:
        str: The translated PDDL plan.
    
    Raises:
        ValueError: If the translation fails.
    """
    domain_translator = get_domain_traslator(domain=t._domain)
    try:
        pddl_plan = domain_translator.translate_natural_language_plan_to_pddl(nl_plan=nl_plan)
    except Exception as e:
        raise ValueError(f"Failed to translate natural language plan to PDDL: {e}")
    return pddl_plan.strip()