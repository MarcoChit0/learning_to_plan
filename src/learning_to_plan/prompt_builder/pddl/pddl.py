from __future__ import annotations
from string import Template
from typing import Union
from learning_to_plan import config
from learning_to_plan.prompt_builder import base
from learning_to_plan.data import task
from learning_to_plan import database

BASIC_PDDL_EXAMPLE_TEMPLATE = Template(f"""{config.TOKENS.EXAMPLE_START.value}
{config.TOKENS.DOMAIN_START.value}
$domain
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.PROBLEM_START.value}
$instance
{config.TOKENS.PROBLEM_END.value}
{config.TOKENS.PLAN_START.value}
$plan
{config.TOKENS.PLAN_END.value}
{config.TOKENS.EXAMPLE_END.value}
""")

THINKING_PDDL_EXAMPLE_TEMPLATE = Template(f"""{config.TOKENS.EXAMPLE_START.value}
{config.TOKENS.DOMAIN_START.value}
$domain
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.PROBLEM_START.value}
$instance
{config.TOKENS.PROBLEM_END.value}
{config.TOKENS.THINKING_START.value}
$thinking
{config.TOKENS.THINKING_END.value}
{config.TOKENS.PLAN_START.value}
$plan
{config.TOKENS.PLAN_END.value}
{config.TOKENS.EXAMPLE_END.value}
""")

BASIC_PDDL_PROMPT_TEMPLATE = Template(f"""Your task is to generate a plan for the following planning problem. The domain section describes the available actions and objects, and the problem section defines the initial and goal states. The plan must be a sequence of actions that starts from the initial state and reaches the goal state. The entire plan must be enclosed between {config.TOKENS.PLAN_START.value} and {config.TOKENS.PLAN_END.value} tags, and each action must be on a new line. Your response should only contain the plan.
                                
{config.TOKENS.DOMAIN_START.value}
$domain
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.PROBLEM_START.value}
$instance
{config.TOKENS.PROBLEM_END.value}

Here are some examples of plans in the same format as the one you should provide:

$examples

Here is a checklist to help you with your task:

{config.TOKENS.CHECKLIST_START.value}
1) Do not provide any additional text or explanations outside the plan tags.
2) The plan must be in the same format as the examples above.
3) The plan should be preceded by the {config.TOKENS.PLAN_START.value} tag and should be followed by the {config.TOKENS.PLAN_END.value} tag.
4) The actions in the plan must be from the set of actions in the domain described above, that is, they must use the same name and the same number of parameters as one of the action schemas.
5) The plan must be valid, that is, each action must be applicable in the state it is applied, and the plan must end in a goal state.
{config.TOKENS.CHECKLIST_END.value}""")

THINKING_PDDL_PROMPT_TEMPLATE= Template(f"""Your task is to generate a plan for the following planning problem. The domain section describes the available actions and objects, and the problem section defines the initial and goal states. The plan must be a sequence of actions that starts from the initial state and reaches the goal state. The entire plan must be enclosed between {config.TOKENS.PLAN_START.value} and {config.TOKENS.PLAN_END.value} tags, and each action must be on a new line. You must provide the plan in your response. You may also provide your reasoning in the thinking section before the plan, which should be enclosed between {config.TOKENS.THINKING_START.value} and {config.TOKENS.THINKING_END.value} tags. Your response should only contain the reasoning and the plan.

{config.TOKENS.DOMAIN_START.value}
$domain
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.PROBLEM_START.value}
$instance
{config.TOKENS.PROBLEM_END.value}

Here are some examples of plans in the same format as the one you should provide:

$examples

Here is a checklist to help you with your task:

{config.TOKENS.CHECKLIST_START.value}
1) Your response could contain the reasoning inside the thinking section and must contain the plan inside the plan section. Do not provide any additional text or explanations outside these sections.
2) The plan must be in the same format as the examples above.
3) The plan should be preceded by the {config.TOKENS.PLAN_START.value} tag and should be followed by the {config.TOKENS.PLAN_END.value} tag.
4) The actions in the plan must be from the set of actions in the domain described above, that is, they must use the same name and the same number of parameters as one of the action schemas.
5) The plan must be valid, that is, each action must be applicable in the state it is applied, and the plan must end in a goal state.
6) You may provide your reasoning in the thinking section before the plan, which should be enclosed between {config.TOKENS.THINKING_START.value} and {config.TOKENS.THINKING_END.value} tags.
7) Your reasoning should be restricted to the thinking section. So, do not provide any reasoning inside the plan section.
{config.TOKENS.CHECKLIST_END.value}""")

logger = config.get_logger(__name__)

class ThinkingStep:
    def build(self) -> str:
        """
        Returns the action associated with this thinking step.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
class StateStep(ThinkingStep):
    def __init__(self, state_facts: set[str]):
        self.state_facts = state_facts
        self.state_fact_additional_info = {f : [] for f in state_facts}
    
    def build(self) -> str:
        """
        Builds the state step as a string.
        :return: The state step as a string.
        """
        parts = [f"{config.TOKENS.STATE_START.value}"]
        for fact in self.state_facts:
            l = fact.strip()
            if self.state_fact_additional_info[fact]:
                l += "\t; " + ", ".join(self.state_fact_additional_info[fact])
            parts.append(l)
        parts.append(f"{config.TOKENS.STATE_END.value}")
        return "\n".join(parts)

    def __str__(self):
        return f"StateStep(state_facts={self.state_facts})"

class ActionStep(ThinkingStep):
    def __init__(self, action: str, add_effects: list[str] = [], delete_effects: list[str] = []):
        self.action = action
        self.add_effects = add_effects
        self.delete_effects = delete_effects
    
    def build(self) -> str:
        """
        Builds the action step as a string.
        :return: The action step as a string.
        """
        parts = [f"Action: {self.action}"]
        if self.delete_effects:
            parts.append(f"{config.TOKENS.DELETE_EFFECTS_START.value}\n" + "\n".join(self.delete_effects) + f"\n{config.TOKENS.DELETE_EFFECTS_END.value}")
        if self.add_effects:
            parts.append(f"{config.TOKENS.ADD_EFFECTS_START.value}\n" + "\n".join(self.add_effects) + f"\n{config.TOKENS.ADD_EFFECTS_END.value}")
        return "\n".join(parts)
    
    def __str__(self):
        return f"ActionStep(action={self.action}, add_effects={self.add_effects}, delete_effects={self.delete_effects})"

class Thinking:
    def __init__(self, steps: list[ThinkingStep] = []):
        self.steps = steps
    
    def add(self, step: ThinkingStep):
        """
        Adds a step to the thinking process.
        :param step: The step to add.
        """
        if len(self.steps) == 0:
            if not isinstance(step, StateStep):
                raise ValueError("The first step must be a StateStep. Please add a StateStep first.")
            self.steps.append(step)
            
        else:
            last_step = self.steps[-1]
            if isinstance(last_step, StateStep) and isinstance(step, StateStep):
                raise ValueError("Cannot add two StateSteps in a row. Please add an ActionStep in between.")
            if isinstance(last_step, ActionStep) and isinstance(step, ActionStep):
                raise ValueError("Cannot add two ActionSteps in a row. Please add a StateStep in between.")
            
            if isinstance(step, StateStep):
                self.steps.append(step)
            
            elif isinstance(step, ActionStep):
                assert isinstance(last_step, StateStep), "An ActionStep must be preceded by a StateStep. Please add a StateStep first."
                state_facts = last_step.state_facts
                state_facts = state_facts - set(step.delete_effects)
                state_facts = state_facts.union(set(step.add_effects))
                new_state = StateStep(state_facts=state_facts)
                self.steps.append(step)
                self.steps.append(new_state)
            
            else:
                raise ValueError(f"Unknown step type: {type(step)}. Please provide a StateStep or ActionStep.")

    def build(self) -> str:
        """
        Builds the thinking process as a string.
        :return: The thinking process as a string.
        """
        return "\n\n".join(step.build() for step in self.steps)
    
    def add_flag_to_state_facts(self, flag: str, state_facts: set[str]):
        for step in self.steps:
            if isinstance(step, StateStep):
                for fact in state_facts:
                    if fact in step.state_facts:
                        step.state_fact_additional_info[fact].append(flag)
    
    @classmethod
    def from_thinking_style(cls, t : task.Task, thinking_style: config.THINKING_STYLE) -> Thinking:
        if thinking_style == config.THINKING_STYLE.NONE:
            return Thinking()

        elif thinking_style == config.THINKING_STYLE.COT:
            try:
                plan_data = extract_plan_add_and_delete_data(t)
            except Exception as e:
                logger.error(f"Error getting plan effects for task {t.id}: {e}")

            initial_state_regex = r"\({}\s+(.*?)(?=\s+\(:|\s*\)\s*$)".format(re.escape(pattern=":init"))
            try:
                match = re.search(initial_state_regex, t.read_instance(), re.IGNORECASE | re.DOTALL)
                if match:
                    initial_state = match.group(1).strip()
                else:
                    raise ValueError(f"Could not find initial state in task {t.id} instance.")
            except Exception as e:
                logger.error(f"Error extracting initial state from task {t.id}: {e}")
                raise e
            state_facts = set()
            for fact in initial_state.splitlines():
                if fact.replace("(", "").replace(")", "").strip() == "":
                    continue
                state_facts.add(fact.strip().replace("\t", ""))
            
        
            thinking : Thinking = Thinking([StateStep(state_facts=state_facts)])

            for action_index, action_data in plan_data.items():
                action = action_data["name"]
                effects = action_data["effects"]
                thinking.add(ActionStep(
                    action=action,
                    add_effects=effects["add"],
                    delete_effects=effects["delete"]
                ))
            return thinking

        else :
            raise ValueError(f"Unknown thinking style: {self.thinking_style}. Please provide a valid thinking style.")

class PDDLPromptBuilder(base.PromptBuilder):

    def __init__(self, prompt_type=config.PROMPT_TYPE.PDDL, **kwargs):
        super().__init__(prompt_type=prompt_type, **kwargs)
        self.examples : list[task.Task] = []
        self.examples_data : dict[task.Task, dict[str, Union[str, Thinking]]] = {}
        for d in ["hanoi", "storage"]:
            tasks : set[task.Task] = database.task_database.get(filter_by_domain=d)
            assert len(tasks) > 0, f"No tasks found for the {d} domain."
            t = sorted(tasks)[-1]
            self.examples.append(t)
            try:
                self.examples_data[t] = self.get_task_data(t, with_plan=True)
            except Exception as e:
                logger.error(f"Error getting example data for task {t.id}: {e}")
                raise e
            try:
                self.examples_data[t]["thinking"] = Thinking.from_thinking_style(t, self.thinking_style)
            except Exception as e:
                logger.error(f"Error getting thinking for task {t.id}: {e}")
                raise e

    def get_task_data(self, t: task.Task, with_plan: bool) -> dict[str, str]:
        """
        Returns the data for the task.
        :param t: The task to get the data for.
        :return: A dictionary with the task data.
        """
        try:
            domain = t.read_domain().strip()
            instance = t.read_instance().strip()
            plan = t.get_plan() if with_plan else ""
        except Exception as e:
            logger.error(f"Error getting task data for task {t.id}: {e}")
            raise e
        return {
            "domain": domain,
            "instance": instance,
            "plan": plan,
        }

    def get_examples(self, **kwargs) -> str:
        template = self.get_example_template(**kwargs)
        examples = []
        for ex in self.examples:
            ex_data = self.examples_data[ex]
            if "thinking" in ex_data:
                thinking = ex_data["thinking"].build()
                _d = ex_data.copy()
                _d["thinking"] = thinking
            examples.append(template.substitute(
                **_d
            ))
        return "\n".join(examples)

    def get_chat(self, t : task.Task, with_plan: bool = True, **kwargs) -> list[dict[str, str]]:
        prompt = self.get_prompt_template(t, **kwargs)
        try:
            data = self.get_task_data(t, with_plan=with_plan)
        except Exception as e:
            logger.error(f"Error getting task data for task {t.id}: {e}")
            raise e
        try:
            data["examples"] = self.get_examples(**kwargs).strip()
        except Exception as e:
            logger.error(f"Error getting examples for task {t.id}: {e}")
            raise e

        content = prompt.substitute(**data)
        
        chat = [
            {"role": "system", "content": "You are an expert in AI Planning."},
            {"role": "user", "content": content}
        ]
        if with_plan:
            chat.append({"role": "assistant", "content": f"{config.TOKENS.PLAN_START.value}\n{t.get_plan()}\n{config.TOKENS.PLAN_END.value}"})
        
        with open("debug_pddl.txt", "w") as f:
            f.write(content)

        return chat
    
    def process_response(self, response: str) -> str:
        """
        Processes the response from the model.
        :param response: The response from the model.
        :return: The processed response.
        """
        print(response)
        if response == "":
            raise ValueError("The response is empty. Please provide a valid plan.")
        
        if config.TOKENS.PLAN_START.value not in response:
            raise ValueError(f"The response does not contain the plan start token '{config.TOKENS.PLAN_START.value}'. Please provide a valid plan.")
        
        if config.TOKENS.PLAN_END.value not in response:
            raise ValueError(f"The response does not contain the plan end token '{config.TOKENS.PLAN_END.value}'. Please provide a valid plan.")
        
        plan_start_index = response.index(config.TOKENS.PLAN_START.value)
        plan_end_index = response.index(config.TOKENS.PLAN_END.value)

        if plan_start_index > plan_end_index:
            raise ValueError("The plan start token is after the plan end token. Please provide a valid plan.")
        
        plan = response[plan_start_index + len(config.TOKENS.PLAN_START.value):plan_end_index].strip()

        if plan == "":
            raise ValueError("After removing the plan start and end tokens, the plan is empty. Please provide a valid plan.")
        
        return plan
        
    def get_example_template(self, **kwargs) -> Template:
        """
        Returns the template to be used for generating the prompt.
        :param kwargs: Additional keyword arguments.
        :return: The template to be used for generating the prompt.
        """
        if self.thinking_style == config.THINKING_STYLE.NONE:
            return BASIC_PDDL_EXAMPLE_TEMPLATE
        elif self.thinking_style == config.THINKING_STYLE.COT:
            return THINKING_PDDL_EXAMPLE_TEMPLATE
        else:
            raise ValueError(f"Unknown thinking style: {self.thinking_style}. Please provide a valid thinking style.")

    def get_prompt_template(self, t: task.Task, **kwargs) -> Template:
        """
        Returns the prompt template to be used for generating the prompt.
        :param kwargs: Additional keyword arguments.
        :return: The prompt template to be used for generating the prompt.
        """
        if self.thinking_style == config.THINKING_STYLE.NONE:
            return BASIC_PDDL_PROMPT_TEMPLATE
        elif self.thinking_style == config.THINKING_STYLE.COT:
            return THINKING_PDDL_PROMPT_TEMPLATE
        else:
            raise ValueError(f"Unknown thinking style: {self.thinking_style}. Please provide a valid thinking style.")

from learning_to_plan import utils 
import re
def extract_plan_add_and_delete_data(t: task.Task) -> str:
    """
    Gets the add and delete effects for each action in a plan for a given task.
    :param t: The task to execute.
    :return: A dictionary mapping each action in the plan to its add and delete effects.
    """
    try:
        plan = t.get_plan()
        if not plan:
            raise ValueError(f"The task {t.id} has no plan. Please provide a valid plan.")
        result = utils.call_val(t, plan)
        if result is None:
            raise ValueError("The result of the execution trace is None. Please provide a valid plan.")
        
        is_plan_valid = False
        for line in result.splitlines():
                if "Plan valid" in line:
                    is_plan_valid = True
                    break
        if not is_plan_valid:
            raise ValueError("The plan is not valid. Please provide a valid plan.")
    except Exception as e:
        logger.error(f"Error executing plan for task {t.id}: {e}")
        raise e
    
    plan_actions = plan.strip().splitlines()
    map_action_to_effects = {i : {
        "effects" : {
            "delete" : [],
            "add" : [],            
        },
        "name" : plan_actions[i]
    } for i in range(len(plan_actions))}

    start_trace_pattern = r"Checking next happening \(time (\d+)\)"
    delete_fact_pattern = r"Deleting (\(.*\))"
    add_fact_pattern = r"Adding (\(.*\))"
    lines = result.splitlines()
    with open(f"debug_trace_{t.domain}.txt", "w") as f:
        f.write(result)
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(start_trace_pattern, line):
            action_index = int(re.search(start_trace_pattern, line).group(1)) - 1
            i += 1
            while i < len(lines) and not re.match(start_trace_pattern, lines[i]):
                if re.match(delete_fact_pattern, lines[i]):
                    fact = re.search(delete_fact_pattern, lines[i]).group(1)
                    map_action_to_effects[action_index]["effects"]["delete"].append(fact)
                elif re.match(add_fact_pattern, lines[i]):
                    fact = re.search(add_fact_pattern, lines[i]).group(1)
                    map_action_to_effects[action_index]["effects"]["add"].append(fact)
                i += 1
        else:
            i += 1

    return map_action_to_effects


                    
                        