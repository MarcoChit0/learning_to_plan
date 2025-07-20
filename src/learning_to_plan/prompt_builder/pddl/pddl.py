from string import Template
from learning_to_plan import config
from learning_to_plan.prompt_builder import base
from learning_to_plan.data import task
from learning_to_plan import database

BASIC_EXAMPLE_TEMPLATE = Template(f"""{config.TOKENS.EXAMPLE_START.value}
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

THINKING_EXAMPLE_TEMPLATE = Template(f"""{config.TOKENS.EXAMPLE_START.value}
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

BASIC_PDDL_TEMPLATE_PROMPT = Template(f"""Your task is to generate a plan for the following planning problem. The domain section describes the available actions and objects, and the problem section defines the initial and goal states. The plan must be a sequence of actions that starts from the initial state and reaches the goal state. The entire plan must be enclosed between {config.TOKENS.PLAN_START.value} and {config.TOKENS.PLAN_END.value} tags, and each action must be on a new line. Your response should only contain the plan.
                                
{config.TOKENS.DOMAIN_START.value}
$domain_description
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.PROBLEM_START.value}
$instance_content
{config.TOKENS.PROBLEM_END.value}
$additional_info
          
Here is a checklist to help you with your task:

{config.TOKENS.CHECKLIST_START.value}
1) Do not provide any additional text or explanations outside the plan tags.
2) The plan must be in the same format as the examples above.
3) The plan should be preceded by the {config.TOKENS.PLAN_START.value} tag and should be followed by the {config.TOKENS.PLAN_END.value} tag.
4) The actions in the plan must be from the set of actions in the domain described above, that is, they must use the same name and the same number of parameters as one of the action schemas.
5) The plan must be valid, that is, each action must be applicable in the state it is applied, and the plan must end in a goal state.
{config.TOKENS.CHECKLIST_END.value}""")

THINKING_PDDL_TEMPLATE_PROMPT = Template(f"""Your task is to generate a plan for the following planning problem. The domain section describes the available actions and objects, and the problem section defines the initial and goal states. The plan must be a sequence of actions that starts from the initial state and reaches the goal state. The entire plan must be enclosed between {config.TOKENS.PLAN_START.value} and {config.TOKENS.PLAN_END.value} tags, and each action must be on a new line. You must provide the plan in your response. You may also provide your reasoning in the thinking section before the plan, which should be enclosed between {config.TOKENS.THINKING_START.value} and {config.TOKENS.THINKING_END.value} tags. Your response should only contain the reasoning and the plan.
                     
{config.TOKENS.DOMAIN_START.value}
$domain_description
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.PROBLEM_START.value}
$instance_content
{config.TOKENS.PROBLEM_END.value}
$additional_info

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

class PDDLPromptBuilder(base.PromptBuilder):

    def __init__(self, prompt_type=config.PROMPT_TYPE.PDDL, **kwargs):
        super().__init__(prompt_type=prompt_type, **kwargs)
        self.examples : list[task.Task] = []
        for d in ["hanoi", "storage"]:
            tasks : set[task.Task] = database.task_database.get(filter_by_domain=d)
            assert len(tasks) > 0, f"No tasks found for the {d} domain."
            self.examples.append(sorted(tasks)[-1])

    def get_additional_info(self, t: task.Task) -> str:
        _ = t  # Unused task
        s = "Here are some examples of plans in the same format as the one you should provide:\n\n"

        if self.thinking_style == config.THINKING_STYLE.NONE:
            template = BASIC_EXAMPLE_TEMPLATE
        else:
            template = THINKING_EXAMPLE_TEMPLATE
        
        examples = []
        for ex in self.examples:
            try:
                domain_description = ex.read_domain().strip()
                instance_content = ex.read_instance().strip()
            except Exception as e:
                logger.error(f"Error reading PDDL domain and instance for task {ex.id}: {e}")
                raise e
            
            ex_data = {
                "domain": domain_description,
                "instance": instance_content,
                "plan": ex.get_plan(),
            }
            if self.thinking_style != config.THINKING_STYLE.NONE:
                try:
                    ex_data["thinking"] = self.get_thinking(ex).strip()
                except Exception as e:
                    logger.error(f"Error getting thinking for task {ex.id}: {e}")
                    raise e
            
            examples.append(template.substitute(**ex_data))
        s += "\n\n".join(examples)
        return s

    def get_chat(self, t : task.Task, with_plan: bool = True, **kwargs) -> list[dict[str, str]]:
        _ = kwargs # Unused kwargs
        if self.thinking_style == config.THINKING_STYLE.NONE:
            prompt = BASIC_PDDL_TEMPLATE_PROMPT
        else:
            prompt = THINKING_PDDL_TEMPLATE_PROMPT
        try:
            domain_description = t.read_domain().strip()
            instance_content = t.read_instance().strip()
        except Exception as e:
            logger.error(f"Error reading pddl domain and instance for task {t.id}: {e}")
            raise e

        try:
            additional_info = self.get_additional_info(t)
        except Exception as e:
            logger.error(f"Error getting additional info for task {t.id}: {e}")
            raise e
        
        content = prompt.substitute(
            domain=t.domain,
            domain_description=domain_description,
            instance_content=instance_content,
            additional_info=additional_info
        )
        
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
    
    def get_thinking(self, t : task.Task) -> str:
        """
        Returns the thinking style for the task.
        :param t: The task to execute.
        :return: The thinking style for the task.
        """
        if self.thinking_style == config.THINKING_STYLE.NONE:
            return ""
        
        elif self.thinking_style == config.THINKING_STYLE.COT:
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
            thinking = f"{config.TOKENS.STATE_START.value}\n"
            thinking += "\n".join(sorted(state_facts))
            thinking += f"\n{config.TOKENS.STATE_END.value}\n\n"
            for action_index, action_data in plan_data.items():
                action = action_data["name"]
                effects = action_data["effects"]
                thinking += f"Action: {action}\n"
                if effects["delete"]:
                    delete_effects = '\n'.join(effects['delete'])
                    thinking += f"{config.TOKENS.DELETE_EFFECTS_START.value}\n{delete_effects}\n{config.TOKENS.DELETE_EFFECTS_END.value}\n"
                if effects["add"]:
                    add_effects = '\n'.join(effects['add'])
                    thinking += f"{config.TOKENS.ADD_EFFECTS_START.value}\n{add_effects}\n{config.TOKENS.ADD_EFFECTS_END.value}\n\n"
                
                state_facts = state_facts - set(effects["delete"])
                state_facts = state_facts.union(set(effects["add"]))
                thinking += f"{config.TOKENS.STATE_START.value}\n"
                thinking += "\n".join(sorted(state_facts))
                thinking += f"\n{config.TOKENS.STATE_END.value}\n"

                if action_index < len(plan_data) - 1:
                    thinking += "\n"
            return thinking

        else :
            raise ValueError(f"Unknown thinking style: {self.thinking_style}. Please provide a valid thinking style.")

from learning_to_plan import utils 
import re
def extract_plan_add_and_delete_data(t : task.Task) -> str:
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


                    
                        