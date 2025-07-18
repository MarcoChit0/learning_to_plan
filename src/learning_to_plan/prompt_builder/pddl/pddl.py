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

PDDL_TEMPLATE_PROMPT = Template(f"""Your task is to generate a plan for the following planning problem. The domain section describes the available actions and objects, and the problem section defines the initial and goal states. The plan must be a sequence of actions that starts from the initial state and reaches the goal state. The entire plan must be enclosed between {config.TOKENS.PLAN_START.value} and {config.TOKENS.PLAN_END.value} tags, and each action must be on a new line. Your response should only contain the plan.
                                
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
        s = "Here are some examples of plans in the same format as the one you should provide:"
        s += "\n\n".join([BASIC_EXAMPLE_TEMPLATE.substitute(
            domain=ex.read_domain().strip(),
            instance=ex.read_instance().strip(),
            plan=ex.get_plan()
        ) for ex in self.examples])
        return s

    def get_chat(self, t : task.Task, with_plan: bool = True, **kwargs) -> list[dict[str, str]]:
        content = PDDL_TEMPLATE_PROMPT.substitute(
            domain= t.domain,
            domain_description= t.read_domain().strip(),
            instance_content= t.read_instance().strip(),
            additional_info=self.get_additional_info(t)
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
        
