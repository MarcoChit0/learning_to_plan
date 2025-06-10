from __future__ import annotations
from copy import deepcopy
import threading
import abc
import re
import json
import os
from learning_to_plan import config
from typing import Optional
from enum import Enum

logger = config.get_logger(__name__)

instance_pattern = re.compile(r"instance-(\d+)\.pddl$")
lock = threading.Lock()

class Task(abc.ABC):
    class TYPE(Enum):
        TRAIN = "train"
        VALIDATION = "validation"
        TEST = "test"

    class PAAS_STATUS(Enum):
        OK = "ok"
        ERROR = "error"
    
    class DomainTranslator(abc.ABC):
        FACT_PATTERN = r"\(\s*[\w-]+\s*[\w\s-]*\)"

        def get_section_content(self, section_name: str, text: str) -> str | None:
            # --- Helper function to extract section content ---
            # Pattern explanation:
            # \(:section_name\s+   : Match the keyword (e.g., :init) followed by whitespace
            # (.*?)                : Non-greedily capture the content (re.DOTALL makes '.' match newline)
            # (?=\s+\(:|\s*\)\s*$) : Positive lookahead assertion:
            #    \s+\(:            : Ensure the match is followed by whitespace and the start of another section '(:...'
            #    |                 : OR
            #    \s*\)\s*$         : Ensure the match is followed by the final closing parenthesis of the define block
            #                       (allowing for potential whitespace)
            pattern = r"\({}\s+(.*?)(?=\s+\(:|\s*\)\s*$)".format(re.escape(section_name))
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip() # Return the stripped content
            return None
        
        def get_initial_state_facts_from_pddl_instance(self, pddl_instance: str) -> list[str]:
            """
            Extracts initial state facts from a PDDL instance.
            This method should be implemented to return a list of initial state facts in PDDL format.
            """
            init_content = self.get_section_content(":init", pddl_instance)
            if init_content is None:
                raise ValueError("Invalid PDDL instance format: Could not find :init section content.")
            
            init_facts_raw = re.findall(self.FACT_PATTERN, init_content)
            init_facts_pddl = [re.sub(r"\s+", " ", fact).strip() for fact in init_facts_raw if fact.strip()]
            init_facts_nl = []
            for fact in init_facts_pddl:
                try:
                    init_facts_nl.append(self.translate_pddl_fact_to_natural_language(fact))
                except ValueError as e:
                    logger.warning(f"Skipping initial fact due to conversion error: {e} (Fact: '{fact}')")
            return init_facts_nl
        
        def get_goal_facts_from_pddl_instance(self, pddl_instance: str) -> list[str]:
            """
            Extracts goal facts from a PDDL instance.
            This method should be implemented to return a list of goal facts in PDDL format.
            """
            goal_content = self.get_section_content(":goal", pddl_instance)
            if goal_content is None:
                raise ValueError("Invalid PDDL instance format: Could not find :goal section content.")

            goal_facts_raw = []
            and_match = re.search(r"^\(and\s+(.*)\s*\)$", goal_content, re.DOTALL | re.IGNORECASE)
            if and_match:
                and_content = and_match.group(1).strip()
                goal_facts_raw = re.findall(self.FACT_PATTERN, and_content)
            else:
                stripped_goal_content = goal_content.strip()
                if stripped_goal_content and re.fullmatch(self.FACT_PATTERN, stripped_goal_content):
                    goal_facts_raw = [stripped_goal_content]
                elif not stripped_goal_content:
                    pass 

            goal_facts_pddl = [re.sub(r"\s+", " ", fact).strip() for fact in goal_facts_raw if fact.strip()]

            goal_facts_nl = []
            for fact in goal_facts_pddl:
                try:
                    goal_facts_nl.append(self.translate_pddl_fact_to_natural_language(fact))
                except ValueError as e:
                    logger.warning(f"Skipping goal fact due to conversion error: {e} (Fact: '{fact}')")

            return goal_facts_nl

        @abc.abstractmethod
        def translate_pddl_fact_to_natural_language(self, fact: str) -> str:
            raise NotImplementedError("Subclasses must implement this method.")

        @abc.abstractmethod
        def translate_pddl_plan_to_natural_language(self, pddl_plan: str) -> str:
            raise NotImplementedError("Subclasses must implement this method.")
        
        @abc.abstractmethod
        def translate_natural_language_plan_to_pddl(self, nl_plan: str) -> str:
            raise NotImplementedError("Subclasses must implement this method.")
        
        @property
        @abc.abstractmethod
        def _domain_description_in_natural_language(self) -> str:
            raise NotImplementedError("Subclasses must implement this property.")


    def __init__(self, domain : str, domain_file_path : str, instance_file_path : str):
        self._domain :str = domain
        self._domain_file_path : str = domain_file_path
        self._instance_file_path : str = instance_file_path
        self._is_longer_plan : bool = True if config.LONG_INSTANCES in self._instance_file_path else False
        self._id : int = int(re.search(instance_pattern, self._instance_file_path).group(1))
        self._paas_status : Optional[Task.PAAS_STATUS] = None # ok, error
        self._pddl_plan : Optional[str] = None
        self._type : Optional[Task.TYPE] = None # training, validation, test | None

        if self._domain == "blocksworld":
            self._domain_translator : Task.DomainTranslator = BlocksworldTranslator()
        else:
            raise ValueError(f"Unknown domain: {self._domain}. Supported domains are: blocksworld.")

    def __lt__(self, other):
        if not isinstance(other, Task):
            return NotImplemented

        if self._domain != other._domain:
            return self._domain < other._domain

        if self._is_longer_plan != other._is_longer_plan:
            return not self._is_longer_plan

        self_match = instance_pattern.search(self._instance_file_path)
        other_match = instance_pattern.search(other._instance_file_path)
        if self_match and other_match:
            return int(self_match.group(1)) < int(other_match.group(1))
        else:
            return self._instance_file_path < other._instance_file_path

    def __str__(self):
        long_part = ", long" if self._is_longer_plan else ""
        type_part = f": {self._type.value}" if self._type else ""
        return f"Task {self._domain}, {self._id}{long_part}{type_part}"

    def __hash__(self):
        return hash((self._domain_file_path, self._instance_file_path))

    def __eq__(self, other):
        if not isinstance(other, Task):
            return NotImplemented
        return self._instance_file_path == other._instance_file_path and self._domain_file_path == other._domain_file_path

    def get_task_components_in_natural_language(self, with_plan:bool = True) -> dict[str, str]:
        domain_description = self._domain_translator._domain_description_in_natural_language.strip()
        pddl_instance = self.read_instance()
        initial_state_facts_nl = self._domain_translator.get_initial_state_facts_from_pddl_instance(pddl_instance=pddl_instance)
        goal_facts_nl = self._domain_translator.get_goal_facts_from_pddl_instance(pddl_instance=pddl_instance)
        if self._pddl_plan:
            plan_nl = self._domain_translator.translate_pddl_plan_to_natural_language(pddl_plan=self._pddl_plan)
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

    def get_chat(self, with_plan: bool = True, prompt_type : config.PROMPT_TYPE = config.PROMPT_TYPE.IO, **kwargs) -> list[dict[str, str]]:
        task_components_in_nl = self.get_task_components_in_natural_language(with_plan=with_plan)
        if prompt_type == config.PROMPT_TYPE.IO:
            # TODO: VERIFICAR SE A MODIFICAÇÃO NÃO CAUSOU ERROS
            initial_state_facts_str = "As initial conditions I have that: " + (", ".join(task_components_in_nl['initial_state_facts']))
            goal_facts_str = "My goal is to have that: " + (", ".join(task_components_in_nl['goal_facts']))      
            chat: list[dict[str, str]] = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{task_components_in_nl['domain_description']}\n{initial_state_facts_str}\n{goal_facts_str}"},
            ]       
            if with_plan:
                chat.append({"role": "assistant", "content": f"My plan is as follows:\n{config.TOKENS.PLAN_START.value}\n{task_components_in_nl['plan']}\n{config.TOKENS.PLAN_END.value}"})
            return chat
        elif prompt_type == config.PROMPT_TYPE.FEW_SHOT:
            few_shot = kwargs.get("few_shot", 1)
            few_shot_examples = get_few_shot_examples(few_shot=few_shot)
            examples = []
            initial_state_facts_str = "\n".join(task_components_in_nl['initial_state_facts'])
            goal_facts_str = "\n".join(task_components_in_nl['goal_facts'])
            for data in few_shot_examples:
                examples.append(f"""{config.TOKENS.EXAMPLE_START.value}
{config.TOKENS.DOMAIN_START.value}
{data['domain_description']}
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.INITIAL_STATE_START.value}
{data['initial_state_facts']}
{config.TOKENS.INITIAL_STATE_END.value}
{config.TOKENS.GOAL_START.value}
{data['goal_facts']}
{config.TOKENS.GOAL_END.value}
{config.TOKENS.PLAN_START.value}
{data['plan']}
{config.TOKENS.PLAN_END.value}
{config.TOKENS.EXAMPLE_END.value}\n""")
            
            examples_content = "\n".join(examples)
            
            content = f"""Your task is to find a plan for a Blocksworld problem based on the provided domain and instance. The following examples show the required output format. Your response should contain only the plan.

{config.TOKENS.DOMAIN_START.value}
{task_components_in_nl['domain_description']}
{config.TOKENS.DOMAIN_END.value}
{config.TOKENS.INITIAL_STATE_START.value}
{initial_state_facts_str}
{config.TOKENS.INITIAL_STATE_END.value}
{config.TOKENS.GOAL_START.value}
{goal_facts_str}
{config.TOKENS.GOAL_END.value}

{examples_content}

Here is a checklist to help you with your task:
1) Provide only the plan, without any additional text.
2) The plan must be in the same format as the examples above.
3) Use the tags "{config.TOKENS.PLAN_START.value}...{config.TOKENS.PLAN_END.value}" around the plan.
4) The actions in the plan must be from the set of actions in the domain described above, that is, they must use the same name and the same number of parameters as one of the action schemas.
5) The plan must be valid, that is, each action must be applicable in the state it is applied, and the plan must end in a goal state.
"""
            chat = [
                {"role": "system", "content": "You are an expert in AI Planning."},
                {"role": "user", "content": content}
            ]
            if with_plan:
                chat.append({"role": "assistant", "content": f"{config.TOKENS.PLAN_START.value}\n{task_components_in_nl['plan']}\n{config.TOKENS.PLAN_END.value}"})
            return chat
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}. Supported types are: {list(config.PROMPT_TYPE)}.")


    def to_json(self):
        try:
            data = {
                "domain": self._domain,
                "id": self._id,
                "domain_file_path": self._domain_file_path,
                "instance_file_path": self._instance_file_path,
                "paas_status": self._paas_status.value if self._paas_status else None,
                "pddl_plan": self._pddl_plan,
                "is_longer_plan": self._is_longer_plan,
                "type": self._type.value if self._type else None,
            }
        except (NotImplementedError, AssertionError, Exception) as e:
            logger.error(f"Error generating prompt for task {self._id}: {e}")
            raise e

        return json.dumps(data, ensure_ascii=False)


    def from_json(self, json_obj):
        # --- Basic Fields ---
        # id is derived in __init__, but we can load it from JSON if present
        json_id = json_obj.get("id")
        if json_id is not None:
            try:
                self._id = int(json_id)
            except (ValueError, TypeError):
                logger.error(f"Invalid id value in JSON: '{json_id}'. Expected integer.")
                # Decide if this should raise an error or just log
                raise ValueError(f"Invalid id value in JSON: '{json_id}'")

        # --- Optional Fields ---
        self._pddl_plan = json_obj.get("pddl_plan", None)

        # --- Enum Fields ---
        # Use correct internal field names (_pddl_status, _type)
        for field_name, json_key, enum_type in [
            ("_paas_status", "paas_status", Task.PAAS_STATUS),
            ("_type", "type", Task.TYPE)
        ]:
            json_value = json_obj.get(json_key)
            if json_value is not None:
                if not isinstance(json_value, str):
                    msg = f"Expected string for {json_key}, but got {type(json_value)}"
                    logger.error(msg)
                    raise ValueError(msg)
                try:
                    setattr(self, field_name, enum_type(json_value.strip()))
                except (ValueError, KeyError):
                    msg = f"Invalid {json_key} value in JSON: '{json_value}'"
                    logger.error(msg)
                    raise ValueError(msg)
                    
            else:
                setattr(self, field_name, None)

    def process_paas_response(self, response):
        plan = ""
        status = response.get("status", "error")
        if status == "ok":
            plan = response["result"]["output"]["sas_plan"]
        
        self._paas_status = Task.PAAS_STATUS(status) if status in [e.value for e in Task.PAAS_STATUS] else None
        self._pddl_plan = plan

    def read_instance(self):
        with lock and open(self._instance_file_path, "r", encoding='utf-8') as f:
            instance_content = f.read()
        return instance_content

    def read_domain(self):
        with lock and open(self._domain_file_path, "r", encoding='utf-8') as f:
            domain_content = f.read()
        return domain_content

class BlocksworldTranslator(Task.DomainTranslator):
        def __init__(self):
            self._color_map = {
            "a": "red",
            "b": "blue",
            "c": "yellow",
            "d": "orange",
            "e": "green",
            "f": "purple",
            "g": "pink",
            "h": "brown",
            "i": "gray",
            "j": "cyan",
            "k": "magenta",
            "l": "lime",
            "m": "navy",
            "n": "teal",
            "o": "coral",
            "p": "salmon",
            "q": "gold",
            "r": "khaki",
            "s": "lavender",
            "t": "plum",
            "u": "peach",
            "v": "tan",
            "w": "beige",
            "x": "ivory",
            "y": "mint",
            "z": "pearl",
        }

        def translate_pddl_fact_to_natural_language(self, pddl_fact:str) -> str:
            """
                Converts a PDDL fact into natural language, using color mapping.

                Args:
                    fact: str - a line of PDDL representing a fact (e.g., "(on a b)", "(clear c)").

                Returns:
                    str - The natural language representation of the fact.

                Raises:
                    ValueError: If the fact format is unknown or if a block identifier
                                cannot be mapped to a color using self._color_map.
            """
            # print(f"fact: @{fact}@")
            if not pddl_fact:
                return ""

            # Regex patterns for different fact types
            patterns = {
                "handempty": re.compile(r"^\(\s*handempty\s*\)$"),
                "holding": re.compile(r"^\(\s*holding\s+(\w+)\s*\)$"),
                "clear": re.compile(r"^\(\s*clear\s+(\w+)\s*\)$"),
                "ontable": re.compile(r"^\(\s*ontable\s+(\w+)\s*\)$"),
                "on": re.compile(r"^\(\s*on\s+(\w+)\s+(\w+)\s*\)$"),
            }

            match = patterns["handempty"].match(pddl_fact)
            if match:
                return "the hand is empty"

            match = patterns["holding"].match(pddl_fact)
            if match:
                block = match.group(1)
                if block not in self._color_map:
                    raise ValueError(f"Unknown block '{block}' in fact: {pddl_fact}")
                return f"you are holding the {self._color_map[block]} block"

            match = patterns["clear"].match(pddl_fact)
            if match:
                block = match.group(1)
                if block not in self._color_map:
                    raise ValueError(f"Unknown block '{block}' in fact: {pddl_fact}")
                return f"the {self._color_map[block]} block is clear"

            match = patterns["ontable"].match(pddl_fact)
            if match:
                block = match.group(1)
                if block not in self._color_map:
                    raise ValueError(f"Unknown block '{block}' in fact: {pddl_fact}")
                return f"the {self._color_map[block]} block is on the table"

            match = patterns["on"].match(pddl_fact)
            if match:
                block1 = match.group(1)
                block2 = match.group(2)
                if block1 not in self._color_map:
                    raise ValueError(f"Unknown block '{block1}' in fact: {pddl_fact}")
                if block2 not in self._color_map:
                    raise ValueError(f"Unknown block '{block2}' in fact: {pddl_fact}")
                return f"the {self._color_map[block1]} block is on top of the {self._color_map[block2]} block"
            # If none of the patterns matched
            raise ValueError(f"Unknown or malformed fact format: {pddl_fact}")


        def translate_pddl_plan_to_natural_language(self, pddl_plan:str) -> str:
            """
            Converts a PDDL plan into natural language, using color mapping.

            Args:
                plan: str - PDDL plan string, with actions potentially separated by newlines or semicolons.

            Returns:
                str - The natural language representation of the plan, with actions separated by newlines.

            Raises:
                ValueError: If an action format is unknown or if a block identifier
                    cannot be mapped to a color using self._color_map.
            """
            nl_actions = []
            # Regex to capture action name and arguments (1 or 2 blocks)
            action_pattern = re.compile(r"^\(\s*([\w-]+)\s+(\w+)(?:\s+(\w+))?\s*\)$")

            action_templates = {
                "unstack": "unstack the {color1} block from the {color2} block",
                "pick-up": "pick up the {color1} block",
                "stack": "stack the {color1} block on top of the {color2} block",
                "put-down": "put down the {color1} block",
            }

            # Normalize line endings, remove comments, split into lines
            lines = pddl_plan.split(";")[0].replace("\r\n", "\n").strip().split("\n")

            for line in lines:
                action_str = line.strip().lower()
                if not action_str:
                    continue # Skip empty lines

                match = action_pattern.match(action_str)
                if not match:
                    raise ValueError(f"Unknown or malformed action format: {action_str}")

                action_name, block1_id, block2_id = match.groups() # block2_id might be None

                if action_name not in action_templates:
                    raise ValueError(f"Unknown action type '{action_name}' in action: {action_str}")

                # Validate and map block IDs to colors
                if block1_id not in self._color_map:
                    raise ValueError(f"Unknown block '{block1_id}' in action: {action_str}")
                color1 = self._color_map[block1_id]

                format_args = {"color1": color1}

                # Handle actions requiring two blocks ("unstack", "stack")
                if action_name in ["unstack", "stack"]:
                    if not block2_id: # Error: Missing second block
                        raise ValueError(f"Action '{action_name}' requires two blocks, but found only one in: {action_str}")
                    # Second block exists, validate and add it
                    if block2_id not in self._color_map:
                        raise ValueError(f"Unknown block '{block2_id}' in action: {action_str}")
                    format_args["color2"] = self._color_map[block2_id]

                # Handle actions requiring one block ("pick-up", "put-down")
                elif action_name in ["pick-up", "put-down"]:
                    if block2_id: # Error: Unexpected second block
                        raise ValueError(f"Action '{action_name}' requires one block, but found two in: {action_str}")
                    # Correct case for one-block actions: block2_id is None, nothing else needed.


                try:
                    nl_action = action_templates[action_name].format(**format_args)
                    nl_actions.append(nl_action)
                except KeyError as e:
                # This might happen if the template expects color2 but it wasn't provided
                    raise ValueError(f"Formatting error for action '{action_name}'. Missing argument: {e}. Action: {action_str}")


            return "\n".join(nl_actions)

        def translate_natural_language_plan_to_pddl(self, nl_plan:str) -> str:
            """
            Converts a natural language plan into PDDL format.

            Args:
                plan: A string containing actions in natural language, separated by newlines or semicolons.

            Returns:
                A string representing the plan in PDDL format.

            Raises:
                ValueError: If an unknown or malformed action is encountered.
            """
            pddl_actions = []
            action_mapping = {
                "pick up": "pick-up",
                "put down": "put-down",
                "unstack": "unstack",
                "stack": "stack"
            }
            for line in nl_plan.replace(";", "\n").strip().split("\n"):
                action_str = line.strip().lower()
                if not action_str:
                    continue

                # Regex to match action and extract block IDs
                action_pattern = re.compile(r"^(pick up|put down|unstack|stack) the (\w+) block(?: from the (\w+) block)?(?: on top of the (\w+) block)?$")
                match = action_pattern.match(action_str)
                if not match:
                    logger.warning(f"Skipping action due to unknown format: {action_str}")
                    continue
                    # raise ValueError(f"Unknown or malformed action format: {action_str}")
                
                action_name, first_block_color, unstack_block_color, stack_block_color = match.groups()
                if action_name not in action_mapping:
                    raise ValueError(f"Unknown action type '{action_name}' in action: {action_str}")
                
                first_block_color = first_block_color.strip()
                id_1 = next((k for k, v in self._color_map.items() if v == first_block_color), None)
                if not id_1:
                    raise ValueError(f"Unknown block color '{first_block_color}' in action: {action_str}")
                
                if action_name in ["pick up", "put down"]:
                    if unstack_block_color or stack_block_color:
                        raise ValueError(f"Action '{action_name}' should not have additional block colors: {action_str}")
                    pddl_actions.append(f"({action_mapping[action_name]} {id_1})")
                elif action_name == "unstack":
                    if not unstack_block_color:
                        raise ValueError(f"Action '{action_name}' requires a second block color: {action_str}")
                    if stack_block_color:
                        raise ValueError(f"Action '{action_name}' should not have additional block colors: {action_str}")
                    unstack_block_color = unstack_block_color.strip()
                    id_2 = next((k for k, v in self._color_map.items() if v == unstack_block_color), None)
                    if not id_2:
                        raise ValueError(f"Unknown block color '{unstack_block_color}' in action: {action_str}")
                    pddl_actions.append(f"({action_mapping[action_name]} {id_1} {id_2})")
                elif action_name == "stack":
                    if not stack_block_color:
                        raise ValueError(f"Action '{action_name}' requires a second block color: {action_str}")
                    if unstack_block_color:
                        raise ValueError(f"Action '{action_name}' should not have additional block colors: {action_str}")
                    stack_block_color = stack_block_color.strip()
                    id_2 = next((k for k, v in self._color_map.items() if v == stack_block_color), None)
                    if not id_2:
                        raise ValueError(f"Unknown block color '{stack_block_color}' in action: {action_str}")
                    pddl_actions.append(f"({action_mapping[action_name]} {id_1} {id_2})")
                else:
                    raise ValueError(f"Unknown action type '{action_name}' in action: {action_str}")

            # Join actions into a single PDDL string
            pddl_plan = "\n".join(pddl_actions)
            return pddl_plan

        @property
        def _domain_description_in_natural_language(self) -> str:

            return """I am playing with a set of blocks where I need to arrange the blocks into stacks.
Here are the actions that can be performed:
Pick up a block
Unstack a block from on top of another block
Put down a block
Stack a block on top of another block
The following are the restrictions on the actions:
I can only pick up or unstack one block at a time.
I can only pick up or unstack a block if my hand is empty.
I can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.
I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.
I can only unstack a block from on top of another block if the block I am unstacking is clear.
Once I pick up or unstack a block, I am holding the block.
I can only put down a block that I am holding.
I can only stack a block on top of another block if I am holding the block being stacked.
I can only stack a block on top of another block if the block onto which I am stacking is clear.
Once I put down or stack a block, my hand becomes empty.
Once you stack a block on top of a second block, the second block is no longer clear.\n"""


# Removed unused import 'Dataset'
# from datasets import Dataset
def get_tasks_from_domain_directory(domain: str) -> set[Task]:
    # Assuming config.RAW_DIR, config.DOMAIN_FILE_NAME, config.BASIC_INSTANCES, config.LONG_INSTANCES are defined
    domain_file_path = os.path.join(config.RAW_DIR, domain, config.DOMAIN_FILE_NAME)

    # Determine which instance directories to include
    instance_dirs = [
        os.path.join(config.RAW_DIR, domain, config.BASIC_INSTANCES), # training, validation, test instances
        os.path.join(config.RAW_DIR, domain, config.LONG_INSTANCES) # out of distribution instances
    ]
    # Collect tasks
    tasks:set[Task] = set()
    for instance_dir in instance_dirs:
        if not os.path.exists(instance_dir):
            raise ValueError(f"Instance directory not found: {instance_dir}")

        for file_name in os.listdir(instance_dir):
            if instance_pattern.search(file_name):
                tasks.add(
                    Task(
                        domain=domain,
                        domain_file_path=domain_file_path,
                        instance_file_path=os.path.join(instance_dir, file_name)
                    )
                )

    return tasks

DATASET: set[Task] = set()
def get_dataset() -> set[Task]:
    global DATASET
    if not DATASET:
        raise ValueError("Dataset is empty. Please load the dataset first.")
    return DATASET
    

def get_tasks(filter_by_domain: Optional[str] = None,  filter_by_type: Optional[Task.TYPE] = None, is_longer_plan:Optional[bool] = None, number_of_instances: Optional[int] = None) -> set[Task]:
    global DATASET
    if not DATASET:
        raise ValueError("Dataset is empty. Please load the dataset first.")
    tasks = DATASET
    # --- Filter by domain and type ---
    if filter_by_domain:
        tasks = {t for t in tasks if t._domain == filter_by_domain}
        if len(tasks) == 0:
            raise ValueError(f"No tasks found for domain '{filter_by_domain}'.")
    if filter_by_type:
        tasks = {t for t in tasks if t._type == filter_by_type}
        if len(tasks) == 0:
            raise ValueError(f"No tasks found for type '{filter_by_type}'.")
    # --- Filter by basic or long tasks ---
    if is_longer_plan is not None:
        tasks = {t for t in tasks if t._is_longer_plan == is_longer_plan}
        if len(tasks) == 0:
            raise ValueError(f"No tasks found with is_longer_plan={is_longer_plan}.")
    # --- Limit number of instances ---
    if number_of_instances is not None and isinstance(number_of_instances, int):
        tasks = set(sorted(tasks)[:min(number_of_instances, len(tasks))])
        if len(tasks) == 0:
            raise ValueError(f"No tasks found after filtering.")
    return tasks

def get_task(domain_file_path: str, instance_file_path: str) -> Task:
    global DATASET
    if not DATASET:
        raise ValueError("Dataset is empty. Please load the dataset first.")
    for task in DATASET:
        if task._domain_file_path == domain_file_path and task._instance_file_path == instance_file_path:
            return task
    raise ValueError(f"Task with domain file path '{domain_file_path}' and instance file path '{instance_file_path}' not found.")

def load() -> None:
    jsonl_file_path = config.TASKS_DATASET_FILE_PATH
    global DATASET
    if not os.path.exists(jsonl_file_path):
        raise ValueError(f"JSONL file not found: {jsonl_file_path}")
    tasks = set()
    logger.info(f"Loading tasks from {jsonl_file_path}.")
    with open(jsonl_file_path, "r", encoding='utf-8') as f:
        for line in f:
            try:
                json_obj = json.loads(line)
                domain = json_obj.get("domain", None)
                instance_file_path = json_obj.get("instance_file_path", None)
                domain_file_path = json_obj.get("domain_file_path", None)
                assert domain, "Domain is not specified in the JSON object."
                assert instance_file_path, "Instance file path is not specified in the JSON object."
                assert domain_file_path, "Domain file path is not specified in the JSON object."
                task = Task(
                    domain,
                    domain_file_path,
                    instance_file_path
                )
                task.from_json(json_obj)
                tasks.add(task)
            except Exception as e:
                m = f"Error processing task from file {jsonl_file_path}: {e}"
                # Changed config.log to logger.error
                logger.error(m)
                raise e
    DATASET = tasks
    logger.info(f"Loaded {len(DATASET)} tasks from {jsonl_file_path}.")

def save()-> None:
    jsonl_file_path = config.TASKS_DATASET_FILE_PATH
    global DATASET
    if not DATASET:
        raise ValueError("Dataset is empty. Please load the dataset first.")
    logger.info(f"Saving {len(DATASET)} tasks to {jsonl_file_path}.")
    with open(jsonl_file_path, "w", encoding='utf-8') as f:
        for task in sorted(DATASET):
            try:
                json_str = task.to_json() # Get the JSON string representation
                f.write(json_str + "\n") # Write the JSON string followed by a newline
            except Exception as e:
                m = f"Error saving task to file {jsonl_file_path}: {e}"
                # Changed config.log to logger.error
                logger.error(m)
                raise e
    logger.info(f"Saved {len(DATASET)} tasks to {jsonl_file_path}.")

import numpy as np
def get_few_shot_examples(few_shot:int) -> list[dict[str, str]]:
    # TODO: LATER, CHANGE THIS FUNCTION TO USE ONLY TEST TASKS AND FROM OTHER DOMAINS
    # FOR NOW, WE HAVE A LIST OF SAMPLES THAT WE CAN USE AS FEW-SHOT EXAMPLES
    gripper_data = {
        "domain_description": """I have to plan how to move objects between rooms using a robot with grippers. The robot can move between rooms and pick up or drop objects using its grippers.
Here are the actions that can be performed:
Move the robot from one room to another room.
A robot pick up an object from a room.
A robot drop an object in a room.
The following are the restrictions on the actions:
A robot can move from one room to another room only if the robot is in the from-room.
Once the robot has moved from one room to another room, the robot is no longer in the from-room and is in the to-room.
A robot can pick up an object from a room only if the robot is in the room and the object is also in the same room.
A robot can pick up an object from a room only if the robot's gripper is free.
Once the robot has picked up an object from a room, the object is no longer in the room and is carried by the robot.
Once the robot has picked up an object from a room, the robot's gripper is no longer free.
A robot can drop an object in a room only if the robot is in the room and the robot is carrying the object.
Once the robot has dropped an object in a room, the object is in the room and is no longer carried by the robot.
Once the robot has dropped an object in a room, the robot's gripper is free.""",
        "initial_state_facts": """ball_1 is at room_1
ball_2 is at room_1
ball_3 is at room_2
ball_4 is at room_2
ball_5 is at room_1
ball_6 is at room_1
ball_7 is at room_2
ball_8 is at room_1
ball_9 is at room_1
robot_1 is at room_2
robot_2 is at room_2
robot_3 is at room_2
robot_4 is at room_2
robot_1's left_gripper_1 is free
robot_1's right_gripper_1 is free
robot_2's left_gripper_2 is free
robot_2's right_gripper_2 is free
robot_3's left_gripper_3 is free
robot_3's right_gripper_3 is free
robot_4's left_gripper_4 is free
robot_4's right_gripper_4 is free""",
        "goal_facts": """ball_1 is at room_2
ball_2 is at room_2
ball_3 is at room_1
ball_4 is at room_2
ball_5 is at room_2
ball_6 is at room_1
ball_7 is at room_2
ball_8 is at room_1
ball_9 is at room_1""",
        "plan": """robot_4 picks ball_3 at room_2 with left_gripper_4
move robot_4 from room_2 to room_1
robot_4 drops ball_3 at room_1 with left_gripper_4
robot_4 picks ball_1 at room_1 with left_gripper_4
move robot_4 from room_1 to room_2
robot_4 drops ball_1 at room_2 with left_gripper_4
move robot_1 from room_2 to room_1
robot_1 picks ball_2 at room_1 with left_gripper_1
robot_1 picks ball_5 at room_1 with right_gripper_1
move robot_1 from room_1 to room_2
robot_1 drops ball_2 at room_2 with left_gripper_1
robot_1 drops ball_5 at room_2 with right_gripper_1"""
    }
    childsnack_data = {
        "domain_description": """I have to plan how to make and serve sandwiches for a group of children, taking into account that some of them are allergic to gluten.
There are two types of sandwiches: regular and gluten-free.
Here are the actions that can be performed:
Make a gluten-free sandwich.
Make a regular sandwich.
Put a sandwich on a tray.
Serve a gluten-free sandwich to an allergic child.
Serve a regular sandwich to a child.
Move a tray between kitchen and tables.
The following are the restrictions on the actions:
We can make a gluten-free sandwich only if there is a bread at kitchen and the bread is gluten-free.
We can make a gluten-free sandwich only if there is a content at kitchen and the content is gluten-free.
Once we make a gluten-free sandwich, the bread and content are no longer at kitchen.
Once we make a gluten-free sandwich, the sandwich is at kitchen and is gluten-free.
We can make a regular sandwich only if there is a bread at kitchen.
We can make a regular sandwich only if there is a content at kitchen.
Once we make a regular sandwich, the bread and content are no longer at kitchen.
Once we make a regular sandwich, the sandwich is at kitchen.
We can put a sandwich on a tray only if the sandwich is at kitchen.
We can put a sandwich on a tray only if the tray is also at kitchen.
Once we put a sandwich on a tray, the sandwich is no longer at kitchen but is on the tray.
We can serve a gluten-free sandwich to an allergic child only if the child is allergic to gluten.
We can serve a gluten-free sandwich to an allergic child only if the sandwich is on a tray and the sandwich is gluten-free.
We can serve a gluten-free sandwich to an allergic child only if the child is waiting for the sandwich at the table.
We can serve a regular sandwich to a child only if the tray is at the table where the child is waiting.
Once we serve a gluten-free sandwich to an allergic child, the sandwich is no longer on the tray.
Once we serve a gluten-free sandwich to an allergic child, we say the child has been served.
We can serve a regular sandwich to a child only if the child is not allergic to gluten.
We can serve a regular sandwich to a child only if the child is waiting for the sandwich at the table.
We can serve a regular sandwich to a child only if the sandwich is on a tray and the tray is at the table where the child is waiting.
Once we serve a regular sandwich to a child, the sandwich is no longer on the tray.
Once we serve a regular sandwich to a child, we say the child has been served.
We can move a tray from from-place to to-place only if the tray is at from-place.
Once we move a tray from from-place to to-place, the tray is no longer at from-place but is at to-place.""",
        "initial_state_facts": """child_2 is allergic to gluten
tray_1 is at kitchen
tray_2 is at kitchen
tray_3 is at kitchen
bread_1 is at kitchen
bread_2 is at kitchen
bread_3 is at kitchen
content_1 is at kitchen
content_2 is at kitchen
content_3 is at kitchen
bread_2 is gluten-free
content_3 is gluten-free
child_1 is not allergic to gluten
child_3 is not allergic to gluten
sandwich_1 is not ready yet
sandwich_2 is not ready yet
sandwich_3 is not ready yet
sandwich_4 is not ready yet
sandwich_5 is not ready yet
sandwich_6 is not ready yet
child_1 is waiting for sandwich at table_2
child_2 is waiting for sandwich at table_2
child_3 is waiting for sandwich at table_2""",
        "goal_facts": """child_1 has been served
child_2 has been served
child_3 has been served""",
        "plan": """make a gluten-free sandwich_1 using bread_2 and content_3
put sandwich_1 on tray_3
move tray_3 from kitchen to table_2
use tray_3 to serve gluten-free sandwich_1 to child_2 at table_2
make a regular sandwich_6 using bread_1 and content_1
put sandwich_6 on tray_2
move tray_2 from kitchen to table_2
use tray_2 to serve regular sandwich_6 to child_1 at table_2
make a regular sandwich_5 using bread_3 and content_2
put sandwich_5 on tray_1
move tray_1 from kitchen to table_2
use tray_1 to serve regular sandwich_5 to child_3 at table_2"""
    }
    logistics_data = {
        "domain_description": """I have to plan logistics to transport packages within cities via trucks and between cities via airplanes. Locations within a city are directly connected (trucks can move between any two such locations), and so are the cities. In each city there is exactly one truck and each city has one location that serves as an airport.
Here are the actions that can be performed:
Load a package into a truck.
Load a package into an airplane.
Unload a package from a truck.
Unload a package from an airplane.
Drive a truck from one location to another location.
Fly an airplane from one city to another city.
The following are the restrictions on the actions:
A package can be loaded into a truck only if the package and the truck are in the same location.
Once a package is loaded into a truck, the package is not at the location and is in the truck.
A package can be loaded into an airplane only if the package and the airplane are in the same location.
Once a package is loaded into an airplane, the package is not at the location and is in the airplane.
A package can be unloaded from a truck only if the package is in the truck.
Once a package is unloaded from a truck, the package is not in the truck and is at the location of the truck.
A package can be unloaded from an airplane only if the package is in the airplane.
Once a package is unloaded from an airplane, the package is not in the airplane and is at the location of the airplane.
A truck can be driven from one location to another if the truck is at the from-location and both from-location and to-location are locations in the same city.
Once a truck is driven from one location to another, it is not at the from-location and is at the to-location.
An airplane can be flown from one city to another if the from-location and the to-location are airports and the airplane is at the from-location.
Once an airplane is flown from one city to another the airplane is not at the from-location and is at the to-location.""",
        "initial_state_facts": """location_0-0 is an airport
location_1-0 is an airport
airplane_0 is at location_0-0
airplane_1 is at location_0-0
airplane_2 is at location_1-0
package_0 is at location_1-0
package_1 is at location_0-1
package_2 is at location_1-0
package_3 is at location_0-0
package_4 is at location_0-1
truck_0 is at location_0-0
truck_1 is at location_1-1
location_0-0 is in the city city_0
location_0-1 is in the city city_0
location_1-0 is in the city city_1
location_1-1 is in the city city_1""",
        "goal_facts": """package_0 is at location_1-1
package_1 is at location_0-0
package_2 is at location_1-1
package_3 is at location_0-1
package_4 is at location_0-1""",
        "plan": """load package_3 into truck_0 at location_0-0
drive truck_1 from location_1-1 to location_1-0 in city_1
load package_2 into truck_1 at location_1-0
load package_0 into truck_1 at location_1-0
drive truck_1 from location_1-0 to location_1-1 in city_1
unload package_2 from truck_1 at location_1-1
unload package_0 from truck_1 at location_1-1
drive truck_0 from location_0-0 to location_0-1 in city_0
unload package_3 from truck_0 at location_0-1
load package_1 into truck_0 at location_0-1
drive truck_0 from location_0-1 to location_0-0 in city_0
unload package_1 from truck_0 at location_0-0"""
    }
    satellite_data = {
        "domain_description": """I have to plan how to operate satellites in space equipped with various instruments. The satellites can be turned to point in different directions, their instruments can be switched on or off, calibrated, and used to take images using specific modes.
Here are the actions that can be performed:
Turn a satellite to a direction.
Switch on an instrument on a satellite.
Switch off an instrument on a satellite.
Calibrate an instrument on a satellite by pointing it to a calibration target.
Take an image of a direction using an instrument with a specific mode.
The following are the restrictions on the actions:
A satellite can be turned from a previous direction to a new direction only if the satellite is currently pointing to the previous direction. Once turned, it is no longer pointing to the previous direction.
An instrument can be switched on on a satellite only if the instrument is on board and power-available. Once switched on, the instrument becomes power-on, is no longer power-available, and becomes uncalibrated.
An instrument can be switched off on a satellite only if the instrument is on board and power-on. Once switched off, the instrument becomes power-available and is no longer power-on.
An instrument can be calibrated on a satellite only if it is on board, power-on, and pointing to its calibration target. Once calibrated, the instrument is marked as calibrated.
To take an image, an instrument must be on board, power-on, calibrated, the satellite must be pointing to the required direction, and the instrument must support the specified mode.
Once an image is taken, the image is available.""",
                "initial_state_facts": """the calibration target of instrument_0 is star_2
the calibration target of instrument_1 is star_2
the calibration target of instrument_2 is ground_station_1
the calibration target of instrument_3 is ground_station_1
the calibration target of instrument_4 is ground_station_1
the calibration target of instrument_5 is ground_station_4
the calibration target of instrument_6 is ground_station_4
the calibration target of instrument_7 is ground_station_4
the calibration target of instrument_8 is ground_station_4
the calibration target of instrument_9 is star_2
instrument_0 is on board satellite_0
instrument_1 is on board satellite_1
instrument_2 is on board satellite_1
instrument_3 is on board satellite_2
instrument_4 is on board satellite_2
instrument_5 is on board satellite_3
instrument_6 is on board satellite_3
instrument_7 is on board satellite_4
instrument_8 is on board satellite_4
instrument_9 is on board satellite_4
satellite_0 is pointing to star_0
satellite_1 is pointing to ground_station_3
satellite_2 is pointing to star_6
satellite_3 is pointing to star_0
satellite_4 is pointing to star_2
satellite_0 is power-available
satellite_1 is power-available
satellite_2 is power-available
satellite_3 is power-available
satellite_4 is power-available
instrument_0 supports image_mode_0
instrument_0 supports infrared_mode_1
instrument_1 supports image_mode_0
instrument_1 supports infrared_mode_1
instrument_2 supports infrared_mode_1
instrument_3 supports image_mode_0
instrument_3 supports infrared_mode_1
instrument_4 supports infrared_mode_1
instrument_5 supports infrared_mode_1
instrument_6 supports image_mode_0
instrument_7 supports image_mode_0
instrument_7 supports infrared_mode_1
instrument_8 supports infrared_mode_1
instrument_9 supports image_mode_0
instrument_9 supports infrared_mode_1""",
                "goal_facts": """phenomenon_5 has image in infrared_mode_1
star_6 has image in infrared_mode_1
star_7 has image in infrared_mode_1
star_8 has image in infrared_mode_1
satellite_0 is pointing to star_6""",
        "plan": """switch on instrument_9 on satellite_4
calibrate instrument_9 on satellite_4 pointing to calibration target star_2
turn satellite_4 from phenomenon_5 to star_2
take image of phenomenon_5 using instrument_9 on satellite_4 with infrared_mode_1
turn satellite_4 from star_6 to phenomenon_5
take image of star_6 using instrument_9 on satellite_4 with infrared_mode_1
turn satellite_4 from star_7 to star_6
take image of star_7 using instrument_9 on satellite_4 with infrared_mode_1
turn satellite_4 from star_8 to star_7
take image of star_8 using instrument_9 on satellite_4 with infrared_mode_1
turn satellite_0 from star_6 to star_0"""
    }
    barman_data = {
        "domain_description": """I have to plan actions for a robotic bartender to prepare cocktails. The bartender has two hands and works with various containers and ingredients to mix and serve drinks.
Here are the actions that can be performed:
Grasp a container (shot or shaker) from the table.
Leave a container on the table.
Fill-shot with an ingredient from a dispenser.
Refill-shot with the same ingredient it contained before.
Empty a shot.
Clean a shot.
Pour the content of a shot into a clean shaker.
Pour the content of a shot into a used shaker that already contain some ingredient.
Empty a shaker.
Clean a shaker.
Shake the shaker to mix the ingredients.
Pour the content of a shaker into a shot.""",
        "initial_state_facts": """shaker_23 is clean
shot_295 is clean
cocktail_1 has ingredient_163 as its first ingredient
cocktail_1 has ingredient_383 as its second ingredient
dispenser_114 dispenses ingredient_163
dispenser_213 dispenses ingredient_383
shaker_23 is empty
shot_295 is empty
left_hand is empty
right_hand is empty
level_1 is the next level after level_0
level_2 is the next level after level_1
shaker_23 is on the table
shot_295 is on the table
shaker_23's zero fill level is at level_0
shaker_23's fill level is at level_0""",
        "goal_facts": """shot_295 contains cocktail_1""",
        "plan": """grasp the shaker_23 using left_hand
grasp the shot_295 using right_hand
leave the shaker_23 using left_hand
fill the shot_295 on right_hand with ingredient_163 using dispenser_114 when left_hand is empty
pour from shot_295 containing ingredient_163 to clean shaker_23 using right_hand from level_0 to level_1
clean the shot_295 on right_hand used for ingredient_163 when left_hand is empty
fill the shot_295 on right_hand with ingredient_383 using dispenser_213 when left_hand is empty
pour from shot_295 containing ingredient_383 to used shaker_23 using right_hand from level_1 to level_2
clean the shot_295 on right_hand used for ingredient_383 when left_hand is empty
grasp the shaker_23 using left_hand
leave the shot_295 using right_hand
shake shaker_23 on left_hand containing ingredient_163 and ingredient_383 to get cocktail_1 when right_hand is empty
pour from shaker_23 to shot_295 containing cocktail_1 using left_hand from level_2 to level_1"""}
    data = [
        gripper_data,
        childsnack_data,
        logistics_data,
        satellite_data,
        barman_data
    ]
    rng = np.random.RandomState(config.RANDOM_SEED)
    return rng.choice(data, size=few_shot, replace=False).tolist() 