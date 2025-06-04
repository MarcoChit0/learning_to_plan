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

        def get_section_content(section_name: str, text: str) -> str | None:
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
                chat.append({"role": "assistant", "content": f"My plan is as follows:\n{config.TOKENS.PLAN_START}\n{task_components_in_nl['plan']}\n{config.TOKENS.PLAN_END}"})
            return chat
        elif prompt_type == config.PROMPT_TYPE.FEW_SHOT:
            if self._type == Task.TYPE.TRAIN or self._type == Task.TYPE.VALIDATION:
                available_task_types = {Task.TYPE.TEST}
            else:
                available_task_types = {Task.TYPE.TRAIN, Task.TYPE.VALIDATION}
            few_shot_examples = get_few_shot_examples(
                domain= self._domain,
                available_task_types = available_task_types, 
                few_shot = kwargs.get('few_shot', 0),
                is_longer_plan = False, 
                random_seed = kwargs.get('random_seed', 42),
            )
            assert len(few_shot_examples) > 0, "At least one few-shot example is required."

            examples = []
            for t in few_shot_examples:
                eg_components_nl = t.get_task_components_in_natural_language(with_plan=True)
                examples.append(f"""{config.TOKENS.EXAMPLE_START}
{config.TOKENS.INITIAL_STATE_START}
{eg_components_nl['initial_state_facts']}
{config.TOKENS.INITIAL_STATE_END}
{config.TOKENS.GOAL_START}
{eg_components_nl['goal_facts']}
{config.TOKENS.GOAL_END}
{config.TOKENS.PLAN_START}
{eg_components_nl['plan']}
{config.TOKENS.PLAN_END}
{config.TOKENS.EXAMPLE_END}""")
            
            examples_content = "\n".join(examples)
            
            content = f"""{config.TOKENS.DOMAIN_START}
{task_components_in_nl['domain_description']}
{config.TOKENS.DOMAIN_END}

{config.TOKENS.EXAMPLE_START}
{examples_content}
{config.TOKENS.EXAMPLE_END}

{config.TOKENS.INITIAL_STATE_START}
{task_components_in_nl['initial_state_facts']}
{config.TOKENS.INITIAL_STATE_END}
{config.TOKENS.GOAL_START}
{task_components_in_nl['goal_facts']}
{config.TOKENS.GOAL_END}

Provide only the plan for the given instance. Here is a checklist to help you with your task:
1) The plan must be in the same format as the examples above.
2) Use the tags "<|plan_start|>...<|plan_end|>" around the plan.
3) The actions in the plan must be from the set of actions in the domain described above, that is, they must use the same name and the same number of parameters as one of the action schemas.
4) The plan must be valid, that is, each action must be applicable in the state it is applied, and the plan must end in a goal state.
"""
            chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}
            ]
            if with_plan:
                chat.append({"role": "assistant", "content": f"{config.TOKENS.PLAN_START}\n{task_components_in_nl['plan']}\n{config.TOKENS.PLAN_END}"})
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
def get_few_shot_examples(domain:str, available_task_types:set[Task.TYPE], few_shot: int, is_longer_plan: bool = False, random_seed:int=42) -> set[Task]:
    rng = np.random.RandomState(random_seed)
    try:
        possible_few_shot_examples = set()
        for task_type in available_task_types:
            possible_few_shot_examples.update(get_tasks(domain, type=task_type, is_longer_plan=is_longer_plan))
    except Exception as e:
        logger.error(f"Error getting possible CoT examples: {e}", exc_info=True)
        raise e

    few_shot_examples = set(
        rng.choice(
            list(possible_few_shot_examples),
            size=min(few_shot, len(possible_few_shot_examples)),
            replace=False
        )
    )

    if len(few_shot_examples) < few_shot:
        logger.warning(f"Requested {few_shot} few-shot examples, but only {len(few_shot_examples)} are available. Returning all available examples.")
    return few_shot_examples