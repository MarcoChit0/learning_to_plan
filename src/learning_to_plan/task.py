from __future__ import annotations
from copy import deepcopy
import threading
import abc
import re
import json
import os
import learning_to_plan.config as config
from typing import Optional
from enum import Enum

logger = config.get_logger(__name__)

instance_pattern = re.compile(r"instance-(\d+)\.pddl$")
lock = threading.Lock()

class Task(abc.ABC):
    class Type(Enum):
        TRAIN = "train"
        VALIDATION = "validation"
        TEST = "test"

    class PlanningAsAServiceStatus(Enum):
        OK = "ok"
        ERROR = "error"
    
    class LanguageConverter(abc.ABC):
        @abc.abstractmethod
        def fact_to_natural_language(self, fact: str) -> str:
            raise NotImplementedError("Subclasses must implement this method.")

        @abc.abstractmethod
        def pddl_instance_to_natural_language(self, pddl_instance: str) -> str:
            raise NotImplementedError("Subclasses must implement this method.")

        @abc.abstractmethod
        def pddl_plan_to_natural_language(self, pddl_plan: str) -> str:
            raise NotImplementedError("Subclasses must implement this method.")
        
        @abc.abstractmethod
        def natural_language_plan_to_pddl(self, nl_plan: str) -> str:
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
        self._paas_status : Optional[Task.PlanningAsAServiceStatus] = None # ok, error
        self._pddl_plan : Optional[str] = None
        self._type : Optional[Task.Type] = None # training, validation, test | None

        if self._domain == "blocksworld":
            self._converter : Task.LanguageConverter = BlocksworldLanguageConverter()
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

    def get_prompt(self, with_plan: bool = True,  cot_examples:set[Task] = set(), as_chat_template:bool = False) -> str:
        try:
            is_cot = len(cot_examples) > 0
            domain_description = self._converter._domain_description_in_natural_language
            instance_nl = self._converter.pddl_instance_to_natural_language(pddl_instance=self.read_instance())
            
            # Standard prompt format
            if not as_chat_template:
                prompt = ""
                prompt += domain_description
                prompt += instance_nl
                if with_plan and self._pddl_plan:
                    prompt += config.START_OF_PLAN_TOKEN + "\n"
                    prompt += self._converter.pddl_plan_to_natural_language(pddl_plan=self._pddl_plan)
                    prompt += config.END_OF_PLAN_TOKEN
                return prompt
            
            # Chat template format as a list of dictionaries
            else:
                messages = [{"role": "user", "content": f"{domain_description}\n\n{instance_nl}"}]
                
                if with_plan and self._pddl_plan:
                    # Include plan as assistant's response
                    plan_nl = self._converter.pddl_plan_to_natural_language(pddl_plan=self._pddl_plan)
                    messages.append({"role": "assistant", "content": plan_nl})
                
                return messages
                    
        except Exception as e:
            raise Exception(f"An error occurred while building the prompt: {e}")

    def get_conversation(self, with_plan: bool = True) -> list[dict[str, str]]:
        conversation = []
        prompt = self._converter._domain_description_in_natural_language + self._converter.pddl_instance_to_natural_language(pddl_instance=self.read_instance())
        conversation.append({"role": "user", "content": prompt})
        if with_plan and self._pddl_plan:
            conversation.append({"role": "assistant", "content": config.START_OF_PLAN_TOKEN + "\n" + self._converter.pddl_plan_to_natural_language(pddl_plan=self._pddl_plan) + config.END_OF_PLAN_TOKEN})
        return conversation

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
            ("_paas_status", "paas_status", Task.PlanningAsAServiceStatus),
            ("_type", "type", Task.Type)
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
        
        self._paas_status = Task.PlanningAsAServiceStatus(status) if status in [e.value for e in Task.PlanningAsAServiceStatus] else None
        self._pddl_plan = plan

    def read_instance(self):
        with lock and open(self._instance_file_path, "r", encoding='utf-8') as f:
            instance_content = f.read()
        return instance_content

    def read_domain(self):
        with lock and open(self._domain_file_path, "r", encoding='utf-8') as f:
            domain_content = f.read()
        return domain_content


class BlocksworldLanguageConverter(Task.LanguageConverter):
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

        def fact_to_natural_language(self, pddl_fact:str) -> str:
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

        def pddl_instance_to_natural_language(self, pddl_instance: str) -> str:
            """
            Converts a PDDL instance into natural language format.
            (Implementation with corrected regex for section extraction)
            """
            # --- Helper function to extract section content ---
            def get_section_content(section_name: str, text: str) -> str | None:
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
            # --- End of helper function ---

            objects_match = re.search(r"\(:objects\s+(.*?)\s*\)", pddl_instance, re.DOTALL | re.IGNORECASE)
            objects_str = objects_match.group(1).strip() if objects_match else None
            init_content = get_section_content(":init", pddl_instance)
            goal_content = get_section_content(":goal", pddl_instance)

            if objects_str is None:
                raise ValueError("Invalid PDDL instance format: Could not find :objects section content.")
            if init_content is None:
                raise ValueError("Invalid PDDL instance format: Could not find :init section content.")
            if goal_content is None:
                raise ValueError("Invalid PDDL instance format: Could not find :goal section content.")

            fact_pattern = r"\(\s*[\w-]+\s*[\w\s-]*\)"

            objects = objects_str.split()
            init_facts_raw = re.findall(fact_pattern, init_content)
            goal_facts_raw = []
            and_match = re.search(r"^\(and\s+(.*)\s*\)$", goal_content, re.DOTALL | re.IGNORECASE)
            if and_match:
                and_content = and_match.group(1).strip()
                goal_facts_raw = re.findall(fact_pattern, and_content)
            else:
                stripped_goal_content = goal_content.strip()
                if stripped_goal_content and re.fullmatch(fact_pattern, stripped_goal_content):
                    goal_facts_raw = [stripped_goal_content]
                elif not stripped_goal_content:
                    pass # Goal is empty

            # --- Fact Cleaning (Remains the same) ---
            init_facts_pddl = [re.sub(r"\s+", " ", fact).strip() for fact in init_facts_raw if fact.strip()]
            goal_facts_pddl = [re.sub(r"\s+", " ", fact).strip() for fact in goal_facts_raw if fact.strip()]

            # --- Validation and NL Conversion (Remains the same) ---
            if not objects:
                raise ValueError("Empty objects list in PDDL instance.")

            for obj in objects:
                if not hasattr(self, '_color_map') or obj not in self._color_map:
                    raise ValueError(f"Unknown object '{obj}' found in :objects section or missing in color map.")

            init_facts_nl = []
            for fact in init_facts_pddl:
                try:
                    init_facts_nl.append(self.fact_to_natural_language(fact))
                except ValueError as e:
                    logger.warning(f"Skipping initial fact due to conversion error: {e} (Fact: '{fact}')")
            init_facts_str = "As initial conditions I have that: " + (", ".join(init_facts_nl) if init_facts_nl else "nothing specific")

            goal_facts_nl = []
            for fact in goal_facts_pddl:
                try:
                    goal_facts_nl.append(self.fact_to_natural_language(fact))
                except ValueError as e:
                    logger.warning(f"Skipping goal fact due to conversion error: {e} (Fact: '{fact}')")
            goal_facts_str = "My goal is to have that: " + (", ".join(goal_facts_nl) if goal_facts_nl else "nothing specific")

            nl_output = f"{init_facts_str}.\n{goal_facts_str}."
            return nl_output

        def pddl_plan_to_natural_language(self, pddl_plan:str) -> str:
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

        def natural_language_plan_to_pddl(self, nl_plan:str) -> str:
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
            # Normalize line endings and split, handling potential semicolons
            lines = nl_plan.replace(";", "\n").strip().split("\n")
            for line in lines:
                nl_a = line.lower().replace("block", "").strip()
                if not nl_a: # Skip empty lines
                    continue

                action_found = False
                if nl_a.startswith("unstack"):
                    match = re.search(r"unstack\s+(\w+)\s+from\s+(\w+)", nl_a)
                    if match:
                        pddl_actions.append(f"(unstack {match.group(1)} {match.group(2)})")
                        action_found = True
                elif nl_a.startswith("pick up"):
                    match = re.search(r"pick up\s+(\w+)", nl_a)
                    if match:
                        pddl_actions.append(f"(pick-up {match.group(1)})")
                        action_found = True
                elif nl_a.startswith("stack"):
                    match = re.search(r"stack\s+(\w+)\s+on\s+(\w+)", nl_a)
                    if match:
                        pddl_actions.append(f"(stack {match.group(1)} {match.group(2)})")
                        action_found = True
                elif nl_a.startswith("put down"):
                    match = re.search(r"put down\s+(\w+)", nl_a)
                    if match:
                        pddl_actions.append(f"(put-down {match.group(1)})")
                        action_found = True

                if not action_found:
                    # Raise error only if the line was not empty and didn't match any known pattern
                    raise ValueError(f"Unknown or malformed action: '{nl_a}'")

            return "\n".join(pddl_actions) + "\n" if pddl_actions else ""

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
    

def get_tasks(filter_by_domain: Optional[str] = None,  filter_by_type: Optional[Task.Type] = None, is_longer_plan:Optional[bool] = None, number_of_instances: Optional[int] = None) -> set[Task]:
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