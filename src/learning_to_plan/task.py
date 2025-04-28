import threading
import abc
import re
import json
import os
from typing import Optional
from enum import Enum

instance_pattern = re.compile(r"instance-(\d+)\.pddl$")
lock = threading.Lock()

class NaturalLanguagePlan:
    def __init__(self, plan:str, is_valid:bool=False, validated:bool=False):
        self._plan = plan
        self._is_valid = is_valid
        self._validated = validated
    
    def validate(self, is_valid:bool):
        self._is_valid = is_valid
        self._validated = True
    
    def to_json(self):
        return {
            "plan": self._plan,
            "is_valid": self._is_valid,
            "validated": self._validated
        }

    def from_json(self, json_obj):
        self._plan = json_obj.get("plan", None)
        self._is_valid = json_obj.get("is_valid", False)
        self._validated = json_obj.get("validated", False)

class Task(abc.ABC):
    class TaskType(Enum):
        TRAIN = "train"
        VALIDATION = "validation"
        TEST = "test"

    class TaskStatus(Enum):
        OK = "ok"
        ERROR = "error"

    def __init__(self, domain : str, domain_file_path : str, instance_file_path : str):
        self._domain :str = domain
        self._domain_file_path : str = domain_file_path
        self._instance_file_path : str = instance_file_path
        self._instance : str = instance_pattern.search(self._instance_file_path).group(0)
        self._is_longer_plan : bool = True if config.LONG_INSTANCES in self._instance_file_path else False
        self._status : Optional[Task.TaskStatus] = None
        self._error_message : Optional[str] = None
        self._plan : Optional[str] = None
        self._type : Optional[Task.TaskType] = None # training, validation, test | None
        self._model_generated_plans : Optional[dict[str, list[NaturalLanguagePlan]]] = None # Updated type hint

    def validate_plan(self, model_name : str, plan_idx : int, is_valid: bool):
        if self._model_generated_plans:
            if model_name in self._model_generated_plans:
                if len(self._model_generated_plans[model_name]) > plan_idx:
                    self._model_generated_plans[model_name][plan_idx].validate(is_valid)
                else:
                    raise IndexError(f"Plan index {plan_idx} out of range for model {model_name}.")
            else:
                raise KeyError(f"Model {model_name} not found in generated plans.")
        else:
            raise ValueError("No generated plans available to validate.")

    @abc.abstractmethod
    def convert_instance_into_natural_language(self, plan) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def convert_plan_into_natural_language(self, plan) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def build_prompt(self, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    def add_separator(self, prompt: str) -> str:
        if "## Plan." in prompt:
            return prompt
        else:
            return prompt + "\n## Plan.\n\n"
        
    def get_prompt(self, tokenizer_eos: str = "", with_plan: bool = True) -> str:
        try:
            prompt = self.build_prompt()
            prompt = self.add_separator(prompt)
            prompt += tokenizer_eos
            if with_plan and self._plan:
                prompt += self._plan
            return prompt
        except NotImplementedError as e:
            raise NotImplementedError(f"Method not implemented: {e}")
        except Exception as e:
            raise Exception(f"An error occurred while building the prompt: {e}")
        


    @property
    def _id(self):
        return f"{self._domain_file_path} - {self._instance_file_path}"

    def add_generated_plans(self, model_name: str, plans: list[str], overwrite: bool = True):
        if self._model_generated_plans is None:
            self._model_generated_plans = {}
        if model_name not in self._model_generated_plans or overwrite:
            self._model_generated_plans[model_name] = [] # Initialize/clear the list for the model
            if overwrite:
                config.log(f"Overwriting plans for model {model_name} in task {self._id}.")

        # Create NaturalLanguagePlan objects for the new plans
        new_nl_plans = [NaturalLanguagePlan(plan) for plan in plans]
        self._model_generated_plans[model_name].extend(new_nl_plans)
        config.log(f"Added {len(plans)} plans for model {model_name} in task {self._id}.")


    def to_json(self):
        # Serialize model_generated_plans
        serialized_plans = None
        if self._model_generated_plans:
            serialized_plans = {}
            for model_name, nl_plans_list in self._model_generated_plans.items():
                serialized_plans[model_name] = [nl_plan.to_json() for nl_plan in nl_plans_list]

        data = {
            "domain_file_path": self._domain_file_path,
            "instance_file_path": self._instance_file_path,
            "instance": self._instance,
            "status": self._status.value if self._status else None,
            "plan": self._plan,
            "error_message": self._error_message,
            "domain": self._domain,
            "is_longer_plan": self._is_longer_plan,
            "type": self._type.value if self._type else None,
            "model_generated_plans": serialized_plans # Use the serialized version
        }
        try:
             data['prompt'] = self.add_separator(self.build_prompt())
        except (NotImplementedError, AssertionError, Exception) as e:
            data['prompt'] = None
            config.log(f"Could not generate prompt for task {self._id}: {e}", level=config.logging.WARNING)

        return json.dumps(data, ensure_ascii=False)

    def __lt__(self, other):
        if not isinstance(other, Task):
            return NotImplemented

        if self._is_longer_plan != other._is_longer_plan:
            return not self._is_longer_plan

        self_match = instance_pattern.search(self._instance_file_path)
        other_match = instance_pattern.search(other._instance_file_path)
        if self_match and other_match:
            return int(self_match.group(1)) < int(other_match.group(1))
        else:
            return self._instance_file_path < other._instance_file_path


    def __str__(self):
        return f"{self._id} : {self._status}, {self._type}"

    def __hash__(self):
        return hash((self._domain_file_path, self._instance_file_path))

    def __eq__(self, other):
        if not isinstance(other, Task):
            return NotImplemented
        return self._instance_file_path == other._instance_file_path and self._domain_file_path == other._domain_file_path

    def from_json(self, json_obj):
        # Deserialize enums
        for field_name, enum_type in [("status", Task.TaskStatus), ("type", Task.TaskType)]:
            json_value = json_obj.get(field_name)
            if json_value is not None:
                if not isinstance(json_value, str):
                     msg = f"Expected string for {field_name}, but got {type(json_value)}"
                     config.log(msg, level=config.logging.ERROR)
                     continue
                try:
                    setattr(self, f"_{field_name}", enum_type(json_value.strip()))
                except (ValueError, KeyError):
                    msg = f"Invalid {field_name} value in JSON: '{json_value}'"
                    config.log(msg, level=config.logging.ERROR)
                    raise ValueError(msg)

        # Deserialize simple string fields
        for field_name in ["plan", "error_message"]:
            value = json_obj.get(field_name)
            if value is not None:
                 if not isinstance(value, str):
                    raise TypeError(f"{field_name} must be a string or null, but got {type(value)}")
                 setattr(self, f"_{field_name}", value)

        # Deserialize model_generated_plans
        plans_map_json = json_obj.get("model_generated_plans")
        if plans_map_json is not None:
            if not isinstance(plans_map_json, dict):
                raise TypeError(f"model_generated_plans must be a dictionary or null, but got {type(plans_map_json)}")

            self._model_generated_plans = {}
            for model_name, plans_list_json in plans_map_json.items():
                if not isinstance(model_name, str):
                    raise TypeError(f"Keys in model_generated_plans must be strings, but got {type(model_name)}")
                if not isinstance(plans_list_json, list):
                     raise TypeError(f"Values in model_generated_plans must be lists, but got {type(plans_list_json)} for key '{model_name}'")

                nl_plans_list = []
                for plan_json in plans_list_json:
                    if not isinstance(plan_json, dict):
                        raise TypeError(f"Elements in the plan list for model '{model_name}' must be dictionaries, but got {type(plan_json)}")
                    nl_plan = NaturalLanguagePlan(plan="") # Create an empty instance
                    nl_plan.from_json(plan_json) # Populate from JSON dict
                    nl_plans_list.append(nl_plan)

                self._model_generated_plans[model_name] = nl_plans_list


    def update_status(self, response):
        status = response.get("status", "error")
        if status == "ok":
            plan_text = response["result"]["output"]["sas_plan"]
            plain_text_plan = self.convert_plan_into_natural_language(plan_text) if plan_text else ""
            err_msg = ""
        else:
            plain_text_plan = ""
            err_msg = response.get("error", "Missing plan details or planning failed.")
        self._status = Task.TaskStatus(status) if status in [e.value for e in Task.TaskStatus] else None
        self._plan = plain_text_plan
        self._error_message = err_msg

    def read_instance(self):
        with lock and open(self._instance_file_path, "r", encoding='utf-8') as f:
            instance_content = f.read()
        return instance_content

    def read_domain(self):
        with lock and open(self._domain_file_path, "r", encoding='utf-8') as f:
            domain_content = f.read()
        return domain_content


class BlocksworldTask(Task):
    def __init__(self, domain_file_path, instance_file_path):
        super().__init__("blocksworld", domain_file_path, instance_file_path)

    def convert_instance_into_natural_language(self, pddl_text:str) -> str:
        obj_match = re.search(r"\(:objects\s+(.*?)\)", pddl_text, re.DOTALL)
        objects = obj_match.group(1).split() if obj_match else []
        objects_str = "blocks: " + ", ".join(objects) + "."
        init_match = re.search(r"\(:init\s+(.*?)\)", pddl_text, re.DOTALL)
        init_lines = init_match.group(1).split("\n") if init_match else []
        init_facts = []
        for line in init_lines:
            line = line.strip()
            if not line: continue
            fact = line.strip("()")
            tokens = fact.split()
            if tokens[0] == "handempty":
                init_facts.append("your hand is empty.")
            elif tokens[0] == "holding":
                init_facts.append(f"you are holding {tokens[1]}.")
            elif tokens[0] == "clear":
                init_facts.append(f"{tokens[1]} is clear.")
            elif tokens[0] == "ontable":
                init_facts.append(f"{tokens[1]} is on the table.")
            elif tokens[0] == "on":
                init_facts.append(f"{tokens[1]} is on {tokens[2]}.")
            else:
                init_facts.append(fact + ".")
        init_text = "initial state:\n" + "\n".join(init_facts)
        goal_match = re.search(r"\(:goal\s+\(and\s+(.*?)\)\s*\)", pddl_text, re.DOTALL)
        goal_lines = goal_match.group(1).split("\n") if goal_match else []
        goal_facts = []
        for line in goal_lines:
            line = line.strip()
            if not line: continue
            fact = line.strip("()")
            tokens = fact.split()
            if tokens[0] == "handempty":
                goal_facts.append("your hand is empty.")
            elif tokens[0] == "holding":
                goal_facts.append(f"you are holding {tokens[1]}.")
            elif tokens[0] == "clear":
                goal_facts.append(f"{tokens[1]} is clear.")
            elif tokens[0] == "ontable":
                goal_facts.append(f"{tokens[1]} is on the table.")
            elif tokens[0] == "on":
                goal_facts.append(f"{tokens[1]} is on {tokens[2]}.")
            else:
                goal_facts.append(fact + ".")
        goal_text = "goal state:\n" + "\n".join(goal_facts)
        return f"{objects_str}\n\n{init_text}\n\n{goal_text}"

    def convert_plan_into_natural_language(self, plan:str) -> str:
        """
            plan: str - actions in PDDL format, each action in a new line
        """
        actions = plan.split(";")[0].strip().split("\n")
        nl_plan = ""
        for action in actions:
            action = action.strip()
            if action.startswith("(") and action.endswith(")"):
                action = action[1:-1].strip()
                parts = action.split()
                if parts[0] == "unstack":
                    nl_plan += f"unstack {parts[1]} from {parts[2]};\n"
                elif parts[0] == "pick-up":
                    nl_plan += f"pick up {parts[1]};\n"
                elif parts[0] == "stack":
                    nl_plan += f"stack {parts[1]} on {parts[2]};\n"
                elif parts[0] == "put-down":
                    nl_plan += f"put down {parts[1]};\n"
                else:
                    raise ValueError(f"Unknown action: {action}")
        return nl_plan
    
    def convert_plan_into_pddl(self, plan:str) -> str:
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
        lines = plan.replace(";", "\n").strip().split("\n")

        for line in lines:
            nl_a = line.strip()
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

        return "\n".join(pddl_actions)

    def build_prompt(self, **kwargs):
        problem_description = self.convert_instance_into_natural_language(self.read_instance())
        prompt = (
                "# Goal.\n\n"
                "Use the available actions to transform the initial state into the goal state.\n\n"
                "# Output Format.\n\n"
                "Return a sequence of actions, one per line, in the order they should be applied.\n\n"
                "# Warnings.\n\n"
                "An action can only be applied if all its preconditions are true in the current state.\n"
                "When an action is applied, its effects update the current state by adding and removing facts.\n"
                "The goal is reached when all facts in the goal state are present in the current state.\n"
                "A valid plan must transform the initial state into the goal state using only applicable actions.\n"
                "Starting from the initial state, choose an applicable action, apply it, and repeat this process until the goal is reached.\n"
                "If no sequence of actions can reach the goal, return nothing.\n\n"
                "# Context.\n\n"
                "## Available actions.\n\n"
                "### Action: pick up block.\n"
                "preconditions:\n"
                "block is on the table.\n"
                "block is clear.\n"
                "your hand is empty.\n\n"
                "effects:\n"
                "you are holding block.\n"
                "your hand is not empty.\n"
                "block is not on the table.\n"
                "block is not clear.\n\n"
                "### Action: put down block.\n"
                "preconditions:\n"
                "you are holding block.\n\n"
                "effects:\n"
                "block is on the table.\n"
                "block is clear.\n"
                "your hand is empty.\n"
                "you are not holding block.\n\n"
                "### Action: stack block1 on block2.\n"
                "preconditions:\n"
                "you are holding block1.\n"
                "block2 is clear.\n\n"
                "effects:\n"
                "your hand is empty.\n"
                "block1 is clear.\n"
                "block2 is not clear.\n"
                "you are not holding block1.\n"
                "block1 is on block2.\n\n"
                "### Action: unstack block1 from block2.\n"
                "preconditions:\n"
                "block1 is clear.\n"
                "block1 is on block2.\n"
                "your hand is empty.\n\n"
                "effects:\n"
                "you are holding block1.\n"
                "your hand is not empty.\n"
                "block2 is clear.\n"
                "block1 is not clear.\n"
                "block1 is not on block2.\n\n"
                "## Instance.\n\n"
                + problem_description + "\n\n"
                "## Plan.\n\n"
            )
        return prompt

import learning_to_plan.config as config

def get_task_from_domain(domain, domain_file_path, instance_file_path):
    if domain == "blocksworld":
        task = BlocksworldTask(domain_file_path, instance_file_path)
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return task

def get_task_from_json(json_obj):
    domain = json_obj.get("domain", None)
    instance_file_path = json_obj.get("instance_file_path", None)
    domain_file_path = json_obj.get("domain_file_path", None)
    assert domain, "Domain is not specified in the JSON object."
    assert instance_file_path, "Instance file path is not specified in the JSON object."
    assert domain_file_path, "Domain file path is not specified in the JSON object."
    task = get_task_from_domain(
        domain,
        domain_file_path,
        instance_file_path
    )
    task.from_json(json_obj)
    return task

def get_tasks_from_jsonl(jsonl_file_path):
    tasks = set()
    with open(jsonl_file_path, "r", encoding='utf-8') as f:
        for line in f:
            try:
                json_obj = json.loads(line)
                task = get_task_from_json(json_obj)
                tasks.add(task)
            except json.JSONDecodeError as e:
                m = f"Error decoding JSONL {jsonl_file_path}: {e}"
                config.log(m, level=config.logging.ERROR)
                raise e
            except Exception as e:
                m = f"Error processing task from file {jsonl_file_path}: {e}"
                config.log(m, level=config.logging.ERROR)
                raise e
    return tasks

def save_tasks_to_jsonl(tasks:set[Task], jsonl_file_path:str):
    with open(jsonl_file_path, "w", encoding='utf-8') as f:
        for task in tasks:
            try:
                json_str = task.to_json() # Get the JSON string representation
                f.write(json_str + "\n") # Write the JSON string followed by a newline
            except Exception as e:
                m = f"Error saving task to file {jsonl_file_path}: {e}"
                config.log(m, level=config.logging.ERROR)
                raise e

from typing import Union, Set
from datasets import Dataset
def get_tasks_from_domain_directory(domain: str, number_of_problems_per_domain: Union[str, int] = "all") -> Set[Task]:
    """
    Get tasks from a domain directory.

    Args:
        domain: Domain name
        number_of_problems_per_domain: "all", "basic", "long", or a positive integer

    Returns:
        Set of Task objects
    """
    domain_file_path = os.path.join(config.RAW_DIR, domain, config.DOMAIN_FILE_NAME)

    # Determine which instance directories to include
    instance_dirs = []
    # Corrected variable name here
    if number_of_problems_per_domain in ("basic", "all") or isinstance(number_of_problems_per_domain, int):
        instance_dirs.append(os.path.join(config.RAW_DIR, domain, config.BASIC_INSTANCES))
    if number_of_problems_per_domain in ("long", "all") or isinstance(number_of_problems_per_domain, int):
        instance_dirs.append(os.path.join(config.RAW_DIR, domain, config.LONG_INSTANCES))

    # Collect tasks
    tasks = []
    for instance_dir in instance_dirs:
        if not os.path.exists(instance_dir):
            raise ValueError(f"Instance directory not found: {instance_dir}")

        # Get tasks from this directory using list comprehension
        tasks.extend([
            get_task_from_domain(domain, domain_file_path, os.path.join(instance_dir, file_name))
            for file_name in os.listdir(instance_dir)
            if instance_pattern.search(file_name)
        ])

    # Sort tasks
    tasks.sort()

    # Limit number of tasks if specified
    if isinstance(number_of_problems_per_domain, int):
        if number_of_problems_per_domain <= 0:
            raise ValueError(f"Number of problems per domain must be positive")
        elif number_of_problems_per_domain < len(tasks):
            tasks = tasks[:number_of_problems_per_domain]

    return set(tasks)
 

def convert_tasks_into_dataset(tasks:set[Task], tokenizer_eos:str="", with_plan:bool=True) -> Dataset:
    """
    Converts a set of Task objects into a datasets.Dataset object.

    Args:
        tasks: A set of Task objects.
        tokenizer_eos: End-of-sequence token to append to the prompt.
        with_plan: Whether to include the plan in the prompt.

    Returns:
        A datasets.Dataset object containing the prompts.
    """
    dataset_list = []
    for t in tasks:
        try:
            p = t.get_prompt(tokenizer_eos=tokenizer_eos, with_plan=with_plan)
            dataset_list.append({
                    "text": p,
                })
        except Exception as e:
            m = f"Error converting task {t._id} into dataset format: {e}"
            config.log(m, level=config.logging.ERROR)
            # Depending on requirements, you might want to skip the task or raise the exception
            raise e
    # Create a Dataset object from the list of dictionaries
    return Dataset.from_list(dataset_list)