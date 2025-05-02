from __future__ import annotations
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

class PlanManager:
    # TODO: add on the future the possibility of distinguishing between multiple finetuned versions of the same model
    class PromptType(Enum):
        IO = "io"
        COT = "cot"

    class Plan:
        def __init__(self, plan: str, is_valid: Optional[bool] = None):
            self._plan : str = plan
            # None means that the plan is not validated yet
            self._is_valid : Optional[bool] = is_valid

        def to_json(self):
            return {
                "plan": self._plan,
                "is_valid": self._is_valid,
            }

        def from_json(self, json_obj):
            self._plan = json_obj.get("plan", None)
            self._is_valid = json_obj.get("is_valid", None)
            assert self._plan is not None, "NaturalLanguagePlan must have a plan."

    def __init__(self, model_name : Optional[str] = None):
        self._model_name = model_name
        self._prompt_to_plan_mapping : dict[PlanManager.PromptType, PlanManager.Plan] = {}

    def __hash__(self):
        return hash(self._model_name)

    def __eq__(self, other):
        if not isinstance(other, PlanManager):
            return NotImplemented
        return self._model_name == other._model_name

    def add_plan(self, prompt_type: PlanManager.PromptType, plan: str):
        if prompt_type not in PlanManager.PromptType:
            raise ValueError(f"Invalid prompt type: {prompt_type}.")
        if prompt_type in self._prompt_to_plan_mapping:
            # Changed config.log to logger.warning
            logger.warning(f"Prompt type {prompt_type} already exists for model {self._model_name}. Overwriting it.")
        else:
            # Changed config.log to logger.info
            logger.info(f"Adding plan for model {self._model_name} with prompt type {prompt_type}.")
        
        self._prompt_to_plan_mapping.update({prompt_type: PlanManager.Plan(plan=plan)})

    def validate(self, prompt_type: PlanManager.PromptType , is_valid: bool):
        if prompt_type not in self._prompt_to_plan_mapping:
            raise ValueError(f"Prompt type {prompt_type} not found for model {self._model_name}.")
        plan = self._prompt_to_plan_mapping[prompt_type]
        plan._is_valid = is_valid
        self._prompt_to_plan_mapping.update({prompt_type: plan})
        # Changed config.log to logger.info
        logger.info(f"Plan for model {self._model_name} with prompt type {prompt_type} is valid: {is_valid}.")

    def to_json(self):
        serialized_plans = {}
        for prompt_type, plan in self._prompt_to_plan_mapping.items():
            serialized_plans[prompt_type.value] = plan.to_json()
        return {
            "model_name": self._model_name,
            "plans": serialized_plans
        }

    def from_json(self, json_obj):
        '''
        {
            "model_name": "model_name",
            "plans": {
                "io": {
                    "plan": "plan",
                    "is_valid": false
                },
                "cot": {
                    "plan": "plan",
                    "is_valid": true
                }
            },
        }
        '''

        self._model_name = json_obj.get("model_name", None)
        if self._model_name is None:
            raise ValueError("Model name is required.")

        plans_json = json_obj.get("plans", {})
        for prompt_type_str, plan_json in plans_json.items():
            try:
                prompt_type = PlanManager.PromptType(prompt_type_str)
                plan = PlanManager.Plan(plan="")
                plan.from_json(plan_json)
                self._prompt_to_plan_mapping[prompt_type] = plan
            except ValueError as e:
                # Changed config.log to logger.error
                logger.error(f"Invalid prompt type in JSON: {e}")
                raise e



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
        # Assuming config.LONG_INSTANCES is defined elsewhere or needs replacement
        # For now, keeping it as is, but it might need adjustment depending on where LONG_INSTANCES comes from.
        self._is_longer_plan : bool = True if config.LONG_INSTANCES in self._instance_file_path else False
        self._status : Optional[Task.TaskStatus] = None
        self._error_message : Optional[str] = None
        self._plan : Optional[str] = None
        self._type : Optional[Task.TaskType] = None # training, validation, test | None
        self._plan_managers : set[PlanManager] = set() # Updated type

    @abc.abstractmethod
    def convert_pddl_instance_to_natural_language(self, pddl_instance) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def convert_pddl_plan_to_natural_language(self, pddl_plan) -> str:
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abc.abstractmethod
    def convert_natural_language_plan_to_pddl(self, nl_plan) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    @abc.abstractmethod
    def _domain_description_in_natural_language(self) -> str:
        raise NotImplementedError("Subclasses must implement this property.")

    def get_prompt(self, eos_token: Optional[str] = None, with_plan: bool = True,  cot_examples:set[Task] = set()) -> str:
        try:
            is_cot = len(cot_examples) > 0
            prompt = ""
            prompt += self._domain_description_in_natural_language
            if is_cot:
                assert len(cot_examples) > 0, "is_cot is True but no examples provided."
                prompt += "## Examples.\n\n"
                for i, example in enumerate(cot_examples):
                    # assert the subclasses are the same
                    assert type(example) == type(self), f"Example task {example._id} is not of the same type as the current task {self._id}."
                    prompt += f"### Example {i+1}/{len(cot_examples)}.\n\n"
                    prompt += f"#### Example {i+1} Instance.\n\n"
                    prompt += example.convert_pddl_instance_to_natural_language(example.read_instance())
                    prompt += f"#### Example {i+1} Plan.\n\n"
                    if example._plan:
                        prompt += example._plan
                    else:
                        raise ValueError(f"Example {i+1} -- {example._id} -- does not have a plan.")
                    prompt += "\n\n"
            prompt += "## Instance.\n\n"
            prompt += self.convert_pddl_instance_to_natural_language(self.read_instance())
            prompt += "## Plan.\n\n"
            if eos_token:
                prompt += eos_token
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

    def add_plan(self, model_name: str, prompt_type: PlanManager.PromptType, plan : str):
        plan_manager = next((pm for pm in self._plan_managers if pm._model_name == model_name), None)
        if not plan_manager:
            plan_manager = PlanManager(model_name)
            # Changed config.log to logger.info
            logger.info(f"Plan Manager for model {model_name} not found... Creating one for prompt type {prompt_type}.")
            self._plan_managers.add(plan_manager) # Ensure the new manager is added
        plan_manager.add_plan(prompt_type, plan)
        # Changed config.log to logger.info
        logger.info(f"Plan added to Plan Manager of model {model_name} with prompt type {prompt_type}.")

    def to_json(self):
        try:
            # Serialize plan_managers
            # It is stored on the json as a list of dictionaries
            # In the class plan_managers is a set of PlanManager
            plan_manager_list_json = [plan_manager.to_json() for plan_manager in self._plan_managers]
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
                "plan_managers": plan_manager_list_json,
            }
        except (NotImplementedError, AssertionError, Exception) as e:
            # Changed config.log to logger.error
            logger.error(f"Error generating prompt for task {self._id}: {e}")
            raise e

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
                     # Changed config.log to logger.error
                     logger.error(msg)
                     continue
                try:
                    setattr(self, f"_{field_name}", enum_type(json_value.strip()))
                except (ValueError, KeyError):
                    msg = f"Invalid {field_name} value in JSON: '{json_value}'"
                    # Changed config.log to logger.error
                    logger.error(msg)
                    raise ValueError(msg)

        # Deserialize simple string fields
        for field_name in ["plan", "error_message"]:
            value = json_obj.get(field_name)
            if value is not None:
                if not isinstance(value, str):
                   raise TypeError(f"{field_name} must be a string or null, but got {type(value)}")
                setattr(self, f"_{field_name}", value)

        # Deserialize plan_managers
        # It is stored on the json as a list of dictionaries
        # In the class plan_managers is a set of PlanManager
        plan_manager_list_json = json_obj.get("plan_managers")
        self._plan_managers = set()
        if plan_manager_list_json is not None:
            if not isinstance(plan_manager_list_json, list):
                raise TypeError(f"plan_managers must be a list or null, but got {type(plan_manager_list_json)}")

            for plan_manager_json in plan_manager_list_json:
                if not isinstance(plan_manager_json, dict):
                    raise TypeError(f"Elements in plan_managers must be dictionaries, but got {type(plan_manager_json)}")

                try:
                    plan_manager = PlanManager()
                    plan_manager.from_json(plan_manager_json)
                    self._plan_managers.add(plan_manager)
                except Exception as e:
                    # Changed config.log to logger.error
                    logger.error(f"Error deserializing PlanManager: {e}")
                    raise e


    def update_status(self, response):
        status = response.get("status", "error")
        if status == "ok":
            plan_text = response["result"]["output"]["sas_plan"]
            plain_text_plan = self.convert_pddl_plan_to_natural_language(plan_text) if plan_text else ""
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

    def fact_to_natural_language(self, fact:str) -> str:
        """
            fact : str - a line of PDDL representing a fact
        """
        fact = fact.strip()
        if not fact: return ""
        fact = fact.strip("()")
        tokens = fact.split()
        if tokens[0] == "handempty":
            return "your hand is empty."
        elif tokens[0] == "holding":
            return f"you are holding {tokens[1]}."
        elif tokens[0] == "clear":
            return f"{tokens[1]} is clear."
        elif tokens[0] == "ontable":
            return f"{tokens[1]} is on the table."
        elif tokens[0] == "on":
            return f"{tokens[1]} is on {tokens[2]}."
        else:
            raise ValueError(f"Unknown fact: {fact}")


    def convert_pddl_instance_to_natural_language(self, instance:str):
        """
            Example:
            instance: str - PDDL instance

            (define (problem BW-rand-6)
            (:domain blocksworld-4ops)
            (:objects a b c d e f )
            (:init
            (handempty)
            (ontable a)
            (on b d)
            (on c a)
            (on d c)
            (on e f)
            (on f b)
            (clear e)
            )
            (:goal
            (and
            (on a b)
            (on d f)
            (on e d)
            (on f c))
            )
            )

            should be converted to:

            blocks: a, b, c, d, e, f.

            initial state:
            your hand is empty.
            a is on the table.
            b is on d.
            c is on a.
            d is on c.
            e is on f.
            f is on b.
            e is clear.

            goal state:
            a is on b.
            d is on f.
            e is on d.
            f is on c.
        """
        # Use regex to extract the relevant parts of the PDDL instance

        objects_match = re.search(r"\(:objects\s+(.+?)\s+\)", instance, re.DOTALL)
        init_match = re.search(r"\(:init\s+(.+?)\s+\)", instance, re.DOTALL)
        goal_match = re.search(r"\(:goal\s+\(and\s+(.*?)\)\s*\)", instance, re.DOTALL)

        if not (objects_match and init_match and goal_match):
            raise ValueError("Invalid PDDL instance format.")

        objects = objects_match.group(1).strip().split()
        init_facts = init_match.group(1).strip().split("\n")
        goal_facts = goal_match.group(1).strip().split("\n")

        if objects == [] or init_facts == [] or goal_facts == []:
            raise ValueError("Empty objects, init, or goal in PDDL instance.")

        # Process objects
        objects = [obj.strip() for obj in objects if obj.strip()]
        objects_str = "blocks: " + ", ".join(objects) + "."

        # Process initial state
        init_facts = [self.fact_to_natural_language(fact) for fact in init_facts if fact.strip()]
        init_facts_str = "initial state:\n" + "\n".join(init_facts)

        # Process goal state
        goal_facts = [self.fact_to_natural_language(fact) for fact in goal_facts if fact.strip()]
        goal_facts_str = "goal state:\n" + "\n".join(goal_facts)

        return f"{objects_str}\n\n{init_facts_str}\n\n{goal_facts_str}\n\n"



    def convert_pddl_plan_to_natural_language(self, plan:str) -> str:
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

    def convert_natural_language_plan_to_pddl(self, plan:str) -> str:
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
        return (
            "# Goal.\n\n"
            "Using only the available actions described below, create a plan (sequence of actions) that transforms the initial state into the goal state.\n\n"
            "# Output Format.\n\n"
            "Your response must contain *only* the plan.\n"
            "List each action on a new line.\n"
            "Use the exact action format shown in the examples.\n"
            "Do not add any explanations, comments, or introductory text.\n\n"
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
        )

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
                # Changed config.log to logger.error
                logger.error(m)
                raise e
            except Exception as e:
                m = f"Error processing task from file {jsonl_file_path}: {e}"
                # Changed config.log to logger.error
                logger.error(m)
                raise e
    return tasks

def save_tasks_to_jsonl(tasks:set[Task], jsonl_file_path:str):
    with open(jsonl_file_path, "w", encoding='utf-8') as f:
        for task in sorted(tasks):
            try:
                json_str = task.to_json() # Get the JSON string representation
                f.write(json_str + "\n") # Write the JSON string followed by a newline
            except Exception as e:
                m = f"Error saving task to file {jsonl_file_path}: {e}"
                # Changed config.log to logger.error
                logger.error(m)
                raise e

from typing import Union, Set
# Removed unused import 'Dataset'
# from datasets import Dataset
def get_tasks_from_domain_directory(domain: str, number_of_problems_per_domain: Union[str, int] = "all") -> Set[Task]:
    """
    Get tasks from a domain directory.

    Args:
        domain: Domain name
        number_of_problems_per_domain: "all", "basic", "long", or a positive integer

    Returns:
        Set of Task objects
    """
    # Assuming config.RAW_DIR, config.DOMAIN_FILE_NAME, config.BASIC_INSTANCES, config.LONG_INSTANCES are defined
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