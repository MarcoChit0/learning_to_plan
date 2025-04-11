import threading
import abc
import re
import json
import os

instance_pattern = re.compile(r"instance-(\d+)\.pddl$")
lock = threading.Lock()

class Task(abc.ABC): 
    def __init__(self, domain, domain_file_path, instance_file_path):
        self._domain = domain
        self._domain_file_path = domain_file_path 
        self._instance_file_path = instance_file_path
        self._instance = instance_pattern.search(self._instance_file_path).group(0)
        self._status = None
        self._error_message = None
        self._plan = None
    
    @abc.abstractmethod
    def convert_instance_into_natural_language(self, plan) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def convert_plan_into_natural_language(self, plan) -> str:
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abc.abstractmethod
    def build_prompt(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def __lt__(self, other):
        if not isinstance(other, Task):
            return NotImplemented
        self_match = instance_pattern.search(self._instance_file_path)
        other_match = instance_pattern.search(other._instance_file_path)
        if self_match and other_match:
            return int(self_match.group(1)) < int(other_match.group(1))
        else:
            raise ValueError("Invalid instance file path format.")


    def read_from_dict(self, row):
        if row.get("instance", None) == self._instance:
            self._status = row["status"]
            self._plan = row["plan"]
            self._error_message = row["error"]

    def write_to_dict(self):
        return {
            "domain_file_path": self._domain_file_path,
            "instance_file_path": self._instance_file_path,
            "instance": self._instance,
            "status": self._status,
            "plan": self._plan,
            "error": self._error_message,
            "domain": self._domain,
        }

    def update_status(self, response):
        status = response.get("status", "error")
        if status == "ok":
            plan_text = response["result"]["output"]["sas_plan"]
            plain_text_plan = self.convert_plan_into_natural_language(plan_text) if plan_text else ""
            err_msg = ""
        else:
            plain_text_plan = ""
            err_msg = response.get("error", "Missing plan details or planning failed.")
        self._status = status
        self._plan = plain_text_plan
        self._error_message = err_msg

    def read_instance(self):
        with lock and open(self._instance_file_path, "r") as f:
            instance_content = f.read()
        return instance_content

    def read_domain(self):
        with lock and open(self._domain_file_path, "r") as f:
            domain_content = f.read()
        return domain_content

def get_task_from_domain(domain, domain_file_path, instance_file_path):
    if domain == "blocksworld":
        task = BlocksworldTask(domain_file_path, instance_file_path)
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return task

def get_task_from_csv(row):
    domain = row.get("domain", None)
    instance_file_path = row.get("instance_file_path", None)
    domain_file_path = row.get("domain_file_path", None)
    assert domain, "Domain is not specified in the CSV row."
    assert instance_file_path, "Instance file path is not specified in the CSV row."
    assert domain_file_path, "Domain file path is not specified in the CSV row."
    task = get_task_from_domain(
        domain,
        domain_file_path,
        instance_file_path
    )
    task.read_from_dict(row)
    return task

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

    def convert_plan_into_natural_language(self, plan) -> str:
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
    
    def build_prompt(self, **kwargs):
        assert type(self._plan) == str, "Plan should be a string."
        problem_description = self.convert_instance_into_natural_language(self.read_instance())
        data = {
            "prompt": (
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
                + self._plan
            )
        }
        return json.dumps(data, ensure_ascii=False)

from learning_to_plan.config import *

def get_tasks_from_domain_directory(domain, number_of_problems_per_domain=None):
    tasks = {}
    instance_directory = os.path.join(RAW_DIR, domain, INSTANCES_SUBDIRECTORY)
    domain_file_path = os.path.join(RAW_DIR, domain, DOMAIN_FILE_NAME)
    for file_name in os.listdir(instance_directory):
        if instance_pattern.search(file_name).group(0):
            instance_file_path = os.path.join(instance_directory, file_name)
            task = get_task_from_domain(domain, domain_file_path, instance_file_path)
            tasks[task._instance] = task
    if number_of_problems_per_domain:
        tasks = {k: v for k, v in sorted(tasks.items(), key=lambda item: item[1])[:number_of_problems_per_domain]}
    return tasks