from __future__ import annotations
import threading
from learning_to_plan import config
from typing import Optional
from enum import Enum
from learning_to_plan.data import base

logger = config.get_logger(__name__)
lock = threading.Lock()

def _read_pddl_file(pddl_file : str, ignore_comments: bool = True) -> str:
    """
    Reads the PDDL file and returns its content.
    :param ignore_comments: If True, ignores lines that start with ";".
    :return: The content of the PDDL file.
    """
    with lock and open(pddl_file, "r", encoding='utf-8') as f:
        content = f.read()
    
    if ignore_comments:
        # Filter out lines that start with ";"
        filtered_lines = [line for line in content.splitlines() if not line.strip().startswith(";")]
        return "\n".join(filtered_lines)
    
    return content

class Task(base.Data):
    NEXT_ID: int = 0
    SETTED_ID_VARIABLES: bool = False
    FIELD_NAMES =[
            "id", 
            "domain", 
            "domain_file_path", 
            "instance_file_path", 
            "type",
            "paas_status", 
            "pddl_plan", 
            "purpose", 
        ]
    
    class purpose(Enum):
        TRAIN = "train"
        VALIDATION = "validation"
        TEST = "test"
    
    class TYPE(Enum):
        INDISTRIBUTION = "indistribution"
        OUTOFDISTRIBUTION = "outofdistribution"
        UNSEEN = "unseen"
        OBFUSCATED = "obfuscated"

    def __init__(
            self, 
            domain : str, 
            domain_file_path : str, 
            instance_file_path : str,
            type: Task.TYPE | str,
            id: Optional[int]=None, 
            paas_status: Optional[config.STATUS | str] = config.STATUS.ERROR,
            purpose: Optional[Task.purpose | str] = None,
            pddl_plan: Optional[str] = None):
        super().__init__(id)
        self.domain :str = domain
        self.domain_file_path : str = domain_file_path
        self.instance_file_path : str = instance_file_path
        self.pddl_plan: Optional[str] = pddl_plan

        self.type: Task.TYPE = config.get_enum_value(type, Task.TYPE, "type")
        assert self.type, "Task type must not be None."

        self.paas_status = config.get_enum_value(paas_status, config.STATUS, "paas_status")
        
        self.purpose = config.get_enum_value(purpose, Task.purpose, "purpose")

    def __str__(self):
        return f"Task(id={self.id}, domain={self.domain}, instance={self.instance_file_path}, type={self.type}, purpose={self.purpose}, paas_status={self.paas_status})"

    def read_instance(self):
        return _read_pddl_file(self.instance_file_path)

    def read_domain(self):
        return _read_pddl_file(self.domain_file_path)

    def get_plan(self) -> str:
        if self.pddl_plan is None:
            raise ValueError(f"The task {self.id} does not have a PDDL plan.")
        p = "\n".join(line for line in self.pddl_plan.splitlines() if line.startswith("(") and line.endswith(")"))
        if not p:
            raise ValueError(f"The PDDL plan for task {self.id} is empty or does not contain valid actions.")
        return p

    @classmethod
    def column_def(cls):
        return [
            "id INTEGER PRIMARY KEY",
            "domain TEXT NOT NULL",
            "domain_file_path TEXT NOT NULL",
            "instance_file_path TEXT NOT NULL",
            "type TEXT NOT NULL",
            "paas_status TEXT",
            "pddl_plan TEXT",
            "purpose TEXT NOT NULL",
        ]
    @classmethod
    def column_constraints(cls):
        return [
            "CONSTRAINT tup_dm_in UNIQUE(domain, instance_file_path)",  # Ensure unique domain-instance pairs
        ]