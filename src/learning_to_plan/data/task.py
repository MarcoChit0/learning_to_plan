from __future__ import annotations
import threading
from learning_to_plan import config
from typing import Optional
from enum import Enum
from learning_to_plan.data import database_manager
from learning_to_plan.data import base

logger = config.get_logger(__name__)
lock = threading.Lock()

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

    # TODO: CHECK WHETHER THIS WHAY DOES NOT INTRODUCES ID ERRORS
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
        with lock and open(self.instance_file_path, "r", encoding='utf-8') as f:
            instance_content = f.read()
        return instance_content

    def read_domain(self):
        with lock and open(self.domain_file_path, "r", encoding='utf-8') as f:
            domain_content = f.read()
        return domain_content

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

task_database: Optional[database_manager.DatabaseManager] = None

def initialize_db():
    global task_database
    if task_database is None:
        task_database = database_manager.DatabaseManager(
            table_name="task",
            data_cls=Task,
            filters={
                "filter_by_domain": "domain = ?",
                "filter_by_purpose": "purpose = ?",
                "filter_by_type": "type = ?",
                "filter_by_paas_status": "paas_status = ?",
            }
        )
    _ = task_database.get()