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
from learning_to_plan import database

logger = config.get_logger(__name__)

INSTANCE_PATTERN = re.compile(r"instance-(\d+)\.pddl$")
lock = threading.Lock()

class Task(database.Data):
    SEEN_IDS: set[int] = set()
    AVAILABLE_ID_POOL: set[int] = set()
    NEXT_ID: int = 0
    class TYPE(Enum):
        TRAIN = "train"
        VALIDATION = "validation"
        TEST = "test"

    class PAAS_STATUS(Enum):
        OK = "ok"
        ERROR = "error"
    
    class LANDMARK_GRAPH_STATUS(Enum):
        OK = "ok"
        ERROR = "error"

    # TODO: CHECK WHETHER THIS WHAY DOES NOT INTRODUCES ID ERRORS
    def __init__(
            self, 
            domain : str, 
            domain_file_path : str, 
            instance_file_path : str,
            id: Optional[int]=None, 
            is_longer_plan: bool = False,
            paas_status: Optional[Task.PAAS_STATUS | str] = None,
            pddl_plan: Optional[str] = None,
            type: Optional[Task.TYPE | str] = None,
            landmark_graph: Optional[str] = None,
            landmark_graph_status: Optional[Task.LANDMARK_GRAPH_STATUS | str] = None):
        super().__init__(id, field_names=[
            "id", 
            "domain", 
            "domain_file_path", 
            "instance_file_path", 
            "is_longer_plan", 
            "paas_status", 
            "pddl_plan", 
            "type", 
            "landmark_graph", 
            "landmark_graph_status"
        ])
        self.domain :str = domain
        self.domain_file_path : str = domain_file_path
        self.instance_file_path : str = instance_file_path
        if is_longer_plan is None:
            self.is_longer_plan : bool = True if config.LONG_INSTANCES in self.instance_file_path else False
        else:
            self.is_longer_plan = is_longer_plan
        self.pddl_plan: Optional[str] = pddl_plan
        self.landmark_graph: Optional[str] = landmark_graph

        self.paas_status = self._get_enum_value(paas_status, Task.PAAS_STATUS, "paas_status")
        self.type = self._get_enum_value(type, Task.TYPE, "type")
        self.landmark_graph_status = self._get_enum_value(landmark_graph_status, Task.LANDMARK_GRAPH_STATUS, "landmark_graph_status")

    def __str__(self):
        long_part = ", long" if self.is_longer_plan else ""
        type_part = f": {self.type.value}" if self.type else ""
        return f"Task {self.domain}, {self.id}{long_part}{type_part}"

    def read_instance(self):
        with lock and open(self.instance_file_path, "r", encoding='utf-8') as f:
            instance_content = f.read()
        return instance_content

    def read_domain(self):
        with lock and open(self.domain_file_path, "r", encoding='utf-8') as f:
            domain_content = f.read()
        return domain_content

    def storage_datatype(self):
        return {
            "id": "INTEGER PRIMARY KEY",
            "domain": "TEXT NOT NULL",
            "domain_file_path": "TEXT NOT NULL",
            "instance_file_path": "TEXT NOT NULL",
            "is_longer_plan": "BOOLEAN",
            "paas_status": "TEXT",
            "pddl_plan": "TEXT",
            "type": "TEXT",
            "landmark_graph": "TEXT",
            "landmark_graph_status": "TEXT"
        }

class TaskDatabase(database.DatabaseManager):
    # TODO: CHANGE FILE PATH
    def __init__(self, file_path: str = config.TASKS_DATASET_FILE_PATH):
        super().__init__(file_path, "tasks", Task)

    def filter_functions(self) -> dict[str, str]:
        return {
            "filter_by_domain": " AND domain = ?",
            "filter_by_task_type": " AND type = ?",
            "is_longer_plan": " AND is_longer_plan = ?",
        }