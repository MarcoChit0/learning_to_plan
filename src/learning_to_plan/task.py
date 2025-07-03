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

INSTANCE_PATTERN = re.compile(r"instance-(\d+)\.pddl$")
lock = threading.Lock()

class Task(abc.ABC):
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

    def __init__(self, domain : str, domain_file_path : str, instance_file_path : str):
        self._domain :str = domain
        self._domain_file_path : str = domain_file_path
        self._instance_file_path : str = instance_file_path
        self._is_longer_plan : bool = True if config.LONG_INSTANCES in self._instance_file_path else False
        self._id : int = int(re.search(INSTANCE_PATTERN, self._instance_file_path).group(1))
        self._paas_status : Optional[Task.PAAS_STATUS] = None # ok, error
        self._pddl_plan : Optional[str] = None
        self._type : Optional[Task.TYPE] = None # training, validation, test | None
        self._landmark_graph : Optional[str] = None # Landmark graph in DOT format, if available
        self._landmark_graph_status : Optional[Task.LANDMARK_GRAPH_STATUS] = None # ok, error

    def __lt__(self, other):
        if not isinstance(other, Task):
            return NotImplemented

        if self._domain != other._domain:
            return self._domain < other._domain

        if self._is_longer_plan != other._is_longer_plan:
            return not self._is_longer_plan

        self_match = INSTANCE_PATTERN.search(self._instance_file_path)
        other_match = INSTANCE_PATTERN.search(other._instance_file_path)
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
                "landmark_graph": self._landmark_graph,
                "landmark_graph_status": self._landmark_graph_status.value if self._landmark_graph_status else None
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
        self._landmark_graph = json_obj.get("landmark_graph", None)

        # --- Enum Fields ---
        # Use correct internal field names (_pddl_status, _type)
        for field_name, json_key, enum_type in [
            ("_paas_status", "paas_status", Task.PAAS_STATUS),
            ("_type", "type", Task.TYPE),
            ("_landmark_graph_status", "landmark_graph_status", Task.LANDMARK_GRAPH_STATUS)
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

    def read_instance(self):
        with lock and open(self._instance_file_path, "r", encoding='utf-8') as f:
            instance_content = f.read()
        return instance_content

    def read_domain(self):
        with lock and open(self._domain_file_path, "r", encoding='utf-8') as f:
            domain_content = f.read()
        return domain_content