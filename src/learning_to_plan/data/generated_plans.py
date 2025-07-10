from __future__ import annotations
import datetime
from enum import Enum
import json
from typing import Dict, Any, Optional
from learning_to_plan import config
from learning_to_plan.data import database_manager, base
# MUST BE THE SAME ID ACROSS ALL MODELS
logger = config.get_logger(__name__)
# TODO : ADD EXPERIMENT TAG TO CONTENT

class GeneratedPlan(base.Data):
    SEEN_IDS: set[int] = set()
    AVAILABLE_ID_POOL: set[int] = set()
    NEXT_ID: int = 0
    FIELD_NAMES = [
            "id",
            "task_id",
            "prompt_type",
            "raw_plan",
            "pddl_plan",
            "validity",
            "model_metadata",
            "prompt_metadata",
            "date",
            "status",
            "error_message"
        ]
    
    class VALIDITY(Enum):
        VALID = "valid"
        INVALID = "invalid"
        UNCHECKED = "unchecked"
    def __init__(
            self,
            task_id: int,
            prompt_type: config.PROMPT_TYPE | str,
            id: Optional[int] = None,
            raw_plan: Optional[str] = None,
            pddl_plan: Optional[str] = None,
            validity: VALIDITY | str = VALIDITY.UNCHECKED,
            model_metadata: Optional[Dict[str, Any] | str] = None,
            prompt_metadata: Optional[Dict[str, Any] | str] = None,
            date: Optional[datetime.datetime | str] = None,
            status: config.STATUS | str = config.STATUS.OK,
            error_message: Optional[str] = None
        ):
        super().__init__(id)
        self.task_id = task_id
        self.prompt_type = config.get_enum_value(prompt_type, config.PROMPT_TYPE, "prompt_type")
        assert self.prompt_type, "Prompt type must not be None."
        self.raw_plan = raw_plan
        self.pddl_plan = pddl_plan

        self.error_message = error_message
        self.status = config.get_enum_value(status, config.STATUS, "status")
        assert self.status, "Status must not be None."
        if self.status == config.STATUS.OK:
            assert raw_plan and pddl_plan, "Both raw_plan and pddl_plan must be provided when status is OK."
            assert self.error_message is None, "Error message must be None when status is OK."
        else:
            assert self.error_message, "Error message must be provided when status is ERROR."
            assert raw_plan is None and pddl_plan is None, "raw_plan and pddl_plan must be None when status is ERROR."
        
        self.validity = config.get_enum_value(validity, GeneratedPlan.VALIDITY, "validity")
        assert self.validity, "Validity must not be None."

        for var in ["model_metadata", "prompt_metadata"]:
            value = locals()[var]
            if isinstance(value, str):
                try:
                    setattr(self, var, json.loads(value))
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding {var} JSON: {e}")
                    setattr(self, var, {})
            elif isinstance(value, dict):
                setattr(self, var, value)
            else:
                logger.error(f"{var} must be a dict or a JSON string.")
                setattr(self, var, {})

        if date is None:
            self.date = datetime.datetime.now()
        elif isinstance(date, str):
            try:
                self.date = datetime.datetime.fromisoformat(date)
            except ValueError as e:
                logger.error(f"Error parsing date string '{date}': {e}")
                self.date = datetime.datetime.now()
        elif isinstance(date, datetime.datetime):
            self.date = date
        else:
            raise ValueError(f"Date must be a datetime object or a string in ISO format, got {type(date)}.")
        
    def was_validated(self) -> bool:
        """
        Returns True if the plan has been validated, False otherwise.
        """
        return self.validity != GeneratedPlan.VALIDITY.UNCHECKED

    def validate(self, is_valid: bool) -> None:
        """
        Validates the plan.
        """
        if is_valid:
            logger.debug(f"Plan '{self.raw_plan}' validated as valid.")
            self.validity = GeneratedPlan.VALIDITY.VALID
        else:
            logger.debug(f"Plan '{self.raw_plan}' validated as invalid.")
            self.validity = GeneratedPlan.VALIDITY.INVALID
    
    def is_valid(self) -> bool:
        """
        Returns True if the plan is valid, False otherwise.
        """
        if self.validity == GeneratedPlan.VALIDITY.VALID:
            return True
        elif self.validity == GeneratedPlan.VALIDITY.INVALID:
            return False
        else:
            raise ValueError(f"Validity status is {self.validity}, which is not valid for checking plan validity.")

    @classmethod
    def column_def(cls):
        return [
            "id INTEGER PRIMARY KEY",
            "task_id INTEGER",
            "prompt_type TEXT NOT NULL",
            "raw_plan TEXT",
            "pddl_plan TEXT",
            "validity TEXT",
            "model_metadata TEXT",
            "prompt_metadata TEXT",
            "date TEXT",
            "status TEXT",
            "error_message TEXT",
        ]

    @classmethod
    def column_constraints(cls):
        return [
            "CONSTRAINT fk_task_id FOREIGN KEY(task_id) REFERENCES tasks(id)",
            "CONSTRAINT tup_tid_pt_pm_mm UNIQUE(task_id, prompt_type, prompt_metadata, model_metadata)" 
        ]

generated_plan_database: Optional[database_manager.DatabaseManager] = None
def initialize_db():
    global generated_plan_database
    if generated_plan_database is None:
        generated_plan_database = database_manager.DatabaseManager(
            table_name="generated_plan",
            data_cls=GeneratedPlan,
            filters={
                "filter_by_task_id": "task_id = ?",
                "filter_by_prompt_type": "prompt_type = ?",
                "filter_by_model_metadata": "model_metadata = ?",
                "filter_by_prompt_metadata": "prompt_metadata = ?",
            }
        )
    _ = generated_plan_database.get()