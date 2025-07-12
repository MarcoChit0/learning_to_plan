from __future__ import annotations
import datetime
from enum import Enum
from typing import Dict, Any, Optional
from learning_to_plan import config
from learning_to_plan.data import base

logger = config.get_logger(__name__)

class GeneratedPlan(base.Data):
    NEXT_ID: int = 0
    SETTED_ID_VARIABLES: bool = False
    FIELD_NAMES = [
            "id",
            "task_id",
            "model_metadata_id",
            "prompt_metadata_id",
            "pddl_plan",
            "validity",
            "date",
            "error_message"
            "generation_metadata"
        ]
    
    class VALIDITY(Enum):
        VALID = "valid"
        INVALID = "invalid"
        UNCHECKED = "unchecked"
    def __init__(
            self,
            task_id: int,
            model_metadata_id: int,
            prompt_metadata_id: int,
            id: Optional[int] = None,
            pddl_plan: Optional[str] = None,
            validity: VALIDITY | str = VALIDITY.UNCHECKED,
            date: Optional[datetime.datetime | str] = None,
            error_message: Optional[str] = None,
            specs: Optional[Dict[str, Any] | str] = None
        ):
        super().__init__(id)
        self.task_id = task_id
        self.prompt_metadata_id = prompt_metadata_id
        self.model_metadata_id = model_metadata_id
        self.pddl_plan = pddl_plan
        self.error_message = error_message
        assert (self.pddl_plan or self.error_message) and not (self.pddl_plan and self.error_message), \
            "Either pddl_plan or error_message must be provided, but not both. If there is an error, use error_message. If the plan is valid, use pddl_plan."

        self.validity = config.get_enum_value(validity, GeneratedPlan.VALIDITY, "validity")
        assert self.validity, "Validity must not be None."

        self.generation_metadata = specs if isinstance(specs, dict) else base.transform_value_from_sqlite_storage(specs)

        if date is None or base.transform_value_from_sqlite_storage(date) is None:
            self.date = datetime.datetime.now()
        else:
            self.date = base.transform_value_from_sqlite_storage(date)
        
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
        validity_values = ', '.join([f"'{v.value}'" for v in cls.VALIDITY])
        return [
            "id INTEGER PRIMARY KEY",
            "task_id INTEGER NOT NULL",
            "model_metadata_id INTEGER NOT NULL",
            "prompt_metadata_id INTEGER NOT NULL",
            "pddl_plan TEXT",
            f"validity TEXT NOT NULL CHECK(validity IN ({validity_values}))",
            "date TEXT",
            "error_message TEXT",
            "generation_metadata TEXT"
        ]

    @classmethod
    def column_constraints(cls):
        return [
            "CONSTRAINT fk_task_id FOREIGN KEY(task_id) REFERENCES tasks(id)",
            "CONSTRAINT fk_model_metadata_id FOREIGN KEY(model_metadata_id) REFERENCES metadata(id)",
            "CONSTRAINT fk_prompt_metadata_id FOREIGN KEY(prompt_metadata_id) REFERENCES metadata(id)",
            "CONSTRAINT tup_tid_pm_mm UNIQUE(task_id, prompt_metadata_id, model_metadata_id)",
        ]