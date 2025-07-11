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
    AVAILABLE_ID_POOL: set[int] = set()
    NEXT_ID: int = 0
    SETTED_ID_VARIABLES: bool = False
    FIELD_NAMES = [
            "id",
            "task_id",
            "pddl_plan",
            "validity",
            "model_metadata",
            "prompt_metadata",
            "date",
            "error_message"
        ]
    
    class VALIDITY(Enum):
        VALID = "valid"
        INVALID = "invalid"
        UNCHECKED = "unchecked"
    def __init__(
            self,
            task_id: int,
            id: Optional[int] = None,
            pddl_plan: Optional[str] = None,
            validity: VALIDITY | str = VALIDITY.UNCHECKED,
            model_metadata: Optional[Dict[str, Any] | str] = None,
            prompt_metadata: Optional[Dict[str, Any] | str] = None,
            date: Optional[datetime.datetime | str] = None,
            error_message: Optional[str] = None
        ):
        super().__init__(id)
        self.task_id = task_id
        self.pddl_plan = pddl_plan
        self.error_message = error_message
        assert (self.pddl_plan or self.error_message) and not (self.pddl_plan and self.error_message), \
            "Either pddl_plan or error_message must be provided, but not both. If there is an error, use error_message. If the plan is valid, use pddl_plan."

        self.validity = config.get_enum_value(validity, GeneratedPlan.VALIDITY, "validity")
        assert self.validity, "Validity must not be None."

        self.prompt_metadata = prompt_metadata if isinstance(prompt_metadata, dict) else base.transform_value_from_sqlite_storage(prompt_metadata)
        if self.prompt_metadata is None:
            self.prompt_metadata = {}
        
        self.model_metadata = model_metadata if isinstance(model_metadata, dict) else base.transform_value_from_sqlite_storage(model_metadata)
        if self.model_metadata is None:
            self.model_metadata = {}

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
            "pddl_plan TEXT",
            f"validity TEXT NOT NULL CHECK(validity IN ({validity_values}))",
            "model_metadata TEXT",
            "prompt_metadata TEXT",
            "date TEXT",
            "error_message TEXT",
        ]

    @classmethod
    def column_constraints(cls):
        return [
            "CONSTRAINT fk_task_id FOREIGN KEY(task_id) REFERENCES tasks(id)",
            "CONSTRAINT tup_tid_pm_mm UNIQUE(task_id, prompt_metadata, model_metadata)" 
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
                "filter_by_model_metadata": "model_metadata = ?",
                "filter_by_prompt_metadata": "prompt_metadata = ?",
            }
        )
    generated_plans = generated_plan_database.get()
    GeneratedPlan.set_id_variables(
        seen_ids={gp.id for gp in generated_plans}
    )