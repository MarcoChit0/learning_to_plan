from __future__ import annotations
import datetime
from enum import Enum
from typing import Dict, Any, Optional
from learning_to_plan import config, task, database
# MUST BE THE SAME ID ACROSS ALL MODELS
logger = config.get_logger(__name__)
class Content:
    AVAILABLE_CONTENT_ID_POOL:set[int] = set()
    NEXT_CONTENT_ID:int = 0
    SEEN_IDS:set[int] = set()
    class STATUS(Enum):
        OK = "ok"
        ERROR = "error"
    
    class VALIDITY(Enum):
        VALID = "valid"
        INVALID = "invalid"
        UNCHECKED = "unchecked"
    def __init__(
            self,
            t: task.Task,
            prompt_type: config.PROMPT_TYPE,
            raw_plan: Optional[str] = None,
            pddl_plan: Optional[str] = None,
            id: Optional[int] = None,
            validity: VALIDITY = VALIDITY.UNCHECKED,
            model_metadata: Optional[Dict[str, Any]] = None,
            prompt_metadata: Optional[Dict[str, Any]] = None,
            date: Optional[datetime.datetime] = None,
            status: STATUS = STATUS.OK,
            error_message: Optional[str] = None
        ):
        self._task = t
        self._prompt_type = prompt_type
        self._raw_plan = raw_plan
        self._pddl_plan = pddl_plan
        self._error_message = error_message
        self._status = status
        if self._status == Content.STATUS.OK:
            assert raw_plan and pddl_plan, "Both raw_plan and pddl_plan must be provided when status is OK."
            assert self._error_message is None, "Error message must be None when status is OK."
        else:
            assert self._error_message, "Error message must be provided when status is ERROR."
            assert raw_plan is None and pddl_plan is None, "raw_plan and pddl_plan must be None when status is ERROR."
        self._validity = validity
        self._model_metadata = model_metadata if model_metadata is not None else {}
        self._prompt_metadata = prompt_metadata if prompt_metadata is not None else {}
        if id is None:
            self._id = Content.get_new_id()
        else:
            try:
                Content.add_new_id(id)
                self._id = id
            except ValueError as e:
                logger.error(f"Error adding new ID {id}: {e}")
                raise e
        self._date = date if date is not None else datetime.datetime.now()
        
    @classmethod
    def add_new_id(cls, new_id: int) -> None:
        """
        Adds a new ID to the available content ID pool.
        This method is used to ensure that IDs are unique and can be reused.
        """
        if new_id in cls.SEEN_IDS:
            raise ValueError(f"ID {new_id} already exists in Content.SEEN_IDS.")
        if new_id in cls.AVAILABLE_CONTENT_ID_POOL:
            cls.AVAILABLE_CONTENT_ID_POOL.remove(new_id)
        cls.SEEN_IDS.add(new_id)
        if new_id >= cls.NEXT_CONTENT_ID:
            for i in range(cls.NEXT_CONTENT_ID, new_id):
                if i not in cls.SEEN_IDS:
                    cls.AVAILABLE_CONTENT_ID_POOL.add(i)
            cls.NEXT_CONTENT_ID = new_id + 1
    
    @classmethod
    def get_new_id(cls) -> int:
        """
        Returns a new ID from the available content ID pool or generates a new one.
        """
        if cls.AVAILABLE_CONTENT_ID_POOL:
            new_id = cls.AVAILABLE_CONTENT_ID_POOL.pop()
        else:
            new_id = cls.NEXT_CONTENT_ID
            cls.NEXT_CONTENT_ID += 1
        cls.SEEN_IDS.add(new_id)
        return new_id
    @classmethod
    def remove_id(cls, id: int) -> None:
        """
        Removes an ID from the seen IDs and adds it to the available content ID pool.
        This is used to recycle IDs when they are no longer needed.
        """
        if id not in cls.SEEN_IDS:
            raise ValueError(f"ID {id} does not exist in Content.SEEN_IDS.")
        cls.SEEN_IDS.remove(id)
        cls.AVAILABLE_CONTENT_ID_POOL.add(id)

    def was_validated(self) -> bool:
        """
        Returns True if the plan has been validated, False otherwise.
        """
        return self._validity != Content.VALIDITY.UNCHECKED

    def validate(self, is_valid: bool) -> None:
        """
        Validates the plan.
        """
        if is_valid:
            logger.debug(f"Plan '{self._raw_plan}' validated as valid.")
            self._validity = Content.VALIDITY.VALID
        else:
            logger.debug(f"Plan '{self._raw_plan}' validated as invalid.")
            self._validity = Content.VALIDITY.INVALID
    
    def is_valid(self) -> bool:
        """
        Returns True if the plan is valid, False otherwise.
        """
        if self._validity == Content.VALIDITY.VALID:
            return True
        elif self._validity == Content.VALIDITY.INVALID:
            return False
        else:
            raise ValueError(f"Validity status is {self._validity}, which is not valid for checking plan validity.")
    def __hash__(self):
        return hash(self._id)
    
    def __eq__(self, other):
        if not isinstance(other, Content):
            return False
        return self._id == other._id
    def __lt__(self, other):
        if not isinstance(other, Content):
            return NotImplemented
        return self._id < other._id
    
    @classmethod
    def read_from_json_row(cls, row: Dict[str, Any]) -> Content:
        """
        Reads a JSON‐serializable dict (from your .jsonl) and returns a Content.
        """
        t = database.get_task(
        row['domain_file_path'],
        row['instance_file_path'],
        )
        prompt_type = config.PROMPT_TYPE[row['prompt_type'].upper()]
        _id = row.get('id')
        raw_plan       = row.get('raw_plan', None)
        pddl_plan      = row.get('pddl_plan', None)
        validity_str       = row.get('validity', Content.VALIDITY.UNCHECKED.value)
        validity = Content.VALIDITY[validity_str.upper()]
        model_metadata = row.get('model_metadata', {})
        prompt_metadata= row.get('prompt_metadata', {})
        date_str       = row.get('date')
        date           = datetime.datetime.fromisoformat(date_str) if date_str else None
        status_str = row.get('status', Content.STATUS.OK.value)
        status = Content.STATUS[status_str.upper()]
        error_message = row.get('error_message', None)
        return cls(
        t=t,
        prompt_type=prompt_type,
        raw_plan=raw_plan,
        pddl_plan=pddl_plan,
        id=_id,
        validity=validity,
        model_metadata=model_metadata,
        prompt_metadata=prompt_metadata,
        date=date,
        status=status,
        error_message=error_message
        )
    def write_to_json_row(self) -> Dict[str, Any]:
        """
        Serializes this Content to a JSON‐serializable dict for .jsonl writing.
        """
        return {
        'domain_file_path': self._task._domain_file_path,
        'instance_file_path': self._task._instance_file_path,
        'prompt_type':       self._prompt_type.value,
        'id':                self._id,
        'raw_plan':          self._raw_plan,
        'pddl_plan':         self._pddl_plan,
        'validity':          self._validity.value,
        'model_metadata':    self._model_metadata,
        'prompt_metadata':   self._prompt_metadata,
        'date':              (self._date or datetime.datetime.now()).isoformat(),
        'status':            self._status.value,
        'error_message':     self._error_message,
        }