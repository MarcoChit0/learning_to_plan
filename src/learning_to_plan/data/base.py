from __future__ import annotations
import abc
import datetime
from enum import Enum
import json
from typing import Optional
from learning_to_plan import config

logger = config.get_logger(__name__)

def transform_value_to_sqlite_storage(value):
    """
    Transforms a value to a format suitable for SQLite storage.
    This function handles different data types and converts them accordingly.
    """
    if value is None:
        return "NULL"
    elif isinstance(value, Enum):
        return value.value
    elif isinstance(value, (list, set, dict)):
        return json.dumps(value)
    elif isinstance(value, datetime.datetime):
        return value.isoformat()
    elif isinstance(value, str):
        return value
    elif isinstance(value, (int, float)):
        return value
    else:
        raise TypeError(f"Unsupported type for SQLite storage: {type(value)}")

def transform_value_from_sqlite_storage(value):
    """
    Transforms a value from SQLite storage to its original format.
    This function handles different data types and converts them accordingly.
    """
    if value is None or value == "NULL":
        return None
    elif isinstance(value, int) or isinstance(value, float):
        return value
    elif isinstance(value, str):
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        try:
            # Try to parse as datetime
            return datetime.datetime.fromisoformat(value)
        except ValueError:
            pass
    return value
    

class Data(abc.ABC):
    AVAILABLE_ID_POOL: set[int] = set()
    NEXT_ID: int = 0

    FIELD_NAMES : list[str] = []

    def __init__(self, id: Optional[int] = None, **kwargs):
        if id is None:
            self.id = self.get_new_id()
        else:
            self.check_id(id)
            self.id = id
        self.__dict__.update(kwargs)
        # field names should be ordered according to the database schema

        assert all(isinstance(name, str) for name in self.get_field_names()), "All field names must be strings."
        assert len(self.get_field_names()) > 0, "field_names must not be an empty list."

    def __del__(self):
        self.remove_id(self.id)

    @classmethod
    @abc.abstractmethod
    def column_def(cls) -> list[str]:
        raise NotImplementedError("Subclasses must implement the columns_names method.")

    @classmethod
    @abc.abstractmethod
    def column_constraints(cls) -> list[str]:
        """
        Returns a list of constraints for the columns in the database.
        This is used to ensure that the database schema is consistent with the data class.
        """
        raise NotImplementedError("Subclasses must implement the column_constraints method.")
    
    @classmethod
    def from_row(cls, row: tuple) -> Data:
        if len(row) != len(cls.get_field_names()):
            raise ValueError(f"Row length {len(row)} does not match field names length {len(cls.get_field_names())}.")
        data_dict = {field_name: transform_value_from_sqlite_storage(value) for field_name, value in zip(cls.get_field_names(), row)}
        return cls(**data_dict)
    
    @classmethod
    def get_field_names(cls) -> list[str]:
        """
        Returns the field names of the data class.
        This is used to ensure that the field names are consistent with the database schema.
        """
        if not cls.FIELD_NAMES:
            raise ValueError(f"Field names for {cls.__name__} are not defined.")
        return cls.FIELD_NAMES

    def to_row(self) -> tuple:
        try:
            for field_name in self.get_field_names():
                if not hasattr(self, field_name):
                    raise ValueError(f"Field '{field_name}' is missing in {self.__class__.__name__} object.")
        except AttributeError as e:
            logger.error(f"Error converting {self.__class__.__name__} to row: {e}")
            raise e
        row = []
        for field_name in self.get_field_names():
            value = getattr(self, field_name, None)
            try:
                transformed_value = transform_value_to_sqlite_storage(value)
            except TypeError as e:
                logger.error(f"Error transforming value for field '{field_name}': {e}")
                raise e
            row.append(transformed_value)
        return tuple(row)

    @classmethod
    def check_id(cls, id: int) -> None:
        if id in cls.AVAILABLE_ID_POOL:
            cls.AVAILABLE_ID_POOL.remove(id)
        cls.SEEN_IDS.add(id)
        if id >= cls.NEXT_ID:
            for _id in range(cls.NEXT_ID, id):
                if _id not in cls.SEEN_IDS:
                    cls.AVAILABLE_ID_POOL.add(_id)
            cls.NEXT_ID = id + 1
    
    @classmethod
    def get_new_id(cls) -> int:
        if cls.AVAILABLE_ID_POOL:
            new_id = cls.AVAILABLE_ID_POOL.pop()
        else:
            new_id = cls.NEXT_ID
            cls.NEXT_ID += 1
        return new_id

    @classmethod
    def remove_id(cls, id: int) -> None:
        cls.AVAILABLE_ID_POOL.add(id)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id

    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot compare {self.__class__.__name__} with {other.__class__.__name__}.")
        return self.id < other.id
