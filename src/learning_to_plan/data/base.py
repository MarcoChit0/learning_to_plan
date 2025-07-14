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
    NEXT_ID: int = 0
    FIELD_NAMES : list[str] = []

    def __init__(self, id: Optional[int] = None, **kwargs):
        if id is None:
            id = self.__class__.NEXT_ID
        elif not isinstance(id, int):
            raise TypeError(f"ID must be an integer, got {type(id)} instead.")
        
        self.id = id
        if id >= self.__class__.NEXT_ID:
            self.__class__.NEXT_ID = id + 1

        self.__dict__.update(kwargs)
        assert all(isinstance(name, str) for name in self.get_field_names()), "All field names must be strings."
        assert len(self.get_field_names()) > 0, "field_names must not be an empty list."

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
