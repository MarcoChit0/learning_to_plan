from __future__ import annotations
# TODO: UPDATE TO USE SQL INSTEAD OF JSONL
# TODO: CREATE A TABLE FOR CONTENT AS WELL
# TODO: MAKE MOST OF METHODS REUSABLE FOR CONTENT AS WELL
# TODO: WHEN CONTENT USES THE DATABASE, ITS PROBLEM WITH IDS WOULD BE SOLVED
import abc
import datetime
from enum import Enum
import sqlite3
import json
import os
from typing import Optional
from learning_to_plan import config
from learning_to_plan.task import Task
from learning_to_plan import generated_plans

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


class Data(abc.ABC):
    SEEN_IDS: set[int] = set()
    AVAILABLE_ID_POOL: set[int] = set()
    NEXT_ID: int = 0
    def __init__(self, id: Optional[int] = None, field_names: list[str] = [], **kwargs):
        if id is None:
            self.id = self.get_new_id()
        else:
            try:
                self.add_new_id(id)
                self.id = id
            except ValueError as e:
                logger.error(f"Error adding new ID {id}: {e}")
                raise e
        self.__dict__.update(kwargs)
        # field names should be ordered according to the database schema
        self.field_names = field_names
        assert isinstance(self.field_names, list), "field_names must be a list of strings."
        assert all(isinstance(name, str) for name in self.field_names), "All field names must be strings."
        assert len(self.field_names) > 0, "field_names must not be an empty list."
    
    # helper method to convert a value to an enum if applicable
    @staticmethod
    def _get_enum_value(value, enum_cls, name):
        if isinstance(value, enum_cls):
            return value
        if isinstance(value, str):
            try:
                return enum_cls(value)
            except ValueError:
                logger.error(f"Invalid {name} value: {value}. Defaulting to None.")
        return None

    @abc.abstractmethod
    def storage_datatype(self) -> dict[str, str]:
        raise NotImplementedError("Subclasses must implement the storage_datatype method.")
    
    def from_row(self, row: tuple) -> Data:
        if len(row) != len(self.field_names):
            raise ValueError(f"Row length {len(row)} does not match field names length {len(self.field_names)}.")
        data_dict = dict(zip(self.field_names, row))
        return self.__class__(id=data_dict.get('id', None), **data_dict)
    
    def to_row(self) -> tuple:
        try:
            for field_name in self.field_names:
                if not hasattr(self, field_name):
                    raise ValueError(f"Field '{field_name}' is missing in {self.__class__.__name__} object.")
        except AttributeError as e:
            logger.error(f"Error converting {self.__class__.__name__} to row: {e}")
            raise e
        row = []
        for field_name in self.field_names:
            value = getattr(self, field_name, None)
            try:
                transformed_value = transform_value_to_sqlite_storage(value)
            except TypeError as e:
                logger.error(f"Error transforming value for field '{field_name}': {e}")
                raise e
            row.append(transformed_value)
        return tuple(row)

    @classmethod
    def add_new_id(cls, new_id: int) -> None:
        """
        Adds a new ID to the available content ID pool.
        This method is used to ensure that IDs are unique and can be reused.
        """
        if new_id in cls.SEEN_IDS:
            raise ValueError(f"ID {new_id} already exists in Content.SEEN_IDS.")
        if new_id in cls.AVAILABLE_ID_POOL:
            cls.AVAILABLE_ID_POOL.remove(new_id)
        cls.SEEN_IDS.add(new_id)
        if new_id >= cls.NEXT_ID:
            for i in range(cls.NEXT_ID, new_id):
                if i not in cls.SEEN_IDS:
                    cls.AVAILABLE_ID_POOL.add(i)
            cls.NEXT_ID = new_id + 1
    
    @classmethod
    def get_new_id(cls) -> int:
        """
        Returns a new ID from the available content ID pool or generates a new one.
        """
        if cls.AVAILABLE_ID_POOL:
            new_id = cls.AVAILABLE_ID_POOL.pop()
        else:
            new_id = cls.NEXT_ID
            cls.NEXT_ID += 1
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
        cls.AVAILABLE_ID_POOL.add(id)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, isinstance(self.__class__)):
            return False
        return self.id == other.id

    def __lt__(self, other):
        if not isinstance(other, isinstance(self.__class__)):
            raise TypeError(f"Cannot compare {self.__class__.__name__} with {other.__class__.__name__}.")
        return self.id < other.id

class DatabaseManager(abc.ABC):
    def __init__(self, table_name: str, data_cls: type[Data], file_path: str, filters: dict[str, str] = {}):
        self.table_name: str = table_name
        self.content_database_cls: type[Data] = data_cls
        self.file_path: str = file_path
        self.connection = sqlite3.connect(file_path)
        # filters are used to filter the data when retrieving it from the database
        # they are expected to be a dictionary where keys are column names and values are the values to filter by
        self.filters: dict[str, str] = filters
        self.filters_ordering: list[str] = list(self.filters.keys())
        self.filters['number_of_instances'] = 'LIMIT ?'  # Special filter for limiting the number of instances
        self.filters_ordering.append('number_of_instances')  # Ensure this is the last filter applied
        self.setup()

    def close(self):
        self.connection.close()

    def commit(self):
        self.connection.commit()

    def cursor(self):
        return self.connection.cursor()
    
    def setup(self):
        cursor = self.cursor()
        # Create table if it does not exist
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                {', '.join([f"{col} {dtype}" for col, dtype in self.content_database_cls().storage_datatype().items()])}
            )
        """)
        self.commit()
    
    def get(self, **filters) -> set[Data]:
        query = f"SELECT * FROM {self.table_name} WHERE 1=1"
        params = []  
        for f in self.filters_ordering:
            if f in filters:
                value = filters[f]
                try:
                    trans_value = transform_value_to_sqlite_storage(value)
                except TypeError as e:
                    logger.error(f"Error transforming value {value} for filter '{f}': {e}")
                    raise e
                query += f" AND {self.filters[f]}"
                params.append(trans_value)

        objs: set[Data] = set()
        cursor = self.cursor()
        cursor.execute(query, params)
        for row in cursor.fetchall():
            obj = self.content_database_cls().from_row(row)
            objs.add(obj)
        return objs

    def get_by_id(self, id: int) -> Optional[Data]:
        query = f"SELECT * FROM {self.table_name} WHERE id = ?"
        cursor = self.cursor()
        cursor.execute(query, (id,))
        row = cursor.fetchone()
        if row:
            return self.content_database_cls().from_row(row)
        return None
    
    def add(self, obj: Optional[Data] = None, objs: Optional[set[Data]] = None):
        assert obj is not None or objs is not None, "Must provide either obj or objs to add."
        assert not (obj is not None and objs is not None), "Cannot provide both obj and objs to add."

        def _add_single(obj: Data):
            if not isinstance(obj, self.content_database_cls):
                raise TypeError(f"Object must be an instance of {self.content_database_cls.__name__}.")
            
            assert hasattr(obj, 'id'), "Object must have an 'id' attribute."

            row = obj.to_row()
            placeholders = ', '.join(['?'] * len(row))
            query = f"INSERT OR REPLACE INTO {self.table_name} VALUES ({placeholders})"
            self.cursor().execute(query, row)
            self.commit()

        def _add_objects(objs: set[Data]):
            if not isinstance(objs, set):
                raise TypeError("objs must be a set of Data objects.")
            for obj in objs:
                try:
                    _add_single(obj)
                except Exception as e:
                    logger.error(f"Error adding object {obj.id}: {e}", exc_info=True)

        if obj is not None:
            _add_single(obj)
        elif objs is not None:
            _add_objects(objs)
    
    def delete(self, id: Optional[int] = None, obj: Optional[Data] = None, objs: Optional[set[Data]] = None):
        assert id is not None or obj is not None or objs is not None, "Must provide either id, obj, or objs to delete."
        assert not (id is not None and (obj is not None or objs is not None)), "Cannot provide both id and obj/objs to delete."
        def _delete_single(id: int):
            query = f"DELETE FROM {self.table_name} WHERE id = ?"
            self.cursor().execute(query, (id,))
            self.commit()
            self.content_database_cls.remove_id(id)
        
        def _delete_object(obj: Data):
            if not isinstance(obj, self.content_database_cls):
                raise TypeError(f"Object must be an instance of {self.content_database_cls.__name__}.")
            _delete_single(obj.id)

        def _delete_objects(objs: set[Data]):
            if not isinstance(objs, set):
                raise TypeError("objs must be a set of Data objects.")
            for obj in objs:
                _delete_object(obj)
        
        if id is not None:
            _delete_single(id)
        elif obj is not None:
            _delete_object(obj)
        elif objs is not None:
            _delete_objects(objs)