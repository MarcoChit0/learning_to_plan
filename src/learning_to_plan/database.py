from __future__ import annotations
import abc
import sqlite3
from typing import Optional
from learning_to_plan import config
from learning_to_plan.data import base, task, generated_plan, metadata

logger = config.get_logger(__name__)

class DatabaseManager(abc.ABC):
    def __init__(self, table_name: str, data_cls: type[base.Data], filters: dict[str, str] = {}):
        self.table_name: str = table_name
        self.data_cls: type[base.Data] = data_cls
        self.connection = sqlite3.connect(config.DATABASE_FILE_PATH)
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

    def __del__(self):
        try:
            self.close()
        except sqlite3.Error as e:
            logger.error(f"Error closing database connection: {e}")
    
    def setup(self):
        cursor = self.cursor()
        columns_def_str = ', '.join(self.data_cls.column_def())
        constraints_str = ', '.join(self.data_cls.column_constraints())
        table_definition = f"{columns_def_str}, {constraints_str}" if constraints_str else columns_def_str
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {self.table_name} ({table_definition})")
        self.commit()
    
    def get(self, **filters) -> set[base.Data]:
        query = f"SELECT * FROM {self.table_name} WHERE 1=1"
        params = []  
        for f in self.filters_ordering:
            if f in filters:
                value = filters[f]
                try:
                    trans_value = base.transform_value_to_sqlite_storage(value)
                except TypeError as e:
                    logger.error(f"Error transforming value {value} for filter '{f}': {e}")
                    raise e
                if f == self.filters_ordering[-1]:
                    sep = " "
                else:
                    sep = " AND "
                query += f"{sep}{self.filters[f]}"
                params.append(trans_value)

        objs: set[base.Data] = set()
        cursor = self.cursor()
        cursor.execute(query, params)
        for row in cursor.fetchall():
            obj = self.data_cls.from_row(row)
            objs.add(obj)
        return objs

    def get_by_id(self, id: int) -> Optional[base.Data]:
        query = f"SELECT * FROM {self.table_name} WHERE id = ?"
        cursor = self.cursor()
        cursor.execute(query, (id,))
        row = cursor.fetchone()
        if row:
            return self.data_cls.from_row(row)
        return None
    
    def _execute_many(self, query: str, obj: set[base.Data] | base.Data):
        assert isinstance(obj, (base.Data, set)), "obj must be an instance of Data or a set of Data objects."
        if isinstance(obj, set):
            assert len(obj) > 0, "Set of objects must not be empty."
            assert all(isinstance(o, self.data_cls) for o in obj), f"All objects in the set must be instances of {self.data_cls.__name__}."
            _data = obj
        else:
            assert isinstance(obj, self.data_cls), f"Object must be an instance of {self.data_cls.__name__}."
            _data = {obj}

        # Validate all objects have id attribute
        for d in _data:
            assert hasattr(d, 'id'), "Object must have an 'id' attribute."
        rows = [d.to_row() for d in _data]  

        try:
            self.cursor().executemany(query, rows)
            self.commit()
        except sqlite3.IntegrityError as e:
            logger.error(f"Integrity error while adding objects: {e}")
            raise e

    def _verify_obj(self, obj: base.Data | set[base.Data]):
        assert isinstance(obj, (base.Data, set)), "obj must be an instance of Data or a set of Data objects."
        if isinstance(obj, set):
            assert len(obj) > 0, "Set of objects must not be empty."
            assert all(isinstance(o, self.data_cls) for o in obj), f"All objects in the set must be instances of {self.data_cls.__name__}."
        else:
            assert isinstance(obj, self.data_cls), f"Object must be an instance of {self.data_cls.__name__}."
        

    def _get_rows(self, obj: base.Data | set[base.Data]):
        try:
            self._verify_obj(obj)
        except AssertionError as e:
            raise ValueError(f"Error verifying object(s): {e}")
        
        if isinstance(obj, set):
            _data = obj
        else:
            _data = {obj}
        
        return [d.to_row() for d in _data]
        

    def add(self, obj: base.Data | set[base.Data]):
        try:
            rows = self._get_rows(obj)
        except Exception as e:
            raise ValueError(f"Error getting rows from object(s): {e}")
        placeholders = ', '.join(['?'] * len(rows[0]))
        query = f"INSERT OR REPLACE INTO {self.table_name} VALUES ({placeholders})"
        try:
            self._execute_many(query, obj)
        except sqlite3.IntegrityError as e:
            logger.error(f"Integrity error while adding objects: {e}")
            raise e
            
    def update(self, obj: base.Data | set[base.Data]):
        try:
            rows = self._get_rows(obj)
        except Exception as e:
            raise ValueError(f"Error getting rows from object(s): {e}")
        placeholders = ', '.join(['?'] * len(rows[0]))
        query = f"REPLACE INTO {self.table_name} VALUES ({placeholders})"
        try:
            self._execute_many(query, obj)
        except sqlite3.IntegrityError as e:
            logger.error(f"Integrity error while updating objects: {e}")
            raise e
    
    def delete(self, obj: int | base.Data | set[base.Data]):
        assert isinstance(obj, (int, base.Data, set)), "obj must be an int, an instance of Data, or a set of Data objects."

        ids_to_delete: list[int] = []
        if isinstance(obj, int):
            ids_to_delete.append(obj)
        else:    
            try:
                self._verify_obj(obj)
                if isinstance(obj, base.Data):
                    ids_to_delete.append(obj.id)
                elif isinstance(obj, set):
                    ids_to_delete.extend(o.id for o in obj)
                else:
                    raise TypeError(f"Unsupported type for deletion: {type(obj)}. Must be int, Data, or set of Data objects.")
            except ValueError as e:
                raise ValueError(f"Error verifying object(s) for deletion: {e}")

        if not ids_to_delete:
            return

        try:
            placeholders = ', '.join(['?'] * len(ids_to_delete))
            query = f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})"
            cursor = self.cursor()
            cursor.execute(query, ids_to_delete)
            
            if cursor.rowcount != len(ids_to_delete):
                logger.warning(f"Attempted to delete {len(ids_to_delete)} object(s), but {cursor.rowcount} were deleted. Some IDs might not exist.")

            self.commit()
            
        except sqlite3.Error as e:
            logger.error(f"Error deleting object(s) with ids {ids_to_delete}: {e}")
            raise e

task_database: Optional[DatabaseManager] = None
generated_plan_database: Optional[DatabaseManager] = None
metadata_database: Optional[DatabaseManager] = None

def initialize():
    global task_database, generated_plan_database, metadata_database
    if task_database is None:
        task_database = DatabaseManager(
            table_name="task",
            data_cls=task.Task,
            filters={
                "filter_by_domain": "domain = ?",
                "filter_by_purpose": "purpose = ?",
                "filter_by_type": "type = ?",
                "filter_by_paas_status": "paas_status = ?",
            }
        )

    if metadata_database is None:
        metadata_database = DatabaseManager(
            table_name="metadata",
            data_cls=metadata.Metadata,
            filters={
                "filter_by_info": "info = ?",
            }
        )

    if generated_plan_database is None:
        generated_plan_database = DatabaseManager(
            table_name="generated_plan",
            data_cls=generated_plan.GeneratedPlan,
            filters={
                "filter_by_task_id": "task_id = ?",
                "filter_by_model_metadata_id": "model_metadata_id = ?",
                "filter_by_prompt_metadata_id": "prompt_metadata_id = ?",
                "filter_by_validity": "validity = ?",
            }
        )

        
    for db in [task_database, metadata_database, generated_plan_database]:
        generated_plans = db.get()
        db.data_cls.NEXT_ID = max((plan.id for plan in generated_plans), default=-1) + 1