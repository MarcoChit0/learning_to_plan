from __future__ import annotations
import abc
import sqlite3
from typing import Optional
from learning_to_plan import config
from learning_to_plan.data import base, task, generated_plan

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
        print(f"CREATE TABLE IF NOT EXISTS {self.table_name} ({table_definition})")
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
    
    def add(self, obj: base.Data | set[base.Data]):
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

        # Batch insert using executemany
        rows = [d.to_row() for d in _data]
        placeholders = ', '.join(['?'] * len(rows[0]))
        query = f"INSERT OR REPLACE INTO {self.table_name} VALUES ({placeholders})"
        
        try:
            self.cursor().executemany(query, rows)
            self.commit()
        except sqlite3.IntegrityError as e:
            logger.error(f"Integrity error while adding objects: {e}")
            raise e
            
    def update(self, obj: base.Data | set[base.Data]):
        assert isinstance(obj, (base.Data, set)), "obj must be an instance of Data or a set of Data objects."
        if isinstance(obj, set):
            assert len(obj) > 0, "Set of objects must not be empty."
            assert all(isinstance(o, self.data_cls) for o in obj), f"All objects in the set must be instances of {self.data_cls.__name__}."
            _data = obj
        else:
            assert isinstance(obj, self.data_cls), f"Object must be an instance of {self.data_cls.__name__}."
            _data = {obj}

        def _update_single(obj: base.Data):
            assert hasattr(obj, 'id'), "Object must have an 'id' attribute."
            try:
                older_obj = self.get_by_id(obj.id)
                if older_obj is None:
                    self.add(obj)
                else:
                    row = obj.to_row()
                    query = f"UPDATE {self.table_name} SET "
                    params = []
                    for field_name, value in zip(self.data_cls.get_field_names(), row):
                        if field_name != "id" and hasattr(older_obj, field_name) and getattr(older_obj, field_name) != getattr(obj, field_name):
                            query += f"{field_name} = ?, "
                            params.append(value)
                    query = query.rstrip(', ') + " WHERE id = ?"
                    params.append(obj.id)  
                    
                    try:
                        self.cursor().execute(query, params)
                        self.commit()
                    except sqlite3.IntegrityError as e:
                        logger.error(f"Integrity error while updating object {obj.id}: {e}")
                        raise e
            except Exception as e:
                raise e

        for d in _data:
            try:
                _update_single(d)
            except ValueError as e:
                logger.error(f"Error updating object {d.id}: {e}")
                raise e
    
    def delete(self, obj : int | base.Data | set[base.Data]):
        assert isinstance(obj, (int, base.Data, set)), "obj must be an int, an instance of Data, or a set of Data objects."

        def _delete_single(id: int):
            try:
                _obj = self.get_by_id(id)
                if _obj is None:
                    raise ValueError(f"Object with id {id} does not exist in the database.")
            except ValueError as e:
                logger.error(f"Error retrieving object with id {id}: {e}")
                raise e
            try:
                query = f"DELETE FROM {self.table_name} WHERE id = ?"
                self.cursor().execute(query, (id,))
                self.commit()
                self.data_cls.remove_id(id)
            except sqlite3.IntegrityError as e:
                logger.error(f"Integrity error while deleting object with id {id}: {e}")
                raise e
        
        def _delete_object(obj: base.Data):
            assert isinstance(obj, self.data_cls), f"Object must be an instance of {self.data_cls.__name__}."
            try:
                _delete_single(obj.id)
            except ValueError as e:
                raise ValueError(f"Could not delete object with id {obj.id}: {e}")

        def _delete_objects(objs: set[base.Data]):
            assert len(obj) > 0, "Set of objects must not be empty."
            assert all(isinstance(o, self.data_cls) for o in obj), f"All objects in the set must be instances of {self.data_cls.__name__}."
            for obj in objs:
                try:
                    _delete_object(obj)
                except Exception as e:
                    objects_ids = ', '.join(str(o.id) for o in objs)
                    raise ValueError(f"Could not delete objects with ids {objects_ids}: {e}")
        try:
            if isinstance(obj, set):
                _delete_objects(obj)
            elif isinstance(obj, base.Data):
                _delete_object(obj)
            elif isinstance(obj, int):
                _delete_single(obj)
            else:
                raise TypeError(f"Unsupported type for deletion: {type(obj)}. Must be int, Data, or set of Data objects.")
        except Exception as e:
            logger.error(f"Error deleting object(s): {e}")
            raise e

task_database: Optional[DatabaseManager] = None
generated_plan_database: Optional[DatabaseManager] = None

def initialize():
    global task_database, generated_plan_database
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

    if generated_plan_database is None:
        generated_plan_database = DatabaseManager(
            table_name="generated_plan",
            data_cls=generated_plan.GeneratedPlan,
            filters={
                "filter_by_task_id": "task_id = ?",
                "filter_by_model_metadata": "model_metadata = ?",
                "filter_by_prompt_metadata": "prompt_metadata = ?",
            }
        )
    
    for db in [task_database, generated_plan_database]:
        generated_plans = db.get()
        db.data_cls.NEXT_ID = max((plan.id for plan in generated_plans), default=-1) + 1