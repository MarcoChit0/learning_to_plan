from __future__ import annotations
from typing import Dict, Any, Optional
from learning_to_plan import config
from learning_to_plan.data import base
# MUST BE THE SAME ID ACROSS ALL MODELS
logger = config.get_logger(__name__)
# TODO: when deleting from the database, all the generated plans with this metadata should be deleted as well

class Metadata(base.Data):
    NEXT_ID: int = 0
    SETTED_ID_VARIABLES: bool = False
    FIELD_NAMES = [
        "id",
        "info",]
    
    def __init__(
            self,
            id: Optional[int] = None,
            info: Optional[Dict[str, Any] | str] = None):
        super().__init__(id)
        self.info = info if isinstance(info, dict) else base.transform_value_from_sqlite_storage(info)
        if self.info is None:
            self.info = {}
        assert isinstance(self.info, dict), "Info must be a dictionary."

    @classmethod
    def column_def(cls):
        return [
            "id INTEGER PRIMARY KEY",
            "info TEXT NOT NULL",
        ]
    @classmethod
    def column_constraints(cls):
        return [
            "UNIQUE (info)"
        ]
    
    def update(self, **kwargs: Any) -> None:
        self.info.update(kwargs)
        

def create_metadata(**info) -> Metadata:
    from learning_to_plan.database import metadata_database
    if metadata_database is None:
        raise RuntimeError("Metadata database is not initialized.")

    metadatas = metadata_database.get(
        filter_by_info=info
    )
    if metadatas:
        return next(iter(metadatas))
    else:
        metadata = Metadata(info=info)
        metadata_database.add(metadata)
        return metadata