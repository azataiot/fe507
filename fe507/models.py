# fe507 / models.py
# Created by azat at 9.01.2023
from pydantic import BaseModel
from typing import TypeVar, Optional

from fe507 import CollectionGroup

CG = TypeVar('CG', bound=CollectionGroup)


class NamedDataGroup(BaseModel):
    name: str
    collection_group: Optional[CollectionGroup] = None

    class Config:
        arbitrary_types_allowed = True


class YearRange(BaseModel):
    from_year: int
    to_year: int
