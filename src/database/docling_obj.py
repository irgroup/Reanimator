from typing import List
from typing import Optional

from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy import Table, Column, Integer, String, Float, ARRAY, Boolean, LargeBinary
from sqlalchemy import ForeignKey


from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import DeclarativeBase

from sqlalchemy.dialects.postgresql import JSONB

from sqlalchemy import JSON


Base = declarative_base()

class Docling_Object(Base):
    __tablename__ = "docling_obj"

    doi: Mapped[str] = mapped_column(primary_key=True)
    docling_json: Mapped[JSON] = mapped_column(type_=JSON, nullable=False)
    