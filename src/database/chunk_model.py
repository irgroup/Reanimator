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


Chunk_Base = declarative_base()


class Chunk(Chunk_Base):
    __tablename__ = "chunk"

    id: Mapped[int] = mapped_column(primary_key=True)
    doi: Mapped[str] = mapped_column(String(1024))
    chunk_text: Mapped[str] = mapped_column(String(2**15), nullable=True)
    chunk_type: Mapped[str] = mapped_column(String(1024), nullable=True)
    modality_type: Mapped[str] = mapped_column(String(1024), nullable=True)

