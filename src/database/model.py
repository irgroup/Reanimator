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

class Publication(Base):
    __tablename__ = "publication"

    id: Mapped[int] = mapped_column(primary_key=True)

    documents: Mapped[List["Document"]] = relationship(back_populates="publication")


association_table = Table(
    "association_table",
    Base.metadata,
    Column("document_doi", ForeignKey("document.doi")),
    Column("author_id", ForeignKey("author.id"))
)

class Document(Base):
    __tablename__ = "document"

    doi: Mapped[str] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(1024), nullable=True)

    publication: Mapped[Publication] = relationship(back_populates="documents")
    publication_id: Mapped[int] = mapped_column(ForeignKey("publication.id"))

    tables: Mapped[List["Table"]] = relationship(back_populates="document")
    figures: Mapped[List["Figure"]] = relationship(back_populates="document")
    equations: Mapped[List["Equation"]] = relationship(back_populates="document")

    
    authors: Mapped[List["Author"]] = relationship(secondary=association_table)

    pdf: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)  # Field to store PDF binary
    pdf_is_sane: Mapped[bool] = mapped_column(Boolean, default=False)  # Indicates whether the PDF is sane

class Author(Base):
    __tablename__ = "author"


    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(1024))
    
    #institutions:  Mapped[List["Institution"]] = relationship(back_populates="authors")


class Institution(Base):
    __tablename__ = "institution"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256))

    #authors: Mapped[List["Author"]] = relationship(back_populates="institutions")


class Venue(Base):
    __tablename__ = "venue"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    pass

class Table(Base):
    __tablename__ = "table"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ir_id: Mapped[str] = mapped_column(String(1024))
    ir_tab_id: Mapped[str] = mapped_column(String(1024))
    table_name = mapped_column(String(2048), nullable=True)

    pm_content: Mapped[str] = mapped_column(String(2**15), nullable=True) #32768

    #header_json: Mapped[JSON] = mapped_column(type_=JSON, nullable=False)
    #content_json: Mapped[JSON] = mapped_column(type_=JSON, nullable=False)

    header: Mapped[List[str]] = mapped_column(ARRAY(String(2**15)), nullable=True)
    #content: Mapped[List[List[str]]] = mapped_column(ARRAY(String(2**15), dimensions=2), nullable=True)
    content: Mapped[List[List[str]]] = mapped_column(JSONB, nullable=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.doi"))
    document: Mapped["Document"] = relationship(back_populates="tables")

    position_page: Mapped[int] = mapped_column(Integer(), nullable=True) #do we want to map this to document pages?
    position_left: Mapped[float] = mapped_column(Float(), nullable=True)
    position_top: Mapped[float] = mapped_column(Float(), nullable=True)
    width: Mapped[float] = mapped_column(Float(), nullable=True)
    height: Mapped[float] = mapped_column(Float(), nullable=True)

    caption: Mapped[str] = mapped_column(String(2**15), nullable=True)
    references: Mapped[List[str]] = mapped_column(ARRAY(String(2**15)), nullable=True)


class Figure(Base):
    __tablename__ = "figure"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ir_id: Mapped[str] = mapped_column(String(1024))

    caption: Mapped[str] = mapped_column(String(2**15), nullable=True)

    document_id: Mapped[int] = mapped_column(ForeignKey("document.doi"))
    document: Mapped["Document"] = relationship(back_populates="figures")

    position_page: Mapped[int] =  mapped_column(Integer()) #do we want to map this to document pages?
    position_left: Mapped[float] = mapped_column(Float())
    position_top: Mapped[float] = mapped_column(Float())
    width: Mapped[float] = mapped_column(Float())
    height: Mapped[float] = mapped_column(Float())

class Equation(Base):
    __tablename__ = "equation"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(String(2**15), nullable=True) #32768

    document_id: Mapped[int] = mapped_column(ForeignKey("document.doi"))
    document: Mapped["Document"] = relationship(back_populates="equations")

    position_page: Mapped[int] = mapped_column(Integer()) #do we want to map this to document pages?
    position_left: Mapped[float] = mapped_column(Float())
    position_top: Mapped[float] = mapped_column(Float())
    width: Mapped[float] = mapped_column(Float())
    height: Mapped[float] = mapped_column(Float())

class Section(Base):
    __tablename__ = "section"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    pass

class Paragraph(Base):
    __tablename__ = "paragraph"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    pass

class Sentence(Base):
    __tablename__ = "sentence"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True) 
    pass