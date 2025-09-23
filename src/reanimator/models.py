from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import pandas as pd
from io import StringIO
import json
import os

@dataclass
class Table:
    """Represents a single table extracted from a document."""
    id: str
    content: pd.DataFrame
    caption: Optional[str] = None
    name: Optional[str] = None
    references: Optional[List[str]] = None
    metadata: Dict = field(default_factory=dict)
    pos_page: Optional[int] = None
    pos_top: Optional[float] = None
    pos_left: Optional[float] = None
    pos_right: Optional[float] = None
    pos_bottom: Optional[float] = None

    def to_dict(self):
        """Converts the Table object to a JSON-serializable dictionary."""
        d = asdict(self)
        d['content'] = self.content.to_json(orient='split')
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'Table':
        """Creates a Table object from a dictionary."""
        d['content'] = pd.read_json(StringIO(d['content']), orient='split')
        return cls(**d)
    
@dataclass
class Figure:
    """Represents a single figure extracted from a document."""
    id: str
    caption: Optional[str] = None
    name: Optional[str] = None
    references: Optional[List[str]] = None
    metadata: Dict = field(default_factory=dict)
    pos_page: Optional[int] = None
    pos_top: Optional[float] = None
    pos_left: Optional[float] = None
    pos_right: Optional[float] = None
    pos_bottom: Optional[float] = None

    def to_dict(self):
        """Converts the Figure object to a JSON-serializable dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'Figure':
        """Creates a Figure object from a dictionary."""
        return cls(**d)

@dataclass
class Formula:
    """Represents a single formula extracted from a document."""
    id: str
    text: str
    latex: Optional[str] = None
    page: int = -1
    metadata: Dict = field(default_factory=dict)
    pos_page: Optional[int] = None
    pos_top: Optional[float] = None
    pos_left: Optional[float] = None
    pos_right: Optional[float] = None
    pos_bottom: Optional[float] = None

    def to_dict(self):
        """Converts the Formula object to a JSON-serializable dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'Formula':
        """Creates a Formula object from a dictionary."""
        return cls(**d)


@dataclass
class Document:
    """Represents a single document in the collection."""
    doc_id: str
    doi: Optional[str] = None
    url: Optional[str] = None
    pdf_path: Optional[str] = None
    text: Optional[str] = None
    tables: List[Table] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    formulas: List[Formula] = field(default_factory=list)
    chunks: List["Chunk"] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        """Converts the Document object to a JSON-serializable dictionary."""
        d = asdict(self)
        d['tables'] = [t.to_dict() for t in self.tables]
        d['figures'] = [f.to_dict() for f in self.figures]
        d['formulas'] = [f.to_dict() for f in self.formulas]
        d['chunks'] = [asdict(c) for c in self.chunks]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'Document':
        """Creates a Document object from a dictionary."""
        table_data = d.pop('tables', [])
        figure_data = d.pop('figures', [])
        formula_data = d.pop('formulas', [])
        chunk_data = d.pop('chunks', [])
        # Recreate the document with the remaining simple fields
        doc = cls(**d)
        # Reconstruct and assign the complex nested objects
        doc.tables = [Table.from_dict(t) for t in table_data]
        doc.figures = [Figure.from_dict(f) for f in figure_data]
        doc.formulas = [Formula.from_dict(f) for f in formula_data]
        doc.chunks = [Chunk(**c) for c in chunk_data]
        return doc


@dataclass
class Topic:
    """Represents a single topic (query)."""
    query_id: str
    query_text: str
    context: Optional[Dict] = field(default_factory=dict)
    rewritten_texts: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    def to_dict(self):
        """Converts the Document object to a JSON-serializable dictionary."""
        d = asdict(self)
        return d

@dataclass
class Chunk:
    """Represents a single chunk of content from a document (e.g., text, table)."""
    chunk_id: str
    doc_id: str
    text: str
    modality: str  # e.g., 'text', 'table'
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        """Converts the Document object to a JSON-serializable dictionary."""
        d = asdict(self)
        return d


@dataclass
class Judgement:
    """Represents a relevance judgement for a query-document pair."""
    query_id: str
    doc_id: str
    score: int
    source: str # e.g., 'human', 'synthetic-gpt4'
    chunk_id: Optional[str] = None

    
    def to_dict(self):
        """Converts the Document object to a JSON-serializable dictionary."""
        d = asdict(self)
        return d

@dataclass
class SearchResult:
    """Represents a single search result for a query."""
    doc_id: str
    score: float
    rank: int
    chunk_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class Ranking:
    """Represents a ranked list of search results for a query."""
    query_id: str
    results: List[SearchResult] = field(default_factory=list)

def save_judgements(judgements: List[Judgement], file_path: str):
    """
    Saves a list of Judgement objects to a JSON file.

    Args:
        judgements (List[Judgement]): The list of judgements to save.
        file_path (str): The path to the output JSON file.
    """
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump([asdict(j) for j in judgements], f, indent=4)

def load_judgements(file_path: str) -> List[Judgement]:
    """
    Loads a list of Judgement objects from a JSON file.

    Args:
        file_path (str): The path to the input JSON file.

    Returns:
        List[Judgement]: The loaded list of judgements.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [Judgement(**d) for d in data]

