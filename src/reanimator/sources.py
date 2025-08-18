import pyterrier as pt
from typing import List, Generator
from .models import Document, Topic, Judgement
import pandas as pd

class CollectionSource:
    """
    Handles loading document and query collections using pyterrier.
    """
    def __init__(self, irds_name: str):
        """
        Initializes the source with a pyterrier dataset ID.

        Args:
            irds_name (str): The name of the dataset in ir-datasets.
        """
        if not pt.java.started():
            pt.init()
        self.dataset = pt.get_dataset(irds_name)

    def get_documents(self) -> Generator[Document, None, None]:
        """
        Yields all documents from the collection as Document objects.
        """
        for row in self.dataset.get_corpus_iter():
            yield Document(
                doc_id=row['docno'],
                doi=row.get('doi'),
                metadata=row
            )

    def get_topics(self) -> List[Topic]:
        """
        Returns all queries from the collection as Query objects.
        """
        return [
            Topic(query_id=str(row['qid']), query_text=str(row['title']), context={"description": row['description'], "narrative": row['narrative']})
            for _, row in self.dataset.get_topics().iterrows()
        ]

    def get_qrels(self) -> List[Judgement]:
        """
        Returns all relevance judgements from the collection.
        """
        return [
            Judgement(
                query_id=str(row['qid']),
                doc_id=str(row['docno']),
                score=int(row['label']),
                source='original'
            )
            for _, row in self.dataset.get_qrels().iterrows()
        ]

class ArxivSource:
    """
    Loads a set of arXiv IDs and exposes them as `Document`s whose `url` points
    to the arXiv PDF endpoint for each ID.
    """
    def __init__(self, arxiv_ids: List[str]):
        """
        Initializes the source with a list of arXiv identifiers (e.g., "2307.03569").

        Args:
            arxiv_ids (List[str]): List of arXiv IDs to load.
        """
        # Normalize IDs to strings and strip whitespace
        self.arxiv_ids = [str(aid).strip() for aid in arxiv_ids if str(aid).strip()]

    def get_documents(self) -> Generator[Document, None, None]:
        """
        Yields arXiv documents with their PDF URLs set.
        """
        for arxiv_id in self.arxiv_ids:
            # arXiv serves PDFs at https://arxiv.org/pdf/{id} (".pdf" suffix optional)
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            yield Document(
                doc_id=arxiv_id,
                doi=None,
                url=pdf_url,
                metadata={"source": "arxiv", "arxiv_id": arxiv_id}
            )

    def get_topics(self) -> List[Topic]:
        """arXiv source does not provide topics by default."""
        return []

    def get_qrels(self) -> List[Judgement]:
        """arXiv source does not provide qrels by default."""
        return []
