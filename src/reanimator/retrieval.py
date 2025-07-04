from typing import List, Dict, Optional
import os
import pickle
import langchain_core.documents
import nltk
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.tokenize import word_tokenize

from .models import Document, Topic, Chunk, Ranking, SearchResult

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class Chunker:
    """Component for splitting documents into chunks."""

    def __init__(self, text_chunk_config: Optional[Dict] = None, table_chunk_config: Optional[Dict] = None):
        if text_chunk_config is None:
            text_chunk_config = {'chunk_size': 512, 'chunk_overlap': 100}
        if table_chunk_config is None:
            table_chunk_config = {'chunk_size': 8192, 'chunk_overlap': 1000}

        self.text_splitter = RecursiveCharacterTextSplitter(**text_chunk_config, add_start_index=True)
        self.table_splitter = RecursiveCharacterTextSplitter(**table_chunk_config, add_start_index=True)

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Chunks text and tables from documents, skipping any with duplicate IDs."""
        all_chunks = []
        seen_chunk_ids = set()

        for doc in documents:
            if doc.text:
                text_chunks = self.text_splitter.split_text(doc.text)
                for i, chunk_text in enumerate(text_chunks):
                    chunk_id = f"{doc.doc_id}-text-{i}"
                    if chunk_id in seen_chunk_ids:
                        continue
                    all_chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc.doc_id, text=chunk_text, modality='text'))
                    seen_chunk_ids.add(chunk_id)

            for table in doc.tables:
                table_content = table.content.to_string()
                table_chunks = self.table_splitter.split_text(f"Caption: {table.caption}\n\n{table_content}")
                for i, chunk_text in enumerate(table_chunks):
                    chunk_id = f"{doc.doc_id}-{table.id}-{i}"
                    if chunk_id in seen_chunk_ids:
                        continue
                    all_chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc.doc_id, text=chunk_text, modality='table'))
                    seen_chunk_ids.add(chunk_id)
        
        doc_map = {doc.doc_id: doc for doc in documents}
        # To prevent adding duplicates if chunk() is called multiple times, clear existing chunks first.
        for doc in documents:
            doc.chunks.clear()
            
        for chunk in all_chunks:
            if chunk.doc_id in doc_map:
                doc_map[chunk.doc_id].chunks.append(chunk)

        return all_chunks


class QueryRewriter:
    """Component for generating query variations."""

    def rewrite(self, query: Topic) -> List[str]:
        variations = [
            query.query_text,
            f"What is the answer to '{query.query_text}'?",
            f"Provide details about '{query.query_text}'.",
        ]
        query.rewritten_texts = variations
        return variations


class Indexer:
    """Component for creating search indexes."""

    def __init__(self, vector_store_path: str = "vectorstore", bm25_path: str = "bm25_retriever.pkl"):
        self.vector_store_path = vector_store_path
        self.bm25_path = bm25_path
        self.vector_store = None
        self.bm25_retriever = None

    def index(self, chunks: List[Chunk]):
        """Creates vector and BM25 indexes."""
        if os.path.exists(self.vector_store_path) or os.path.exists(self.bm25_path):
            print("Indexes already exist. Loading from disk.")
            self.load()
            return

        print("Creating new indexes...")
        lang_docs = [langchain_core.documents.Document(page_content=p.text, metadata={"chunk_id": p.chunk_id, "doc_id": p.doc_id, "modality": p.modality}) for p in chunks]

        embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma.from_documents(documents=lang_docs, embedding=embeddings, persist_directory=self.vector_store_path)
        self.bm25_retriever = BM25Retriever.from_documents(lang_docs, preprocess_func=word_tokenize)
        
        self.save()
        print("Indexes created and saved.")

    def save(self):
        """Saves indexes to disk."""
        if self.vector_store:
            # The Chroma client automatically persists data to disk, so a manual save is not needed.
            # We just need to ensure it's initialized with a persist_directory.
            pass
        if self.bm25_retriever:
            with open(self.bm25_path, "wb") as f:
                pickle.dump(self.bm25_retriever, f)
    
    def load(self):
        """Loads indexes from disk."""
        if os.path.exists(self.vector_store_path):
            embeddings = OpenAIEmbeddings()
            self.vector_store = Chroma(persist_directory=self.vector_store_path, embedding_function=embeddings)
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                self.bm25_retriever = pickle.load(f)


class Retriever:
    """Component for retrieving chunks."""

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def retrieve(self, query: Topic, k: int = 100) -> Dict[str, List[Ranking]]:
        """Retrieves chunks for a query and its variations."""
        if not self.indexer.vector_store or not self.indexer.bm25_retriever:
            raise Exception("Indexes are not loaded. Please run indexer.index() or indexer.load() first.")

        all_rankings = {}
        
        queries_to_run = query.rewritten_texts if query.rewritten_texts else [query.query_text]

        for q_text in queries_to_run:
            dense_results = self.indexer.vector_store.similarity_search_with_score(q_text, k=k)
            sparse_results = self.indexer.bm25_retriever.get_relevant_documents(q_text, k=k)

            dense_ranking = [SearchResult(doc_id=doc.metadata['doc_id'], chunk_id=doc.metadata['chunk_id'], score=score, rank=i) for i, (doc, score) in enumerate(dense_results)]
            sparse_ranking = [SearchResult(doc_id=doc.metadata['doc_id'], chunk_id=doc.metadata['chunk_id'], score=0, rank=i) for i, doc in enumerate(sparse_results)]

            all_rankings[f"dense_{q_text[:20]}"] = Ranking(query_id=query.query_id, results=dense_ranking)
            all_rankings[f"sparse_{q_text[:20]}"] = Ranking(query_id=query.query_id, results=sparse_ranking)
            
        return all_rankings


class RetrievalPipeline:
    """Orchestrates query rewriting, indexing, and retrieval for pre-chunked documents."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config if config else {}
        self.query_rewriter = QueryRewriter(**self.config.get('query_rewriter', {}))
        self.indexer = Indexer(**self.config.get('indexer', {}))
        self.retriever = Retriever(self.indexer)

    def run(self, documents: List[Document], topics: List[Topic], chunks: List[Chunk]) -> Dict[str, List[Ranking]]:
        """
        Executes the retrieval pipeline.
        
        1. Indexes the provided chunks.
        2. Rewrites queries.
        3. Retrieves chunks for each query.
        """
        print("Running retrieval pipeline...")
        
        print("\nStep 1: Indexing chunks...")
        self.indexer.index(chunks)
        print("Indexing complete.")

        print("\nStep 2: Retrieving chunks for queries...")
        final_rankings = {}
        for topic in topics:
            self.query_rewriter.rewrite(topic) 
            query_rankings = self.retriever.retrieve(topic)
            final_rankings[topic.query_id] = query_rankings
        
        print("\nRetrieval pipeline finished.")
        return final_rankings 