from typing import List, Dict, Optional, Union
import os
import pickle
from collections import defaultdict
import langchain_core.documents
import nltk
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.tokenize import word_tokenize

from .models import Document, Topic, Chunk, Ranking, SearchResult
import pandas as pd
import pyterrier as pt

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

    def chunk(self, documents: List[Document], metadata_fields_to_chunk: Optional[List[str]] = None) -> List[Chunk]:
        """Chunks text and tables from documents, skipping any with duplicate IDs."""
        all_chunks = []
        seen_chunk_ids = set()

        for doc in documents:
            if metadata_fields_to_chunk:
                for field in metadata_fields_to_chunk:
                    if field in doc.metadata and doc.metadata[field]:
                        metadata_content = str(doc.metadata[field])
                        metadata_chunks = self.text_splitter.split_text(metadata_content)
                        for i, chunk_text in enumerate(metadata_chunks):
                            chunk_id = f"{doc.doc_id}-metadata-{field}-{i}"
                            if chunk_id in seen_chunk_ids:
                                continue
                            all_chunks.append(Chunk(
                                chunk_id=chunk_id,
                                doc_id=doc.doc_id,
                                text=chunk_text,
                                modality='metadata',
                                metadata={'source_field': field}
                            ))
                            seen_chunk_ids.add(chunk_id)

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

    def __init__(self, path: str, index_type: str = "both", max_docs: int = 200):
        if index_type not in ["vectorstore", "bm25", "both"]:
            raise ValueError("index_type must be one of 'vectorstore', 'bm25', or 'both'")
        self.index_type = index_type
        self.path = path
        self.vector_store_path = os.path.join(self.path, "vector_store")
        self.bm25_path = os.path.join(self.path, "bm25.pkl")
        self.vector_store = None
        self.bm25_retriever = None
        self.max_docs = max_docs

    def index(self, chunks: List[Chunk]):
        """Creates vector and/or BM25 indexes based on index_type."""
        
        should_load_vector = self.index_type in ['vectorstore', 'both'] and os.path.exists(self.vector_store_path)
        should_load_bm25 = self.index_type in ['bm25', 'both'] and os.path.exists(self.bm25_path)

        if should_load_vector or should_load_bm25:
            print("Indexes already exist. Loading from disk.")
            self.load()
        
        should_create_vector = self.index_type in ['vectorstore', 'both'] and not self.vector_store
        should_create_bm25 = self.index_type in ['bm25', 'both'] and not self.bm25_retriever
        
        if not should_create_vector and not should_create_bm25:
            return

        print("Creating new indexes...")
        lang_docs = [langchain_core.documents.Document(page_content=p.text, metadata={"chunk_id": p.chunk_id, "doc_id": p.doc_id, "modality": p.modality}) for p in chunks]

        if should_create_vector:
            embeddings = OpenAIEmbeddings()
            self.vector_store = Chroma.from_documents(documents=lang_docs, embedding=embeddings, persist_directory=self.vector_store_path)
        
        if should_create_bm25:
            self.bm25_retriever = BM25Retriever.from_documents(lang_docs, preprocess_func=word_tokenize, k=self.max_docs)
        
        self.save()
        print("Indexes created and saved.")

    def save(self):
        """Saves indexes to disk."""
        if self.vector_store and self.index_type in ['vectorstore', 'both']:
            # The Chroma client automatically persists data to disk, so a manual save is not needed.
            # We just need to ensure it's initialized with a persist_directory.
            pass
        if self.bm25_retriever and self.index_type in ['bm25', 'both']:
            os.makedirs(os.path.dirname(self.bm25_path), exist_ok=True)
            with open(self.bm25_path, "wb") as f:
                pickle.dump(self.bm25_retriever, f)
    
    def load(self):
        """Loads indexes from disk."""
        if self.index_type in ['vectorstore', 'both'] and os.path.exists(self.vector_store_path):
            embeddings = OpenAIEmbeddings()
            self.vector_store = Chroma(persist_directory=self.vector_store_path, embedding_function=embeddings)
        if self.index_type in ['bm25', 'both'] and os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                self.bm25_retriever = pickle.load(f)


class Retriever:
    """Component for retrieving chunks."""

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def retrieve(self, topic: Topic, k: Optional[int] = None) -> Dict[str, List[Ranking]]:
        """Retrieves chunks for a query and its variations."""
        
        retrieval_k = k if k is not None else self.indexer.max_docs

        use_dense = self.indexer.index_type in ['vectorstore', 'both']
        use_sparse = self.indexer.index_type in ['bm25', 'both']

        if use_dense and not self.indexer.vector_store:
            raise Exception("Vector store is not loaded. Please run indexer.index() or indexer.load() first.")
        if use_sparse and not self.indexer.bm25_retriever:
            raise Exception("BM25 retriever is not loaded. Please run indexer.index() or indexer.load() first.")

        all_rankings = {}
        
        queries_to_run = topic.rewritten_texts if topic.rewritten_texts else [topic.query_text]
        query_id = topic.query_id

        for q_text in queries_to_run:
            if use_dense:
                assert self.indexer.vector_store is not None
                dense_results = self.indexer.vector_store.similarity_search_with_score(q_text, k=retrieval_k)
                dense_ranking = [SearchResult(doc_id=doc.metadata['doc_id'], chunk_id=doc.metadata['chunk_id'], score=score, rank=i) for i, (doc, score) in enumerate(dense_results) if i < retrieval_k]
                all_rankings[f"dense_{q_text[:20]}"] = Ranking(query_id=query_id, results=dense_ranking)

            if use_sparse:
                assert self.indexer.bm25_retriever is not None
                sparse_results = self.indexer.bm25_retriever.invoke(q_text, k=retrieval_k)
                sparse_ranking = [SearchResult(doc_id=doc.metadata['doc_id'], chunk_id=doc.metadata['chunk_id'], score=0, rank=i) for i, doc in enumerate(sparse_results) if i < retrieval_k]
                all_rankings[f"sparse_{q_text[:20]}"] = Ranking(query_id=query_id, results=sparse_ranking)
            
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


def reciprocal_rank_fusion(
    rankings_list: List[Dict[str, Ranking]], 
    k: int = 60
) -> List[str]:
    """
    Fuses multiple rankings into a single list of unique doc_ids using Reciprocal Rank Fusion.

    Args:
        rankings_list: A list of ranking dictionaries. Each dictionary contains rankings
                       for a query, where keys are ranking names and values are Ranking objects.
        k (int): A constant used in the RRF formula to control the influence of lower-ranked items.

    Returns:
        A list of unique doc_ids, sorted by their fused RRF score in descending order.
    """
    rrf_scores = defaultdict(float)

    for ranking_dict in rankings_list:
        for ranking in ranking_dict.values():
            for result in ranking.results:
                # Ranks are 0-indexed, add 1 for 1-based rank in RRF formula
                rrf_scores[result.chunk_id] += 1 / (k + result.rank + 1)
    
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda chunk_id: rrf_scores[chunk_id], reverse=True)
    
    return sorted_doc_ids 




def judgements_to_df(judgements):
    data = []
    for j in judgements:
        # Extract needed fields and add as a tuple/list
        data.append([j.query_id, j.chunk_id, j.score])
    # Create DataFrame
    df = pd.DataFrame(data, columns=['qid', 'docno', 'label'])
    return df

def ranking_to_df(ranking):
    data = []
    ranking_dict = ranking.values()
    
    for ranking in ranking_dict:
        for result in ranking.results:
            data.append([ranking.query_id, result.chunk_id, result.score, result.rank])
    df = pd.DataFrame(data, columns=['qid', 'docno', 'score', 'rank'])
    return df

def topic_to_df(topics):
    data = []
    for topic in topics:
        data.append([topic.query_id, topic.query_text])
    df = pd.DataFrame(data, columns=['qid', 'query'])
    return df


def run_experiment(rankings, topics, judgements, eval_metrics, names):
    exp = pt.Experiment(
        retr_systems=[ranking_to_df(ranking) for ranking in rankings],
        topics=topic_to_df(topics),
        qrels=judgements_to_df(judgements),
        eval_metrics=eval_metrics,
        names=names
    )
    return exp