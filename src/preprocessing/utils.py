import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import FAISS, Chroma
from langchain_community.vectorstores import LanceDB
from langchain.embeddings.openai import OpenAIEmbeddings
import langchain_core.documents
import concurrent.futures
from tqdm import tqdm

from dotenv import dotenv_values, load_dotenv

def create_vectorstore(
    chunks: list,  # chunks to vectorize
    embedding_model: str = "text-embedding-3-small",  # embedding model
    db_backend: str = "chromadb",  # vectorstore backend
    save_path: str = "vectorstores",  # directory to save the vector store
):
    """
    Creates a vectorstore for both tables and text documents.

    Parameters:
        embedding_model (str): Embedding model to use. Default is "text-embedding-3-small".
        db_backend (str): Vector store backend ("faiss", "chromadb", "lancedb", or "pinecone"). Default is "faiss".
        save_path (str): Directory to save the vector store. Defaults to "vectorstores".
    Returns:
        vectorstore: The created vector store (FAISS, Chroma, LanceDB, or Pinecone).
    """
    load_dotenv()
    os.makedirs(save_path, exist_ok=True)


    # create the embedding object
    embedding = OpenAIEmbeddings(model=embedding_model)
    embedding.show_progress_bar=True

    lang_docs = [langchain_core.documents.Document(page_content=chunk.chunk_text, metadata={"id": chunk.id, "doi": chunk.doi, "chunk_type": chunk.chunk_type}) for chunk in chunks]


    # create the vectorstore 
    if db_backend.lower() == "faiss":
        vectorstore = FAISS.from_documents(documents=lang_docs, embedding=embedding)
        vectorstore.save_local(os.path.join(save_path, "text_plus_table_vectorstore"))

    elif db_backend.lower() == "chromadb":
        vectorstore = Chroma.from_documents(
            documents=lang_docs,
            embedding=embedding,
            persist_directory=os.path.join(save_path, "chromadb_store")
        )
    else:
        raise ValueError(f"Unsupported database backend: {db_backend}. Use 'faiss', 'chromadb', 'lancedb', or 'pinecone'.")
    
    #transform chunks to langchain docs

    return vectorstore


def load_vectorstore(vectorstore_path: str, embedding_model: str = "text-embedding-3-small"):
    """
    Loads a vectorstore from a given path.
    """
    load_dotenv()
    embedding = OpenAIEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory=vectorstore_path
    )

    return vectorstore


def process_topic(key, topic, pooling_results, judge_gpt):
    """Process a single topic and return the key and judgments."""

    if not key in pooling_results:
        return key, []
    query = topic["title"] + " " + topic["description"]
    candidates = [
        {"doc": {"segment": chunk.page_content}, "docid": chunk.metadata['id']} 
        for chunk in pooling_results[key]
    ]

    input_dict = {
        "query": {"text": query, "qid": key},
        "candidates": candidates
    }

    judgments = judge_gpt.judge(request_dict=input_dict)
    for i in range(len(judgments)):
        judgments[i]["docid"] = candidates[i]["docid"]

    return key, judgments


def parallel_process_topics(topics, pooling_results, judge_gpt, max_workers=20):
    all_judgments = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  # Adjust max_workers as needed
        # Submit all topics for parallel processing
        future_to_key = {
            executor.submit(process_topic, key, topic, pooling_results, judge_gpt): key 
            for key, topic in topics.items()
        }

        # Use tqdm to track progress
        for future in tqdm(concurrent.futures.as_completed(future_to_key), total=len(future_to_key), desc="Processing topics"):
            key = future_to_key[future]
            try:
                result_key, judgments = future.result()
                all_judgments[result_key] = judgments
            except Exception as e:
                print(f"Error processing {key}: {e}")

    return all_judgments


def gen_full_text_docling(row):
    return f"Title: {row[0]} \n\n {row[1]}"

def gen_table(row):
    table = ""
    table += f"Table Name: {row[0]} \n"
    table += f"Header: {row[1]} \n"
    table += f"Content: {row[2]} \n"
    table += f"Caption: {row[3]} \n"
    table += f"References: {row[4]} \n"
    return table