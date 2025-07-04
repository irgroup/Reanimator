import sys
sys.path.append('/workspace/src/')

import os
import re
import pandas as pd
from typing import List
import json

from umbrela.gpt_judge import GPTJudge

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import FAISS, Chroma
from langchain_community.vectorstores import LanceDB
from langchain.embeddings.openai import OpenAIEmbeddings
import langchain_core.documents
from langchain_community.docstore.in_memory import InMemoryDocstore

import concurrent.futures
from tqdm import tqdm

from database.chunk_model import Chunk_Base, Chunk

import pickle
import faiss
import pandas as pd

from sqlalchemy.orm import Session

from concurrent.futures import ThreadPoolExecutor, as_completed

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


def process_topic(key, topic, candidates, judge_gpt, safe_mode=False, verbose=False, MODEL_NAME="gpt-4o-mini"):
    """Process a single topic and return the key and judgments."""

    if safe_mode:
        #safe judgments as json file
        judge_gpt = GPTJudge(qrel="cord19", prompt_type="bing", engine=MODEL_NAME)
    
    query = topic["title"] + " " + topic["description"]

    input_dict = {
        "query": {"text": query, "qid": key},
        "candidates": candidates
    }
    judgments = judge_gpt.judge(request_dict=input_dict)
    for i in range(len(judgments)):
        judgments[i]["docid"] = candidates[i]["docid"]

    if safe_mode:
        #safe judgments as json file
        with open(f"judgments/{key}.json", "w") as f:
            json.dump(judgments, f)

        del judge_gpt
        return key, judgments
    else:
        return key, judgments


def parallel_process_topics_safe(topics, pooling_results, judge_gpt, max_workers=20, verbose=False, MODEL_NAME="gpt-4o-mini"):
    """
    Process topics in parallel using a ProcessPoolExecutor, skipping those that have already been processed.
    
    For each topic:
      - If judgments/{key}.json exists, load its content.
      - Otherwise, run process_topic (with safe_mode=True) in parallel so that it writes out its judgments.
    Finally, aggregate all judgments into one dictionary and return it.
    
    Note: Ensure that judge_gpt and pooling_results are picklable.
    """
    # Ensure the 'judgments' directory exists.
    judgments_dir = "judgments"
    if not os.path.exists(judgments_dir):
        os.makedirs(judgments_dir)
    
    # Dictionary to store aggregated results: key -> judgments
    aggregated_results = {}

    # Determine which topics still need to be processed.
    topics_to_process = []
    for key, topic in topics.items():
        judgment_file = os.path.join(judgments_dir, f"{key}.json")
        if os.path.exists(judgment_file):
            # Load judgments that have already been processed.
            try:
                with open(judgment_file, "r") as f:
                    aggregated_results[key] = json.load(f)
            except Exception as e:
                print(f"Error reading {judgment_file}: {e}")
                aggregated_results[key] = []
        else:
            topics_to_process.append((key, topic))

    # Process the remaining topics in parallel using ProcessPoolExecutor.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each topic for processing. The process_topic function, when called with safe_mode=True,
        # writes the judgments to a file.
        candidates_dict = {key: [{"doc": {"segment": chunk.page_content}, "docid": chunk.metadata['id']} 
                                 for chunk in pooling_results[key]]
                           for key in [tup[0] for tup in topics_to_process]}

        future_to_key = {
            executor.submit(process_topic, key, topic, candidates_dict[key], judge_gpt, True, verbose, MODEL_NAME): key
            for key, topic in topics_to_process
        }
        
        # Use tqdm to display a progress bar.
        for future in tqdm(concurrent.futures.as_completed(future_to_key),
                           total=len(future_to_key),
                           desc="Processing topics",
                           unit="topic"):
            key = future_to_key[future]
            try:
                # Since process_topic is in safe mode, it writes the file and does not return a value.
                future.result()
            except Exception as e:
                print(f"Error processing topic {key}: {e}")

    # Load the judgments from disk for the topics that were processed in this run.
    for key, _ in topics_to_process:
        judgment_file = os.path.join(judgments_dir, f"{key}.json")
        if os.path.exists(judgment_file):
            try:
                with open(judgment_file, "r") as f:
                    aggregated_results[key] = json.load(f)
            except Exception as e:
                print(f"Error reading {judgment_file} after processing: {e}")
                aggregated_results[key] = []
        else:
            print(f"Warning: Judgments file for topic {key} was not found!")
            aggregated_results[key] = []

    return aggregated_results



def single_process_topics(topics, pooling_results, judge):
    all_judgments = {}

    #process_topic(key, topic, candidates, judge_gpt, safe_mode=False, verbose=False, MODEL_NAME="gpt-4o-mini"):


    candidates_dict = {key: [{"doc": {"segment": chunk.page_content}, "docid": chunk.metadata['id']} 
                                 for chunk in pooling_results[key]]
                           for key in [tup for tup in topics]}
    

    for key, topic in topics.items():
        current_candidates = candidates_dict[key]
        _, judgments = process_topic(key, topic, current_candidates, judge)
        all_judgments[key] = judgments

    return all_judgments


def single_process_topics_safe(topics, pooling_results, judge, intermediate_file='judgments.json', skip_existing=True):
    # Load already processed judgments to avoid re-processing topics.
    if os.path.exists(intermediate_file):
        try:
            with open(intermediate_file, 'r') as f:
                all_judgments = json.load(f)
        except Exception as e:
            print("Error loading intermediate file, starting with an empty judgments dict:", e)
            all_judgments = {}
    else:
        all_judgments = {}

    # Build a dictionary of candidates for each topic.
    candidates_dict = {
        key: [
            {
                "doc": {"segment": chunk.page_content},
                "docid": chunk.metadata['id']
            }
            for chunk in pooling_results[key]
        ]
        for key in topics
    }

    # Process each topic.
    for key, topic in topics.items():
        # Skip this topic if it was already processed.
        if skip_existing and key in all_judgments:
            print(f"Skipping already processed topic: {key}")
            continue

        current_candidates = candidates_dict[key]
        try:
            # process_topic is assumed to return a tuple where the second element is judgments.
            _, judgments = process_topic(key, topic, current_candidates, judge)
            all_judgments[key] = judgments

            # Save intermediate results after each successful processing.
            with open(intermediate_file, 'w') as f:
                json.dump(all_judgments, f)
            print(f"Successfully processed and saved topic: {key}")

        except Exception as e:
            # Catch and log the error, then save intermediate results.
            print(f"Error processing topic {key}: {e}")
            with open(intermediate_file, 'w') as f:
                json.dump(all_judgments, f)
            # Optionally, you can also log the error details or perform other error handling here.
            continue

    return all_judgments




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

def parallel_process_topics_new(topics, pooling_results, judge_gpt, json_dir="results", max_workers=4):
    """
    Process topics in parallel using ThreadPoolExecutor and display progress using tqdm.
    For each topic, check if a JSON file already exists in the given directory.
    If not, process the topic, save its result, and then later aggregate all results.

    Args:
        topics (dict): A dictionary mapping keys to topic dictionaries.
        pooling_results (dict): Pooling results used by process_topic.
        judge_gpt: An object with a .judge() method to process topics.
        json_dir (str): Directory in which to store the per-topic JSON files.
        max_workers (int): Maximum number of worker threads to use.

    Returns:
        dict: A dictionary mapping topic keys to their judgments.
    """
    # Ensure the directory for saving JSON files exists.
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    def process_and_save(key, topic):
        """
        Process a single topic and save its result as a JSON file.
        Checks if the file already exists and skips processing if so.
        """
        file_path = os.path.join(json_dir, f"{key}.json")
        if os.path.exists(file_path):
            # Skip processing if the file already exists.
            return

        # Process the topic.
        processed_key, judgments = process_topic(key, topic, pooling_results, judge_gpt)
        # Write the judgments to a JSON file.
        with open(file_path, "w") as f:
            json.dump(judgments, f)

    # Submit tasks for topics that haven't been processed yet.
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for key, topic in topics.items():
            file_path = os.path.join(json_dir, f"{key}.json")
            if not os.path.exists(file_path):
                future = executor.submit(process_and_save, key, topic)
                futures[future] = key

        # Use tqdm to show progress for the submitted futures.
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing topics"):
            key = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Error processing topic {key}: {exc}")

    # After processing, load all JSON files and aggregate the results.
    all_judgments = {}
    for key in topics:
        file_path = os.path.join(json_dir, f"{key}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                judgments = json.load(f)
            all_judgments[key] = judgments
        else:
            # If the file doesn't exist, assume empty judgments.
            all_judgments[key] = []

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

def chunk_list(lst: List[int], chunk_size: int) -> List[List[int]]:
    """
    Splits a list of integers into chunks of a given size.
    
    :param lst: List of integers to be chunked
    :param chunk_size: Size of each chunk
    :return: List of chunks (sublists)
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")
    
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def load_chunk_by_ids(chunk_session, chunk):

    docs = chunk_session.query(Chunk).filter(Chunk.id.in_(chunk)).all()
    return docs

def get_unique_ids_from_pickles(folder_path):
    unique_ids = set()
    
    for file in tqdm(os.listdir(folder_path)):
        if file.endswith(".pkl"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_pickle(file_path)
            if 'ID' in df.columns:
                unique_ids.update(df['ID'].unique())
    
    return set(unique_ids)

def get_missing_ids(chunk_session):
    print("get all ids from database")
    chunk_ids = chunk_session.query(Chunk.id).distinct().all()
    chunk_ids = set([id[0] for id in chunk_ids])

    print("get processed ids")
    processsed_ids = get_unique_ids_from_pickles("/workspace/src/data/embeddings")
    missing_ids = chunk_ids.difference(processsed_ids)
    return list(missing_ids)

def save_df(df):
    name = f"{df.iloc[0]['ID']}_{df.iloc[-1]['ID']}.pkl"
    path = f"/workspace/src/data/embeddings/{name}"
    pickle.dump(df, open(path, "wb"))


def get_embeddings(client, texts, model="text-embedding-3-small"):
    """Fetch embeddings for a list of texts in a single API request."""
    texts = [text.replace("\n", " ") for text in texts]  # Clean text
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

def chunk_ids_to_embedding(chunk_session, client, chunk):
    """Load document chunks, compute embeddings, and save the results."""
    docs = load_chunk_by_ids(chunk_session, chunk)
    
    # Create DataFrame
    df = pd.DataFrame({
        "ID": [doc.id for doc in docs],
        "Text": [doc.chunk_text for doc in docs]
    })
    
    # Compute embeddings in a single batch request
    df["embeddings"] = get_embeddings(client, df["Text"].tolist())

    # Drop text column to save space
    df = df.drop(columns=["Text"])
    
    # Save DataFrame
    save_df(df)

def process_chunk(chunk_session, client, chunk):
    return chunk_ids_to_embedding(chunk_session, client, chunk)

def parallel_process_chunks(chunks, chunk_session, client, max_workers=4):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(process_chunk, chunk_session, client, chunk): chunk for chunk in chunks}
        
        with tqdm(total=len(chunks), desc="Processing Chunks") as pbar:
            for future in as_completed(future_to_chunk):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                pbar.update(1)
    
    return results


def remove_enumeration(questions):
    """Removes enumeration (number followed by dot and space) from a list of questions."""
    return [re.sub(r'^\d+\.\s*', '', question).strip() for question in questions]


def add_df_to_vectorstore(vectorstore, df, chunk_session: Session):
    """
    Adds a DataFrame containing precomputed embeddings to the vector store.
    
    :param vectorstore: The vector store object
    :param df: DataFrame containing IDs and embeddings
    :param chunk_session: Database session for querying document chunks
    :return: None
    """
    # Build a dictionary for O(1) lookups: {chunk_id: embedding}
    embedding_dict = dict(zip(df['ID'].values, df['embeddings'].values))

    # Fetch only the relevant chunks from the database in one go
    doc_ids = [int(id) for id in embedding_dict.keys()]
    docs = chunk_session.query(Chunk).filter(Chunk.id.in_(doc_ids)).all()

    # Create LangChain Documents and gather embeddings in a single pass
    input_texts = []
    intput_metadatas = []
    pre_computed_embeddings = []
    for doc in docs:
        embedding_value = embedding_dict.get(doc.id)
        if embedding_value is not None:

            input_texts.append(doc.chunk_text)
            intput_metadatas.append({
                        "id": doc.id,
                        "doi": doc.doi,
                        "chunk_type": doc.chunk_type
                    })
            pre_computed_embeddings.append(embedding_value)

    # Add documents to the vector store if we have any
    if input_texts:
        vectorstore._collection.add(
            documents=input_texts,
            metadatas=intput_metadatas,
            embeddings=pre_computed_embeddings,
            ids = [str(doc["id"]) for doc in intput_metadatas]
            
        )

def process_pickle_embeddings_in_chunks(
    vectorstore,
    directory: str,
    chunk_session: Session,
    chunk_size: int = 100
):
    """
    Reads pickle files from a directory in chunks and adds them to the vector store.
    
    :param vectorstore: The vector store object
    :param directory: Path to the directory containing pickle files
    :param chunk_session: Database session for querying document chunks
    :param chunk_size: Number of pickle files to accumulate before pushing to the vector store
    :return: None
    """
    current_dfs = []
    count = 0

    for filename in tqdm(os.listdir(directory)):
        file_path = os.path.join(directory, filename)
        
        # Load one pickle file
        df = pickle.load(open(file_path, "rb"))
        current_dfs.append(df)
        count += 1

        # Once we've hit our chunk limit, concatenate and add to vector store
        if count == chunk_size:
            combined_df = pd.concat(current_dfs, ignore_index=True)
            add_df_to_vectorstore(vectorstore, combined_df, chunk_session)
            
            # Reset for the next chunk
            current_dfs = []
            count = 0

    # Handle any remaining DataFrames
    if current_dfs:
        combined_df = pd.concat(current_dfs, ignore_index=True)
        add_df_to_vectorstore(vectorstore, combined_df, chunk_session)

def load_pool_documents(pool, vectorstore):
    documents = {}
    for key, id_list in pool.items():
        vs_entries = vectorstore._collection.get(ids = [str(id) for id in id_list])
        lang_docs = []
        for i in range(len(vs_entries['documents'])):
            vs_entry = vectorstore._collection.get(ids = [str(id_list[i])])
            lang_docs.append(langchain_core.documents.Document(page_content=vs_entry['documents'][0], 
                                                       metadata={"id": vs_entry['metadatas'][0]['id'], 
                                                                 "doi": vs_entry['metadatas'][0]['doi'], 
                                                                 "chunk_type": vs_entry['metadatas'][0]['chunk_type']}))
            
        documents[key] = lang_docs

    return documents