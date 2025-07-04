from typing import List, Optional
import pickle
import os
import json
from .sources import CollectionSource
from .downloaders import PDFDownloader
from .extractors import BaseExtractor, DoclingExtractor
from .labelers import BaseLabeler, OpenAILabeler
from .models import Document, Judgement
from .retrieval import RetrievalPipeline, Chunker
from tqdm import tqdm
from docling.datamodel.accelerator_options import AcceleratorOptions

class Reanimator:
    """
    High-level orchestrator for the collection reanimation pipeline.
    """
    downloader: "PDFDownloader"
    extractor: "BaseExtractor"
    labeler: "BaseLabeler"
    chunker: "Chunker"
    retrieval_pipeline: "RetrievalPipeline"

    def __init__(self, irds_name: str, email: str, config: Optional[dict] = None):
        """
        Initializes the Reanimator pipeline.

        Args:
            irds_name (str): The ir-datasets name for the collection.
            email (str): Your email, for politeness with the Unpaywall API.
            config (dict, optional): Configuration for custom components. Defaults to None.
        """
        self.config = config if config else {}
        self.source = CollectionSource(irds_name=irds_name)
        
        # Allow for dependency injection or configuration of components
        
        # Downloader
        downloader_conf = self.config.get('downloader')
        if isinstance(downloader_conf, PDFDownloader):
            self.downloader = downloader_conf
        else:
            params = downloader_conf or {}
            if 'email' not in params:
                params['email'] = email
            self.downloader = PDFDownloader(**params)

        # Extractor
        extractor_conf = self.config.get('extractor')
        if isinstance(extractor_conf, BaseExtractor):
            self.extractor = extractor_conf
        else:
            params = extractor_conf or {}
            self.extractor = DoclingExtractor(**params)

        # Labeler
        labeler_conf = self.config.get('labeler')
        if isinstance(labeler_conf, BaseLabeler):
            self.labeler = labeler_conf
        else:
            params = labeler_conf or {}
            self.labeler = OpenAILabeler(**params)

        # Chunker
        chunker_conf = self.config.get('chunker', {})
        self.chunker = Chunker(**chunker_conf)

        # Retrieval Pipeline
        retrieval_conf = self.config.get('retrieval')
        self.retrieval_pipeline = RetrievalPipeline(retrieval_conf)

    def save_documents(self, documents: List[Document], dir_path: str):
        """Saves each Document object to a separate .json file in a directory."""
        print(f"Saving {len(documents)} documents to directory {dir_path}...")
        os.makedirs(dir_path, exist_ok=True)
        for doc in documents:
            file_path = os.path.join(dir_path, f"{doc.doc_id}.json")
            try:
                with open(file_path, "w") as f:
                    json.dump(doc.to_dict(), f, indent=4)
            except Exception as e:
                print(f"Could not save document {doc.doc_id} to {file_path}: {e}")
        print("Finished saving documents.")

    def load_documents_from_file(self, dir_path: Optional[str] = None, file_paths: Optional[List[str]] = None) -> List[Document]:
        """
        Loads Document objects from .json files.
        Provide either a directory path to load all documents within it,
        or a list of explicit file paths to load.
        """
        if not dir_path and not file_paths:
            raise ValueError("Must provide either 'dir_path' or 'file_paths'.")
        if dir_path and file_paths:
            raise ValueError("Cannot provide both 'dir_path' and 'file_paths' simultaneously.")

        documents = []
        paths_to_process = []

        if file_paths:
            paths_to_process = file_paths
        elif dir_path:
            if not os.path.isdir(dir_path):
                print(f"Error: Directory not found at {dir_path}")
                return []
            paths_to_process = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.json')]

        print(f"Attempting to load documents from {len(paths_to_process)} files...")
        for path in paths_to_process:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    documents.append(Document.from_dict(data))
            except FileNotFoundError:
                print(f"Warning: File not found at {path}, skipping.")
            except Exception as e:
                print(f"Could not load or unpickle document from {path}: {e}")
        
        print(f"Successfully loaded {len(documents)} documents.")
        return documents

    def load_documents(self, max_docs: Optional[int] = None) -> List[Document]:
        """
        Loads documents from the source.
        
        Args:
            max_docs (int, optional): The maximum number of documents to process.
                                      Useful for testing. Defaults to all.
        
        Returns:
            List[Document]: A list of document objects.
        """
        print("Step 1: Loading documents from source...")
        documents = list(self.source.get_documents())
        if max_docs:
            documents = documents[:max_docs]
        return documents

    def download_documents(self, documents: List[Document]):
        """
        Fetches URLs and downloads PDFs for the given documents.
        The document objects are updated in-place.
        
        Args:
            documents (List[Document]): A list of document objects.
        """
        print("\nStep 2: Fetching URLs and downloading PDFs...")
        self.downloader.fetch_urls(documents)
        self.downloader.download_pdfs(documents)

    def generate_pdf_list(self,
                          documents: List[Document],
                          output_path: str = "/workspace/data/pdf_list.txt") -> None:
        """
        Generates a list of PDF paths for the given documents.
        Only includes paths for PDFs that actually exist.
        """
        print(f"Generating PDF list at {output_path}...")
        pdf_paths = [doc.pdf_path for doc in documents if doc.pdf_path and os.path.exists(doc.pdf_path)]
        
        # Ensure directory for the list file exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            for path in pdf_paths:
                f.write(f"{path}\n")
        print(f"Found {len(pdf_paths)} existing PDFs. List saved to {output_path}.")

    def extract_content(self, documents: List[Document], accelerator_options: Optional[AcceleratorOptions] = None):
        """
        Extracts content from downloaded PDFs for the given documents.
        The document objects are updated in-place.

        Args:
            documents (List[Document]): A list of document objects.
        """
        print("\nStep 3: Extracting content from PDFs...")
        for doc in tqdm(documents, desc="Extracting Content"):
            self.extractor.extract(doc, accelerator_options)

    def generate_labels(self, documents: List[Document], sample_size: int = 5) -> List[Judgement]:
        """
        Generates synthetic labels for a sample of documents.
        
        Args:
            documents (List[Document]): A list of document objects.
            sample_size (int, optional): The number of documents to label. Defaults to 5.
        
        Returns:
            List[Judgement]: A list of synthetic relevance judgements.
        """
        print("\nStep 4: Generating synthetic labels...")
        topics = self.source.get_topics()
        
        synthetic_judgements = []
        # For demonstration, we'll just label the first few documents for the first query.
        if topics and documents:
            sample_topic = topics[0]
            docs_to_label = [doc for doc in documents if doc.text][:sample_size]
            
            for doc in tqdm(docs_to_label, desc=f"Labeling for Q:{sample_topic.query_id}"):
                judgement = self.labeler.label(sample_topic, doc)
                synthetic_judgements.append(judgement)

            print(f"Generated {len(synthetic_judgements)} judgements.")

        return synthetic_judgements

    def run(self, max_docs: Optional[int] = None):
        """
        Executes the full reanimation pipeline.

        Args:
            max_docs (int, optional): The maximum number of documents to process.
                                      Useful for testing. Defaults to all.
        """
        # 1. Load documents from the source
        documents = self.load_documents(max_docs=max_docs)
        
        # 2. Fetch URLs and download PDFs
        self.download_documents(documents)

        # 3. Extract content from downloaded PDFs
        self.extract_content(documents)

        # 4. Chunk documents
        print("\nStep 4: Chunking documents...")
        chunks = self.chunker.chunk(documents)
        print(f"Created {len(chunks)} chunks.")

        # 5. Generate synthetic labels
        synthetic_judgements = self.generate_labels(documents)

        # 6. Run Retrieval Pipeline (Indexing, Retrieval)
        print("\nStep 6: Running retrieval pipeline...")
        topics = self.source.get_topics()
        if topics:
            rankings = self.retrieval_pipeline.run(documents, topics, chunks)
        else:
            rankings = {}

        print("\nPipeline finished.")
        # The document objects in the list have been enriched in-place.
        return documents, synthetic_judgements, rankings

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reanimate a collection.")
    parser.add_argument("irds_name", help="The ir-datasets name for the collection.")
    parser.add_argument("--email", help="Your email, for politeness with the Unpaywall API.", required=True)
    parser.add_argument("--max_docs", type=int, help="Maximum number of documents to process.")
    args = parser.parse_args()

    reanimator = Reanimator(irds_name=args.irds_name, email=args.email)
    reanimator.run(max_docs=args.max_docs)
