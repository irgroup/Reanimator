from typing import List, Optional
import pickle
import os
import json
import pkgutil
from .sources import CollectionSource
from .downloaders import PDFDownloader
from .extractors import BaseExtractor, DoclingExtractor
from .labelers import BaseLabeler, OpenAILabeler
from .models import Document, Judgement
from .retrieval import RetrievalPipeline, Chunker
from tqdm import tqdm
from docling.datamodel.accelerator_options import AcceleratorOptions

def _deep_merge(source, destination):
    """
    Recursively merges source dict into destination dict.
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # Get node or create one
            node = destination.setdefault(key, {})
            _deep_merge(value, node)
        else:
            destination[key] = value
    return destination

class Reanimator:
    """
    High-level orchestrator for the collection reanimation pipeline.
    """
    downloader: "PDFDownloader"
    extractor: "BaseExtractor"
    labeler: "BaseLabeler"
    chunker: "Chunker"
    retrieval_pipeline: "RetrievalPipeline"
    downloader_params: dict

    def __init__(self, irds_name: str, email: str, config: Optional[dict] = None):
        """
        Initializes the Reanimator pipeline.

        Args:
            irds_name (str): The ir-datasets name for the collection.
            email (str): Your email, for politeness with the Unpaywall API.
            config (dict, optional): Configuration for custom components. Defaults to None.
        """
        # Load default config from package
        default_config_bytes = pkgutil.get_data('reanimator', 'default_config.json')
        if not default_config_bytes:
            raise FileNotFoundError("Could not find default_config.json in the package.")
        
        self.config = json.loads(default_config_bytes)

        # Merge user-provided config over the default
        if config:
            self.config = _deep_merge(config, self.config)

        self.irds_name = irds_name
        self.email = email
        self.source = CollectionSource(irds_name=self.irds_name)

        self._initialize_components()

    def _initialize_components(self):
        """Initializes or re-initializes all components based on the current config."""
        paths_conf = self.config.get('paths')
        if not paths_conf:
            raise ValueError("Configuration error: 'paths' section is missing.")
        
        # Downloader
        downloader_conf = self.config.get('downloader', {})
        if isinstance(downloader_conf, PDFDownloader):
            self.downloader = downloader_conf
        else:
            params = downloader_conf.copy()
            if 'email' not in params:
                params['email'] = self.email
            
            try:
                init_params = {
                    'email': params.pop('email'),
                    'url_cache_path': paths_conf['url_cache'],
                    'output_dir': paths_conf['pdf_downloads']
                }
            except KeyError as e:
                raise ValueError(f"Configuration error: Missing required downloader parameter: {e}")
            
            self.downloader_params = params # Store remaining params like max_workers
            self.downloader = PDFDownloader(**init_params)

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
        retrieval_conf = self.config.get('retrieval', {}).copy()
        
        try:
            indexer_conf = retrieval_conf.get('indexer', {}).copy()
            indexer_conf['vector_store_path'] = paths_conf['vector_store']
            indexer_conf['bm25_path'] = paths_conf['bm25_index']
            retrieval_conf['indexer'] = indexer_conf
        except KeyError as e:
            raise ValueError(f"Configuration error: Missing path in 'paths' section: {e}")
        
        self.retrieval_pipeline = RetrievalPipeline(retrieval_conf)

    def set_config(self, new_config: dict):
        """
        Updates the Reanimator's configuration by merging the new config
        and then re-initializing all components.

        Args:
            new_config (dict): A dictionary with configuration updates.
        """
        self.config = _deep_merge(new_config, self.config)
        self._initialize_components()

    def set_parameter(self, key: str, value):
        """
        Updates a specific configuration parameter using a dot-separated key.
        Re-initializes all components after the update.

        Args:
            key (str): A dot-separated string indicating the parameter to update
                       (e.g., "labeler.model").
            value: The new value for the parameter.
        """
        keys = key.split('.')
        d = self.config
        for k in keys[:-1]:
            if k not in d or not isinstance(d.get(k), dict):
                raise KeyError(f"Invalid key '{key}'. Segment '{k}' is not a valid path in the configuration.")
            d = d[k]

        d[keys[-1]] = value
        self._initialize_components()

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

    def load_documents(self, max_docs: Optional[int] = None, doc_ids: Optional[List[str]] = None) -> List[Document]:
        """
        Loads documents from the source.
        
        Args:
            max_docs (int, optional): The maximum number of documents to process.
                                      Useful for testing. Defaults to all.
            doc_ids (List[str], optional): A list of document IDs to load.
                                           If provided, only these documents will be loaded.
        
        Returns:
            List[Document]: A list of document objects.
        """
        print("Step 1: Loading documents from source...")
        documents = list(self.source.get_documents())

        if doc_ids:
            doc_id_set = set(doc_ids)
            documents = [doc for doc in documents if doc.doc_id in doc_id_set]

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
        
        # Get fetch_urls and download_pdfs params from the config
        fetch_params = {k: v for k, v in self.downloader_params.items() if k in self.downloader.fetch_urls.__code__.co_varnames}
        download_params = {k: v for k, v in self.downloader_params.items() if k in self.downloader.download_pdfs.__code__.co_varnames}

        self.downloader.fetch_urls(documents, **fetch_params)
        self.downloader.download_pdfs(documents, **download_params)

    def generate_pdf_list(self,
                          documents: List[Document]) -> None:
        """
        Generates a list of PDF paths for the given documents.
        Only includes paths for PDFs that actually exist.
        """
        try:
            output_path = self.config['paths']['pdf_list']
        except KeyError:
            raise ValueError("Configuration error: 'paths.pdf_list' is not defined.")

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

        return documents, chunks


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Reanimate a collection.")
    parser.add_argument("irds_name", help="The ir-datasets name for the collection.")
    parser.add_argument("--email", help="Your email, for politeness with the Unpaywall API.", required=False)
    parser.add_argument("--config", help="Path to a JSON configuration file.")
    parser.add_argument("--max_docs", type=int, help="Maximum number of documents to process.")
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Email can be provided via CLI arg or config file. CLI takes precedence.
    email = args.email or config.get('email')
    if not email:
        raise ValueError("Email must be provided either via the --email argument or in the config file.")

    reanimator = Reanimator(irds_name=args.irds_name, email=email, config=config)
    reanimator.run(max_docs=args.max_docs)
