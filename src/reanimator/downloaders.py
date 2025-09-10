import os
import pickle
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from tqdm import tqdm
from .models import Document

class PDFDownloader:
    """
    Handles fetching PDF URLs and downloading the files.
    """
    def __init__(self, email: str, url_cache_path: str, output_dir: str):
        """
        Initializes the downloader.

        Args:
            email (str): Email address for the Unpaywall API.
            url_cache_path (str): Path to store the cached DOI-to-URL mapping.
            output_dir (str): Directory to save downloaded PDFs.
        """
        self.email = email
        self.url_cache_path = url_cache_path
        self.output_dir = output_dir
        os.makedirs(os.path.dirname(self.url_cache_path), exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.urls = self._load_urls()

    def _load_urls(self) -> Dict[str, str]:
        if os.path.exists(self.url_cache_path):
            # Avoid EOFError on empty file
            if os.path.getsize(self.url_cache_path) > 0:
                with open(self.url_cache_path, "rb") as f:
                    try:
                        return pickle.load(f)
                    except (pickle.UnpicklingError, EOFError):
                        # File is corrupt or empty, proceed with an empty cache
                        return {}
        return {}

    def _save_urls(self):
        with open(self.url_cache_path, "wb") as f:
            pickle.dump(self.urls, f)

    def _fetch_single_url(self, doi: str) -> Optional[str]:
        """Fetches a PDF URL for a single DOI from Unpaywall."""
        if not doi:
            return None
        try:
            time.sleep(0.05) # Politeness delay
            res = requests.get(f"https://api.unpaywall.org/v2/{doi}?email={self.email}", timeout=20)
            res.raise_for_status()
            res_dict = res.json()
            
            # Safely access the nested dictionary
            best_loc = res_dict.get("best_oa_location")
            if best_loc:
                return best_loc.get("url_for_pdf")
            
            return None
        except requests.exceptions.RequestException as e:
            print(f"URL fetch failed for {doi}: {e}")
            return None

    def fetch_urls(self, documents: List[Document], max_workers: int = 10):
        """
        Fetches PDF URLs for a list of documents in parallel.
        """
        dois_to_fetch = [doc.doi for doc in documents if doc.doi and doc.doi not in self.urls]
        if not dois_to_fetch:
            print("All DOIs already have cached URLs.")
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_doi = {executor.submit(self._fetch_single_url, doi): doi for doi in dois_to_fetch}

            for future in tqdm(as_completed(future_to_doi), total=len(dois_to_fetch), desc="Fetching URLs"):
                doi = future_to_doi[future]
                url = future.result()
                if url:
                    self.urls[doi] = url
        
        self._save_urls()
        print("URL fetching complete.")

    def _doi_to_path(self, doi: str, output_dir: str) -> str:
        safe_doi = doi.replace("/", "$")
        return os.path.join(output_dir, f"{safe_doi}.pdf")

    def _download_single_pdf(self, doc: Document) -> Optional[str]:
        """Downloads a single PDF if a URL is available."""
        # If DOI present, prefer cached URL; otherwise keep existing URL (e.g., arXiv)
        if doc.doi:
            doc.url = self.urls.get(doc.doi)
        if not doc.url:
            return None

       
        # Choose filename: DOI-based if available, else use doc_id (e.g., arXiv ID)
        if doc.doi:
            pdf_path = self._doi_to_path(doc.doi, self.output_dir)
        else:
            safe_id = str(doc.doc_id).replace("/", "$")
            pdf_path = os.path.join(self.output_dir, f"{safe_id}.pdf")
        doc.pdf_path = pdf_path

        if os.path.exists(pdf_path):
            return pdf_path

        try:
            response = requests.get(doc.url, timeout=60)
            response.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            return pdf_path
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {doc.doi}: {e}")
            doc.pdf_path = None # Unset path on failure
            return None

    def download_pdfs(self, documents: List[Document], max_workers: int = 15):
        """
        Downloads PDFs for a list of documents in parallel.
        """
        # Download if we have a DOI (to resolve URL via cache) or an explicit URL (e.g., arXiv)
        docs_to_download = [doc for doc in documents if doc.doi or doc.url]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_doc = {executor.submit(self._download_single_pdf, doc): doc for doc in docs_to_download}

            for future in tqdm(as_completed(future_to_doc), total=len(docs_to_download), desc="Downloading PDFs"):
                future.result() # The doc object is updated by reference
        
        print("PDF downloading complete.")
