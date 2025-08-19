from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, TYPE_CHECKING
import pandas as pd
from .models import Table, Figure, Document
import re
import docling
import os
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import urllib.request as libreq
import xmltodict
import re
import time
from typing import Optional, Dict, Any
from urllib import request, parse, error

import easyocr
from docling.document_converter import DocumentConverter, PdfFormatOption
if TYPE_CHECKING:
    from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.datamodel.base_models import InputFormat, ConversionStatus

def create_table(tbl, idx, dl_doc, doc):
    """
    create Table Object from Docling table, index (table number in document) and Docling document.

    Args:
    tbl: Docling Table.
    idx: table index.
    dl_doc: Docling document containing tbl.
    doc: current Document for id creation.

    Returns:
    The extracted table name as Table Class Object.
    """
    try:
        df = tbl.export_to_dataframe()
    except Exception:  # pragma: no cover – fall‑back path rarely needed
        # *export_to_dataframe* may fail on exotic edge cases; fall back to
        # the raw 2‑D grid when that happens so callers still get the data.
        grid = getattr(tbl.data, "grid", [])
        df = pd.DataFrame(grid)

    try:
        pos_page = tbl.prov[0].page_no
        pos_left,pos_top,pos_right,pos_bottom = tbl.prov[0].bbox.as_tuple()
        table_id = f"{doc.doc_id}_table_{idx + 1}"

        try:
            caption = tbl.caption_text(dl_doc)
        except:
            return Table(id=table_id, content=df, pos_page=pos_page, pos_left=pos_left,  pos_top=pos_top, pos_right=pos_right, pos_bottom=pos_bottom)

        if (not caption) or (caption == "") or (caption is None):
            return Table(id=table_id, content=df, pos_page=pos_page, pos_left=pos_left,  pos_top=pos_top, pos_right=pos_right, pos_bottom=pos_bottom)

        name = extract_name(caption, type_extraction="table")
        if name and name is not None:
                references = find_mentions(dl_doc, name, caption, type_extraction="table")
        else:
            return Table(id=table_id, content=df, caption=caption, pos_page=pos_page, pos_left=pos_left,  pos_top=pos_top, pos_right=pos_right, pos_bottom=pos_bottom)

        return Table(id=table_id, content=df, caption=caption, name=name, references=references, pos_page=pos_page, pos_left=pos_left,  pos_top=pos_top, pos_right=pos_right, pos_bottom=pos_bottom)
    except Exception:
        print(f"error parsing table.")
        return None
    
def create_figure(fig, idx, dl_doc, doc):
    """
    create Figure Object from Docling figure, index (figure number in document) and Docling document.

    Args:
    fig: Docling Figure.
    idx: figure index.
    dl_doc: Docling document containing fig.
    doc: current Document for id creation.

    Returns:
    The extracted figure name as Figure Class Object.
    """
    try:
        pos_page = fig.prov[0].page_no
        pos_left,pos_top,pos_right,pos_bottom = fig.prov[0].bbox.as_tuple()
        figure_id = f"{doc.doc_id}_figure_{idx + 1}"

        try:
            caption = fig.caption_text(dl_doc) 
        except:
            return Figure(id=figure_id, pos_page=pos_page, pos_left=pos_left, pos_top=pos_top, pos_right=pos_right, pos_bottom=pos_bottom)
        
        if not caption or caption is "" or caption is None:
            return Figure(id=figure_id, pos_page=pos_page, pos_left=pos_left, pos_top=pos_top, pos_right=pos_right, pos_bottom=pos_bottom)
            

        name = extract_name(caption, type_extraction="figure")
        if name and name is not None:
                references = find_mentions(dl_doc, name, caption, type_extraction="figure")
        else:
            return Figure(id=figure_id, caption=caption, pos_page=pos_page, pos_left=pos_left, pos_top=pos_top, pos_right=pos_right, pos_bottom=pos_bottom)

        return Figure(id=figure_id, caption=caption, name=name, references=references, pos_page=pos_page, pos_left=pos_left, pos_top=pos_top, pos_right=pos_right, pos_bottom=pos_bottom)
    except Exception:
        print(f"error parsing figure.")
        return None
    

def extract_name(caption_text, type_extraction="table"):
    """
    Extracts the name of a table from a caption text.

    Args:
    caption_text: The full text of the table caption.

    Returns:
    The extracted table name as a string, or None if no table name is found.
    """
    if not caption_text:
        return None
    if type_extraction == "table":
        match = re.match(r'(?:Table|tab|tab\.)\s+[A-Za-z0-9]+', caption_text, re.IGNORECASE)
    elif type_extraction == "figure": 
        match = re.match(r'(?:Figure|fig|fig\.)\s+[A-Za-z0-9]+', caption_text, re.IGNORECASE)
    else:
        match = None
    if match:
        return match.group(0)
    else:
        return None


def find_mentions(doc, name: str, caption: str, type_extraction: str = "table") -> List[str]:
    """
    Finds sentences in a document that mention a specific table or figure by its name and number,
    excluding the table's or figure's caption.

    Args:
    doc: The document object with a 'texts' attribute containing document text elements.
    name: The name of the table or figure (e.g., "Table 1", "Tab. 22", "Tab X").
    caption: The full caption text of the table or figure.

    Returns:
    A list of sentences that mention the table or figure, excluding the caption itself.
    """
    try:
        type_extraction = type_extraction.lower()
        patterns = {
            "table":  r"(Table|Tab\.?|tab\.?)\s*([A-Za-z0-9]+)",
            "figure": r"(Figure|Fig\.?|fig\.?)\s*([A-Za-z0-9]+)",
        }
        
        if type_extraction not in patterns:
            raise ValueError(f"Unknown entity type: {type_extraction!r}")

        # ── parse the name ──────────────────────────────────────────────────────────
        m = re.match(patterns[type_extraction], name, flags=re.IGNORECASE)
        if not m:
            # Name doesn’t look like “Table 3” or “Fig. 2a”, bail out early.
            return []
        try:
            type_token, number = m.groups()
            number = re.escape(number)
        except:
            print(f"problem with splitting name and number: {name} --> {m}")
            return []

        # Build a pattern that catches the different abbreviations the text might use
        if type_extraction == "table":
            mention_re = re.compile(rf"(?:{type_token}|Table|Tab\.?|tab\.?)\s*{number}", re.IGNORECASE)
        else:  # figure
            mention_re = re.compile(rf"(?:{type_token}|Figure|Fig\.?|fig\.?)\s*{number}", re.IGNORECASE)

        # ── scan the document ──────────────────────────────────────────────────────
        mentions = []
        for element in getattr(doc, "texts", []):
            text = getattr(element, "text", "")
            if not text or text.strip() == caption.strip():
                continue

            sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s+", text)
            mentions.extend(s.strip() for s in sentences if mention_re.search(s))

        return mentions
    except Exception as exc:
        print(f"[find_mentions] unexpected error for {name!r}: {exc}")
        return []
    

_ARXIV_ID_RE = re.compile(
    r"""^(
        \d{4}\.\d{4,5}(v\d+)?                # new style: 2101.12345 or 2101.12345v2
        |
        [a-z\-]+(\.[a-z\-]+)?/\d{7}(v\d+)?   # old style: cs.CL/0301001 or cs/0301001v3
    )$""",
    re.IGNORECASE | re.VERBOSE,
)


def convert_arxiv_metadata_to_reanimator_schema(res: dict) -> dict:
    """
    Convert the arXiv metadata to the Reanimator schema.
    """
    ret = {
        "doi": res.get("arxiv:doi", {}).get("#text"),
        "url": res.get("id"),
        "summary": res.get("summary"),
        "metadata": {
            "updated": res.get("updated"),
            "published": res.get("published"),
            "title": res.get("title"),
            "summary": res.get("summary"),
            "author": res.get("author"),
            "category": res.get("category"),
            "arxiv:comment": res.get("arxiv:comment"),
            "arxiv:journal_ref": res.get("arxiv:journal_ref"),
            "arxiv:primary_category": res.get("arxiv:primary_category"),
        },
    }
    return ret

def get_arxiv_metadata(arxiv_id: str, *, timeout: float = 0.1, max_retries: int = 2) -> Optional[Dict[str, Any]]:
    """
    Fetch metadata for a single arXiv ID via the arXiv API.

    Returns:
        A dict representing the <entry> for the paper, or None if not found.

    Raises:
        ValueError: if `arxiv_id` looks invalid.
    """
    arxiv_id = arxiv_id.strip()
    if not _ARXIV_ID_RE.match(arxiv_id):
        raise ValueError(f"Invalid-looking arXiv ID: {arxiv_id!r}")

    base_url = "https://export.arxiv.org/api/query?id_list="
    url = base_url + parse.quote(arxiv_id)

    # arXiv asks for a descriptive User-Agent with contact info.
    # Replace the email with your own.
    headers = {
        "User-Agent": "arXivMetaFetcher/1.0 (mailto:your.email@example.com)"
    }

    for attempt in range(max_retries + 1):
        try:
            req = request.Request(url, headers=headers)
            with request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()

            results = xmltodict.parse(data.decode("utf-8"))
            feed = results.get("feed", {})
            entry = feed.get("entry")

            if not entry:
                # No result for this ID
                return None

            # For a single id, some parsers still return a list; normalize to dict
            if isinstance(entry, list):
                entry = entry[0]

            return convert_arxiv_metadata_to_reanimator_schema(entry)

        except error.HTTPError as e:
            # Retry on 5xx; fail fast on client errors
            if 500 <= e.code < 600 and attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise
        except (error.URLError, TimeoutError) as e:
            # Network hiccup: retry
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise

    # Shouldn't reach here
    return None



class BaseExtractor(ABC):
    """Abstract base class for all extractors."""
    @abstractmethod
    def extract(self, doc: Document, accelerator_options: Optional["AcceleratorOptions"] = None) -> None:
        """
        Extracts content from a document's file and populates the
        document object with the extracted text and tables.
        
        Args:
            doc (Document): The document to process. Its pdf_path must be set.
            accelerator_options (Optional[AcceleratorOptions]): The accelerator options to use for processing.
        """
        pass

class DoclingExtractor(BaseExtractor):
    """An extractor that leverages IBM's *Docling* toolkit to pull full text and
    structured tables out of PDF files.

    The heavy lifting is done by :class:`docling.document_converter.DocumentConverter`,
    which wraps Docling's standard PDF pipeline (layout analysis, OCR, TableFormer, …)
    behind a simple ``convert()`` call.  The resulting :class:`docling.DoclingDocument`
    exposes helpers like ``export_to_text()`` and ``tables[n].export_to_dataframe()``
    that we map to our internal :class:`Document` and :class:`Table` dataclasses.
    """

    #: Static converter instance – keeps the pipeline (and any downloaded models)
    #: cached across multiple ``extract`` calls, which makes large batch jobs
    #: significantly faster.
    _converter: Optional[DocumentConverter] = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def extract(self, doc: Document, accelerator_options: Optional["AcceleratorOptions"] = None) -> None:  # noqa: C901 – a bit long but readable
        """Populate *doc.text* and *doc.tables* by running Docling on ``doc.pdf_path``.

        All attributes are updated *in‑place*.  In case of any unrecoverable error
        the method leaves them at ``None`` / ``[]`` so that the calling pipeline
        can gracefully skip the document.
        """
        # ----------------------------------------------------------------------------------
        # 0. Sanity checks & early outs
        # ----------------------------------------------------------------------------------
        if not doc.pdf_path:
            # Nothing to do – the upstream crawler didn't locate a file
            return

        # Lazily create the converter the first time we are called.  Re‑using a single
        # instance avoids repeatedly downloading ML weights and re‑initialising the
        # PDF pipeline.
        if self._converter is None:
            if accelerator_options:
                pipeline_options = PdfPipelineOptions()
                pipeline_options.accelerator_options = accelerator_options
                pipeline_options.do_table_structure = True
                pipeline_options.table_structure_options.do_cell_matching = True
                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options,
                        )
                    }
                )
            else:
                self._converter = DocumentConverter()

        try:
            # ----------------------------------------------------------------------------------
            # 1. Let Docling do its magic
            # ----------------------------------------------------------------------------------
            conv_res = self._converter.convert(doc.pdf_path)
            if conv_res.status != ConversionStatus.SUCCESS:
                raise RuntimeError(
                    f"Docling conversion finished with status {conv_res.status}"
                )

            dl_doc = conv_res.document  # the rich DoclingDocument object

            # ----------------------------------------------------------------------------------
            # 2. Extract plain text
            # ----------------------------------------------------------------------------------
            doc.text = dl_doc.export_to_text()

            # ----------------------------------------------------------------------------------
            # 3. Extract tables
            # ----------------------------------------------------------------------------------
            doc.tables = []  # reset in case we are re‑processing
            for idx, tbl in enumerate(dl_doc.tables):
                # Convert to a Pandas DataFrame (keeps cell structure intact)
                
                tab_data = create_table(tbl, idx, dl_doc, doc)
                if tab_data:
                    doc.tables.append(tab_data)
            # ----------------------------------------------------------------------------------
            # 4. Extract figures
            # ----------------------------------------------------------------------------------
            doc.figures = []  # reset in case we are re‑processing
            for idx, fig in enumerate(dl_doc.pictures):
                
                fig_data = create_figure(fig, idx, dl_doc, doc)
                if fig_data:
                    doc.figures.append(fig_data)
            # ----------------------------------------------------------------------------------
            # 5. Extract arXiv metadata
            # ----------------------------------------------------------------------------------
            if _ARXIV_ID_RE.match(doc.doc_id):
                res = get_arxiv_metadata(doc.doc_id)
                doc["doi"] = res["doi"]
                doc["url"] = res["url"]
                doc["summary"] = res["summary"]
                doc["metadata"] = res["metadata"]
            

        except Exception as exc:
            # ------------------------------------------------------------------------------
            # 6. Robust error handling – never crash the ingestion pipeline
            # ------------------------------------------------------------------------------
            print(f"[DoclingExtractor] failed for {doc.pdf_path}: {exc}")
            doc.text = None
            doc.tables = []