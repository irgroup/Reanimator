from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import pandas as pd
from .models import Table, Document
import docling
import os
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import easyocr
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.datamodel.base_models import InputFormat, ConversionStatus


class BaseExtractor(ABC):
    """Abstract base class for all extractors."""
    @abstractmethod
    def extract(self, doc: Document, accelerator_options: Optional[AcceleratorOptions] = None) -> None:
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
    def extract(self, doc: Document, accelerator_options: Optional[AcceleratorOptions] = None) -> None:  # noqa: C901 – a bit long but readable
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
                try:
                    df = tbl.export_to_dataframe()
                except Exception:  # pragma: no cover – fall‑back path rarely needed
                    # *export_to_dataframe* may fail on exotic edge cases; fall back to
                    # the raw 2‑D grid when that happens so callers still get the data.
                    grid = getattr(tbl.data, "grid", [])
                    df = pd.DataFrame(grid)

                caption: Optional[str] = getattr(tbl, "caption", None)
                table_id = f"{doc.doc_id}_table_{idx + 1}"
                doc.tables.append(Table(id=table_id, content=df, caption=caption))

        except Exception as exc:
            # ------------------------------------------------------------------------------
            # 4. Robust error handling – never crash the ingestion pipeline
            # ------------------------------------------------------------------------------
            print(f"[DoclingExtractor] failed for {doc.pdf_path}: {exc}")
            doc.text = None
            doc.tables = []