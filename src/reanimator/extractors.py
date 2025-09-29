from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict
import pandas as pd
from .models import Table, Figure, Formula, Document
import re
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
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import DocItemLabel, DoclingDocument, TextItem
import json
from dataclasses import asdict
from typing import List, Optional
from pathlib import Path


# =============================
# Helper Functions
# =============================
def _as_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

def _get_position_info(item: TextItem) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Extract position information (page, top, left, right, bottom) from a TextItem."""
    provs = _as_list(getattr(item, "prov", None))
    if not provs:
        return None, None, None, None, None
    
    try:
        prov = provs[0]
        page = getattr(prov, "page_no", None)
        bbox = getattr(prov, "bbox", None)
        
        if bbox:
            left, top, right, bottom = bbox.as_tuple()
            return page, top, left, right, bottom
        else:
            return page, None, None, None, None
    except Exception:
        return None, None, None, None, None

# =============================
# Formula Number Extraction
# =============================
def extract_formula_number_from_orig(orig_text: str) -> Optional[str]:
    """
    Extract formula number from the 'orig' field.
    """
    if not orig_text:
        return None
    
    orig_text = orig_text.strip()
    
    # Patterns in priority order
    patterns = [
        # Highest priority: parentheses at the end
        r'\((\d+(?:\.\d+)?[a-z]?)\)\s*$',
        r'\[(\d+(?:\.\d+)?[a-z]?)\]\s*$',
        
        # Equation labels at the end
        r'\bEq\.?\s*(\d+(?:\.\d+)?[a-z]?)\s*$',
        r'\bEqn\.?\s*(\d+(?:\.\d+)?[a-z]?)\s*$',
        r'\bEquation\s*(\d+(?:\.\d+)?[a-z]?)\s*$',
        
        # Any parentheses (lower priority)
        r'\((\d+(?:\.\d+)?[a-z]?)\)',
        r'\[(\d+(?:\.\d+)?[a-z]?)\]',
    ]
    
    best_match = None
    best_score = -1
    
    for i, pattern in enumerate(patterns):
        matches = list(re.finditer(pattern, orig_text, re.IGNORECASE))
        for match in matches:
            number = match.group(1).strip()
            
            # Score based on position and pattern type
            position_score = match.end() / max(1, len(orig_text))  # 0-1
            pattern_score = (len(patterns) - i) / len(patterns)    # 0-1
            
            total_score = position_score + pattern_score
            
            if total_score > best_score:
                best_match = number
                best_score = total_score
    
    return best_match

def clean_reference_text(text: str) -> str:
    """
    Clean reference text.
    """
    if not text:
        return ""
    
    # Remove leading/trailing junk
    text = re.sub(r'^\s*[\.…]+\s*', '', text)
    text = re.sub(r'\s*[\.…]+\s*$', '', text)
    
    # Fix common issues
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Ensure proper sentence casing and punctuation
    if text and len(text) > 1:
        # Capitalize first letter
        text = text[0].upper() + text[1:]
        
        # Ensure it ends with proper punctuation
        if not text[-1] in '.!?':
            text += '.'
    
    return text

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using robust sentence boundary detection.
    """
    sentences = []
    current_sentence = ""
    
    # Iterate through characters to build sentences
    for i, char in enumerate(text):
        current_sentence += char
        
        # Check for sentence endings
        if char in '.!?':
            # Look ahead to see if this is a real sentence boundary
            if i + 1 < len(text):
                next_char = text[i + 1]
                
                # Cases where period is NOT a sentence boundary:
                if (next_char.islower() or 
                    next_char.isdigit() or 
                    next_char in ('/', ')') or
                    current_sentence.strip().endswith(('Fig.', 'Eq.', 'Eqn.', 'etc.', 'i.e.', 'e.g.'))):
                    continue
            
            # This seems like a real sentence boundary
            sentence = current_sentence.strip()
            if sentence and len(sentence.split()) > 3:  # Minimum 4 words
                sentences.append(sentence)
            current_sentence = ""
    
    # Add the last sentence if any
    if current_sentence.strip() and len(current_sentence.strip().split()) > 3:
        sentences.append(current_sentence.strip())
    
    return sentences

def _score_sentence_relevance(sentence: str, formula_number: str) -> float:
    """
    Score a sentence based on its relevance to the formula reference.
    """
    score = 0.0
    sentence_lower = sentence.lower()
    
    # High scores for direct equation references
    direct_ref_patterns = [
        rf'\beq\.?\s*\(?\s*{re.escape(formula_number)}\s*\)',
        rf'\beqn\.?\s*\(?\s*{re.escape(formula_number)}\s*\)',
        rf'\bequation\s*\(?\s*{re.escape(formula_number)}\s*\)',
        rf'\(\s*{re.escape(formula_number)}\s*\)',
    ]
    
    for pattern in direct_ref_patterns:
        if re.search(pattern, sentence_lower):
            score += 5.0
    
    # Medium scores for equation references without parentheses
    medium_patterns = [
        rf'\beq\.?\s+{re.escape(formula_number)}\b',
        rf'\beqn\.?\s+{re.escape(formula_number)}\b',
        rf'\bequation\s+{re.escape(formula_number)}\b',
    ]
    
    for pattern in medium_patterns:
        if re.search(pattern, sentence_lower):
            score += 3.0
    
    # Bonus points for context words
    context_words = ['equation', 'formula', 'expression', 'model', 'parameter', 'fit', 'evidence']
    for word in context_words:
        if word in sentence_lower:
            score += 0.5
    
    # Penalize false positives
    false_positive_indicators = [
        rf'\bsection\s+{re.escape(formula_number)}\b',
        rf'\bchapter\s+{re.escape(formula_number)}\b',
        rf'\bfigure\s+{re.escape(formula_number)}\b',
        rf'\btable\s+{re.escape(formula_number)}\b',
        rf'\bpage\s+{re.escape(formula_number)}\b',
    ]
    
    for pattern in false_positive_indicators:
        if re.search(pattern, sentence_lower):
            score -= 10.0
    
    # Penalize very short or very long sentences
    word_count = len(sentence.split())
    if word_count < 5:
        score -= 2.0
    elif word_count > 40:
        score -= 1.0
    
    return max(0, score)

def find_formula_references_in_text(doc: DoclingDocument, formula_number: str, 
                                   formula_page: int) -> List[str]:
    """
    Find references to a formula number using paragraph-based approach.
    """
    if not formula_number:
        return []
    
    references = []
    found_paragraphs = set()
    
    # Patterns for equation references
    reference_patterns = [
        rf'\bEq\.?\s*\(?\s*{re.escape(formula_number)}\s*\)',
        rf'\bEqn\.?\s*\(?\s*{re.escape(formula_number)}\s*\)',
        rf'\bEquation\s*\(?\s*{re.escape(formula_number)}\s*\)',
        rf'\(\s*{re.escape(formula_number)}\s*\)',
        rf'\bEq\.?\s+{re.escape(formula_number)}(?=[\s\.,;\)\]])',
        rf'\bEqn\.?\s+{re.escape(formula_number)}(?=[\s\.,;\)\]])',
        rf'\bEquation\s+{re.escape(formula_number)}(?=[\s\.,;\)\]])',
    ]
    
    # First, collect paragraphs that contain references
    paragraphs_with_refs = []
    
    for item, _ in doc.iterate_items():
        if not isinstance(item, TextItem) or getattr(item, "label", None) == DocItemLabel.FORMULA:
            continue
            
        text = getattr(item, "text", "")
        if not text or len(text.strip()) < 20:
            continue
            
        # Get page information for filtering
        page_info = _get_position_info(item)
        item_page = page_info[0] if page_info else None
        
        # Filter by page proximity (±2 pages)
        if (item_page is not None and formula_page is not None and 
            abs(item_page - formula_page) > 2):
            continue
        
        # Check if this text contains a reference to our formula
        for pattern in reference_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                paragraphs_with_refs.append((item_page, text))
                break
    
    # Now process each paragraph to extract the best references
    for page, paragraph in paragraphs_with_refs:
        # Clean the paragraph
        paragraph = clean_reference_text(paragraph)
        
        # Split into sentences
        sentences = split_into_sentences(paragraph)
        
        # Score each sentence based on relevance to the formula reference
        scored_sentences = []
        for sentence in sentences:
            score = _score_sentence_relevance(sentence, formula_number)
            if score > 0:
                scored_sentences.append((score, sentence))
        
        # Sort by relevance score (highest first)
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Take the top 2-3 most relevant sentences from this paragraph
        top_sentences = [sentence for _, sentence in scored_sentences[:3]]
        
        if top_sentences:
            # Combine the top sentences into a coherent reference
            reference = " ".join(top_sentences)
            reference = clean_reference_text(reference)
            
            if len(reference) > 30 and reference not in found_paragraphs:
                found_paragraphs.add(reference)
                references.append(reference)
    
    return references[:4]

def extract_context_from_position(doc: DoclingDocument, pos_page: int, 
                                 pos_top: float, pos_bottom: float, 
                                 context_lines: int = 3) -> List[str]:
    """
    Extract context sentences based on formula position when number extraction fails.
    """
    if pos_page is None:
        return []
    
    context_sentences = []
    
    try:
        # Collect all text items on the same page
        page_items = []
        for item, _ in doc.iterate_items():
            if not isinstance(item, TextItem) or getattr(item, "label", None) == DocItemLabel.FORMULA:
                continue
            
            item_page, item_top, _, _, item_bottom = _get_position_info(item)
            if item_page == pos_page and item_top is not None:
                page_items.append((item_top, item_bottom, item))
        
        # Sort by vertical position
        page_items.sort(key=lambda x: x[0])
        
        # Find items near the formula position
        nearby_items = []
        for top, bottom, item in page_items:
            vertical_overlap = (top <= pos_bottom + (context_lines * 50) and 
                              bottom >= pos_top - (context_lines * 50))
            
            if vertical_overlap:
                nearby_items.append(item)
        
        # Extract text from nearby items and split into sentences
        all_text = ""
        for item in nearby_items:
            text = getattr(item, "text", "").strip()
            if text and len(text) > 10:
                all_text += " " + text
        
        if all_text:
            sentences = split_into_sentences(all_text)
            context_sentences = sentences[:4]
    
    except Exception as e:
        print(f"Error extracting context from position: {e}")
    
    return context_sentences

def get_formula_references(doc: DoclingDocument, formula_number: str, 
                          pos_page: int, pos_top: float, pos_bottom: float) -> List[str]:
    """
    Get references for a formula using progressive strategies.
    """
    references = []
    
    # Strategy 1: Pattern matching (always try first)
    if formula_number:
        text_references = find_formula_references_in_text(doc, formula_number, pos_page or 0)
        references.extend(text_references)
        if references:
            return references, "pattern_matching"
    
    # Strategy 2: Position-based context (fallback)
    context_sentences = extract_context_from_position(doc, pos_page, pos_top or 0, pos_bottom or 0)
    if context_sentences:
        references.extend(context_sentences)
        return references, "position_context"
    
    return references, "none"

# =============================
# Main Formula Collection Function
# =============================
def create_formula(doc: DoclingDocument) -> List[Formula]:
    """
    Collect all formulas from the document with improved reference extraction.
    """
    out: List[Formula] = []
    formula_count = 0
    
    print("\n=== Starting Formula Extraction ===")
    
    # First pass: collect all formulas
    formula_items = []
    for item, _lvl in doc.iterate_items():
        if isinstance(item, TextItem) and item.label == DocItemLabel.FORMULA:
            formula_items.append(item)
    
    print(f"Found {len(formula_items)} formulas in document")
    
    # Process each formula
    for item in formula_items:
        formula_count += 1
        
        # Get formula text and original text
        text = getattr(item, "text", "") or ""
        orig = getattr(item, "orig", "") or ""
        
        # Extract position information
        pos_page, pos_top, pos_left, pos_right, pos_bottom = _get_position_info(item)
        
        print(f"\nProcessing Formula #{formula_count}")
        print(f"  Original text: '{orig[:100]}{'...' if len(orig) > 100 else ''}'")
        print(f"  Page: {pos_page}")
        
        # Extract formula number
        formula_number = extract_formula_number_from_orig(orig)
        if not formula_number:
            formula_number = extract_formula_number_from_orig(text)
            if formula_number:
                print(f"  Fallback extraction from text: {formula_number}")
        
        print(f"  Extracted Number: {formula_number if formula_number else 'None'}")
        
        # Get references using progressive strategies
        references, strategy = get_formula_references(
            doc, formula_number, pos_page, pos_top, pos_bottom
        )
        
        # Print results based on strategy used
        if strategy == "pattern_matching":
            print(f"  ✓ Found {len(references)} references via pattern matching")
        elif strategy == "position_context":
            print(f"  ⚠ Found {len(references)} context sentences via position fallback")
        else:
            print("  ✗ No references found")
        
        # Print sample references
        if references:
            print(f"  Sample references:")
            for i, ref in enumerate(references[:2]):
                print(f"    {i+1}. {ref[:80]}...")
        
        # Create Formula object
        formula_id = f"formula_{formula_count}"
        out.append(Formula(
            id=formula_id,
            text=text,
            orig=orig,
            references=references,
            pos_page=pos_page,
            pos_top=pos_top,
            pos_left=pos_left,
            pos_right=pos_right,
            pos_bottom=pos_bottom
        ))
    
    print(f"\n=== Formula Extraction Complete ===")
    print(f"Total formulas extracted: {len(out)}")
    
    return out




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
                pipeline_options.do_formula_enrichment = True  # Enable formula extraction
                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options,
                        )
                    }
                )
            else:
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_formula_enrichment = True  # Enable formula extraction
                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options,
                        )
                    }
                )

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
            # 5. Extract formulas
            # ----------------------------------------------------------------------------------

            doc.formulas = []  # reset in case we are re‑processing
            formulas = create_formula(dl_doc)
            for idx, formula in enumerate(formulas):
                formula.id = f"{doc.doc_id}_formula_{idx + 1}"
                doc.formulas.append(formula)


        except Exception as exc:
            # ------------------------------------------------------------------------------
            # 5. Robust error handling – never crash the ingestion pipeline
            # ------------------------------------------------------------------------------
            print(f"[DoclingExtractor] failed for {doc.pdf_path}: {exc}")
            doc.text = None
            doc.tables = []
            doc.figures = []
            doc.formulas = []
