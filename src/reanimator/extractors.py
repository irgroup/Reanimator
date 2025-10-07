from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict
import pandas as pd
import re
import spacy
from nltk.tokenize import sent_tokenize

from .models import Table, Figure, Formula, Document
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling_core.types.doc import DocItemLabel, DoclingDocument, TextItem

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================


# Load spaCy model if available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

# ============================================================================
# CORE TEXT PROCESSING FUNCTIONS
# ============================================================================

def _robust_sentence_split(text: str) -> List[str]:
    """
    Robust sentence splitting using spaCy's NLP pipeline with NLTK fallback.
    """
    if not text:
        return []
    
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    
    # Use spaCy if available
    if nlp is not None and len(text) > 10:
        try:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            
            meaningful_sentences = []
            for sentence in sentences:
                word_count = len(sentence.split())
                if word_count >= 4 and len(sentence) > 10:
                    meaningful_sentences.append(sentence)
            
            return meaningful_sentences
        except Exception:
            pass
    
    # Fallback to NLTK
    return _nltk_sentence_split(text)


def _nltk_sentence_split(text: str) -> List[str]:
    """
    NLTK-based sentence splitting fallback.
    """
    try:
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.split()) >= 4 and len(s.strip()) > 10]
    except Exception as e:
        print(f"NLTK sentence splitting failed: {e}")
        return []


def build_sentence_index(doc: DoclingDocument) -> List[Dict]:
    """
    Iterate each TextItem ONCE and build an index of sentences with page info.
    
    Returns:
        List[Dict] with keys: page, sentence, raw, label
    """
    idx = []
    for item, _ in doc.iterate_items():
        if not isinstance(item, TextItem):
            continue
        if getattr(item, "label", None) == DocItemLabel.FORMULA:
            continue

        text = getattr(item, "text", "") or ""
        if not text.strip():
            continue

        page, top, left, right, bottom = _get_position_info(item)
        for sent in _robust_sentence_split(text):
            idx.append({
                "page": page,
                "sentence": sent,
                "raw": text,
                "label": getattr(item, "label", None)
            })
    return idx

# ============================================================================
# ENTITY REFERENCE EXTRACTION
# ============================================================================

def _mentions_pattern_for_entity(entity_type: str, name_or_number: str) -> re.Pattern:
    """
    Create regex pattern for entity mentions in text.
    """
    entity_type = entity_type.lower()
    
    # Normalize: pull out the numeric token
    m = re.match(
        r'^(?:Table|Tab\.?|tab\.?|Figure|Fig\.?|fig\.?|Eqn?\.?|Equation)?\s*([A-Za-z0-9]+)$',
        name_or_number, re.IGNORECASE
    )
    token = m.group(1) if m else name_or_number

    if entity_type == "table":
        return re.compile(rf"\b(?:Table|Tab\.?|tab\.?)\s*{re.escape(token)}\b", re.IGNORECASE)
    if entity_type == "figure":
        return re.compile(rf"\b(?:Figure|Fig\.?|fig\.?)\s*{re.escape(token)}\b", re.IGNORECASE)
    if entity_type == "formula":
        return re.compile(
            rf"\b(?:Eq\.?|Eqn\.?|Equation)\s*\(?\s*{re.escape(token)}\s*\)?\b"
            rf"|(?<!Section\s)(?<!Figure\s)(?<!Table\s)\(\s*{re.escape(token)}\s*\)",
            re.IGNORECASE
        )
    
    return re.compile(r"$a")


def find_mentions_from_index(sentence_index: List[Dict], name: str, caption: str, entity_type: str) -> List[str]:
    """
    Use prebuilt sentence index to locate sentences that mention the entity.
    
    Returns:
        List of sentences that contain the mention (max 4)
    """
    if not name:
        return []

    pat = _mentions_pattern_for_entity(entity_type, name)
    cap_norm = (caption or "").strip()

    hits = []
    for row in sentence_index:
        sent = row["sentence"]
        if cap_norm and sent.strip() == cap_norm:
            continue
        if pat.search(sent):
            hits.append(sent.strip())

    # Dedupe and limit results
    seen, out = set(), []
    for s in hits:
        if s not in seen:
            seen.add(s)
            out.append(s)
    
    return out[:4]

# ============================================================================
# POSITION & TEXT UTILITIES
# ============================================================================

def _as_list(x):
    """Convert input to list if not None."""
    if x is None: 
        return []
    return x if isinstance(x, list) else [x]


def _get_position_info(item: TextItem) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Extract position information (page, top, left, right, bottom) from a TextItem.
    """
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

# ============================================================================
# FORMULA PROCESSING
# ============================================================================

def extract_formula_number(text: str) -> Optional[str]:
    """
    Extract formula number from text using prioritized patterns.
    """
    if not text:
        return None
    
    s = text.strip()
    patterns = [
        r'\((\d+(?:\.\d+)?[a-z]?)\)\s*$',    # (3), (2.1), (4a) at end
        r'\[(\d+(?:\.\d+)?[a-z]?)\]\s*$',    # [3] at end
        r'\bEq\.?\s*(\d+(?:\.\d+)?[a-z]?)\s*$', 
        r'\bEqn\.?\s*(\d+(?:\.\d+)?[a-z]?)\s*$',
        r'\bEquation\s*(\d+(?:\.\d+)?[a-z]?)\s*$',
        r'\((\d+(?:\.\d+)?[a-z]?)\)',        # any parens
        r'\[(\d+(?:\.\d+)?[a-z]?)\]',
    ]

    best, best_score = None, -1
    for i, pat in enumerate(patterns):
        for m in re.finditer(pat, s, re.IGNORECASE):
            num = m.group(1).strip()
            position_score = m.end() / max(1, len(s))
            pattern_score = (len(patterns) - i) / len(patterns)
            score = position_score + pattern_score
            if score > best_score:
                best, best_score = num, score
    
    return best


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
        
        # Extract text from nearby items
        all_text = ""
        for item in nearby_items:
            text = getattr(item, "text", "").strip()
            if text and len(text) > 10:
                all_text += " " + text
        
        if all_text:
            sentences = _robust_sentence_split(all_text)
            context_sentences = sentences[:4]
    
    except Exception as e:
        print(f"Error extracting context from position: {e}")
    
    return context_sentences


def create_formula(doc: Document, dl_doc: DoclingDocument, sentence_index: List[Dict] = None) -> List[Formula]:
    """
    Collect formulas and attach in-text references via the sentence index.
    Also save formula_number as 'name'.
    """
    out: List[Formula] = []
    formula_count = 0

    print("\n=== Starting Formula Extraction ===")

    # First pass: collect all formulas
    formula_items = []
    for item, _lvl in dl_doc.iterate_items():
        if isinstance(item, TextItem) and item.label == DocItemLabel.FORMULA:
            formula_items.append(item)

    print(f"Found {len(formula_items)} formulas in document")

    for item in formula_items:
        formula_count += 1
        text = getattr(item, "text", "") or ""
        orig = getattr(item, "orig", "") or ""
        pos_page, pos_top, pos_left, pos_right, pos_bottom = _get_position_info(item)

        print(f"\nProcessing Formula #{formula_count}")
        print(f"  Original text: '{orig[:100]}{'...' if len(orig) > 100 else ''}'")
        print(f"  Page: {pos_page}")

        # Extract formula number
        formula_number = extract_formula_number(orig)
        print(f"  Extracted Number: {formula_number if formula_number else 'None'}")

        # Find references via sentence index
        if formula_number and sentence_index is not None:
            references = find_mentions_from_index(sentence_index, formula_number, None, "formula")
            strategy = "pattern_matching"
        else:
            # Fallback: context near position
            references, strategy = [], "none"
            ctx = extract_context_from_position(dl_doc, pos_page, pos_top or 0, pos_bottom or 0)
            if ctx:
                references = ctx
                strategy = "position_context"

        # Log results
        if strategy == "pattern_matching":
            print(f"   Found {len(references)} references via sentence index")
        elif strategy == "position_context":
            print(f"   Found {len(references)} context sentences via position fallback")
        else:
            print("   No references found")

        if references:
            print("  Sample references:")
            for i, ref in enumerate(references[:2]):
                print(f"    {i+1}. {ref[:80]}...")
        
        formula_id = f"{doc.doc_id}_formula_{formula_count}"
        out.append(Formula(
            id=formula_id,
            text=text,
            orig=orig,
            name=formula_number,
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

# ============================================================================
# TABLE & FIGURE PROCESSING
# ============================================================================

def extract_name(caption_text: str, type_extraction: str = "table") -> Optional[str]:
    """
    Extract entity name from caption text.
    """
    if not caption_text:
        return None
    
    if type_extraction == "table":
        match = re.match(r'(?:Table|tab|tab\.)\s+[A-Za-z0-9]+', caption_text, re.IGNORECASE)
    elif type_extraction == "figure": 
        match = re.match(r'(?:Figure|fig|fig\.)\s+[A-Za-z0-9]+', caption_text, re.IGNORECASE)
    else:
        match = None
    
    return match.group(0) if match else None


def create_table(tbl, idx: int, dl_doc: DoclingDocument, doc: Document, sentence_index: List[Dict] = None) -> Optional[Table]:
    """
    Create Table Object from Docling table.
    """
    try:
        # Extract dataframe
        try:
            df = tbl.export_to_dataframe()
        except Exception:
            grid = getattr(tbl.data, "grid", [])
            df = pd.DataFrame(grid)

        # Extract position and metadata
        pos_page = tbl.prov[0].page_no
        pos_left, pos_top, pos_right, pos_bottom = tbl.prov[0].bbox.as_tuple()
        table_id = f"{doc.doc_id}_table_{idx + 1}"

        try:
            caption = tbl.caption_text(dl_doc)
        except:
            return Table(
                id=table_id, content=df, pos_page=pos_page, 
                pos_left=pos_left, pos_top=pos_top, pos_right=pos_right, pos_bottom=pos_bottom
            )

        if not caption:
            return Table(
                id=table_id, content=df, pos_page=pos_page,
                pos_left=pos_left, pos_top=pos_top, pos_right=pos_right, pos_bottom=pos_bottom
            )

        name = extract_name(caption, type_extraction="table")
        references = []
        
        if name and sentence_index is not None:
            m = re.match(r'(?:Table|Tab\.?|tab\.?)\s*([A-Za-z0-9]+)', name, re.IGNORECASE)
            if m:
                number = m.group(1)
                references = find_mentions_from_index(sentence_index, number, caption, entity_type="table")

        return Table(
            id=table_id, content=df, caption=caption, name=name, references=references,
            pos_page=pos_page, pos_left=pos_left, pos_top=pos_top, pos_right=pos_right, pos_bottom=pos_bottom
        )
    except Exception as e:
        print(f"{e} error parsing table.")
        return None


def create_figure(fig, idx: int, dl_doc: DoclingDocument, doc: Document, sentence_index: List[Dict] = None) -> Optional[Figure]:
    """
    Create Figure Object from Docling figure.
    """
    try:
        pos_page = fig.prov[0].page_no
        pos_left, pos_top, pos_right, pos_bottom = fig.prov[0].bbox.as_tuple()
        figure_id = f"{doc.doc_id}_figure_{idx + 1}"

        try:
            caption = fig.caption_text(dl_doc) 
        except:
            return Figure(
                id=figure_id, pos_page=pos_page, 
                pos_left=pos_left, pos_top=pos_top, pos_right=pos_right, pos_bottom=pos_bottom
            )
        
        if not caption:
            return Figure(
                id=figure_id, pos_page=pos_page,
                pos_left=pos_left, pos_top=pos_top, pos_right=pos_right, pos_bottom=pos_bottom
            )

        name = extract_name(caption, type_extraction="figure")
        references = []
        
        if name and sentence_index is not None:
            m = re.match(r'(?:Figure|Fig\.?|fig\.?)\s*([A-Za-z0-9]+)', name, re.IGNORECASE)
            if m:
                number = m.group(1)
                references = find_mentions_from_index(sentence_index, number, caption, entity_type="figure")

        return Figure(
            id=figure_id, caption=caption, name=name, references=references,
            pos_page=pos_page, pos_left=pos_left, pos_top=pos_top, pos_right=pos_right, pos_bottom=pos_bottom
        )
    except Exception as e:
        print(f"{e} error parsing figure.")
        return None

# ============================================================================
# MAIN EXTRACTOR CLASS
# ============================================================================

class BaseExtractor(ABC):
    """Abstract base class for all extractors."""
    
    @abstractmethod
    def extract(self, doc: Document, accelerator_options: Optional[AcceleratorOptions] = None) -> None:
        """
        Extract content from document and populate document object.
        
        Args:
            doc: The document to process. Its pdf_path must be set.
            accelerator_options: Accelerator options for processing.
        """
        pass


class DoclingExtractor(BaseExtractor):
    """
    Extractor that leverages IBM's Docling toolkit to extract text, tables, 
    figures, and formulas from PDF files.
    """

    _converter: Optional[DocumentConverter] = None

    def extract(self, doc: Document, accelerator_options: Optional[AcceleratorOptions] = None) -> None:
        """
        Populate document with extracted text, tables, figures, and formulas.
        
        All attributes are updated in-place. In case of unrecoverable error,
        method leaves them at None/[] so pipeline can gracefully skip document.
        """
        # ---------------------------------------------------------------------
        # 0. Sanity checks & early outs
        # ---------------------------------------------------------------------
        if not doc.pdf_path:
            return

        # Initialize converter if needed
        if self._converter is None:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
            pipeline_options.do_formula_enrichment = True
            
            if accelerator_options:
                pipeline_options.accelerator_options = accelerator_options

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

        try:
            # -----------------------------------------------------------------
            # 1. Let Docling do its magic
            # -----------------------------------------------------------------
            conv_res = self._converter.convert(doc.pdf_path)
            if conv_res.status != ConversionStatus.SUCCESS:
                raise RuntimeError(f"Docling conversion finished with status {conv_res.status}")

            dl_doc = conv_res.document

            # -----------------------------------------------------------------
            # 2. Extract plain text
            # -----------------------------------------------------------------
            doc.text = dl_doc.export_to_text()
            sentence_index = build_sentence_index(dl_doc)

            # -----------------------------------------------------------------
            # 3. Extract tables
            # -----------------------------------------------------------------
            doc.tables = []
            for idx, tbl in enumerate(dl_doc.tables):
                tab_data = create_table(tbl, idx, dl_doc, doc, sentence_index=sentence_index)
                if tab_data:
                    doc.tables.append(tab_data)

            # -----------------------------------------------------------------
            # 4. Extract figures
            # -----------------------------------------------------------------
            doc.figures = []
            for idx, fig in enumerate(dl_doc.pictures):
                fig_data = create_figure(fig, idx, dl_doc, doc, sentence_index=sentence_index)
                if fig_data:
                    doc.figures.append(fig_data)

            # -----------------------------------------------------------------
            # 5. Extract formulas
            # -----------------------------------------------------------------
            doc.formulas = []
            formulas = create_formula(doc, dl_doc, sentence_index=sentence_index)
            doc.formulas.extend(formulas)

        except Exception as exc:
            # -----------------------------------------------------------------
            # 6. Robust error handling
            # -----------------------------------------------------------------
            print(f"[DoclingExtractor] failed for {doc.pdf_path}: {exc}")
            doc.text = None
            doc.tables = []
            doc.figures = []
            doc.formulas = []
