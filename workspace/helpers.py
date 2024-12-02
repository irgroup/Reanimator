from papermage.magelib import Document, Box, Entity, Layer
import re
import math
from pathlib import Path
from typing import Optional, Union
import pymupdf

def positions_from_box(obj):
    return obj.boxes[0].l, obj.boxes[0].t, obj.boxes[0].w, obj.boxes[0].h, obj.boxes[0].page

def get_cleaned_captions(doc: Document) -> dict:

    return {
        "tables": [
            caption.id
            for page in doc.pages
            for caption in page.captions
            if text_matches_pattern(text=caption.text, layer="tables")
        ],
        "figures": [
            caption.id
            for page in doc.pages
            for caption in page.captions
            if text_matches_pattern(text=caption.text, layer="figures")
        ],
    }

def text_matches_pattern(text: str, layer: str = None) -> bool:
    if layer == "figures":
        return bool(re.search(r"^(fig.?)", text, flags=re.I))
    elif layer == "tables":
        return bool(re.search(r"^(tab.?)", text, flags=re.I))
    else:
        return bool(re.search(r"^(fig.?|tab.?)", text, flags=re.I))
    

def get_caption_for_box(box: Box, captions: Layer, caption_ids: Optional[list] = None) -> str:
    ref_center = get_center(box)
    closest = float("inf")
    _text = ""
    for cap in captions:
        if caption_ids:
            if cap.id not in caption_ids:
                continue
        if box.page == cap.boxes[0].page:
            center = get_center(cap.boxes[0])
            dist = math.dist(ref_center, center)
            if dist < closest:
                closest = dist
                _text = cap.text
    return _text

def get_center(box: Box):
    return box.l + box.w / 2, box.t + box.h / 2

def image_from_box(
    box: Box,
    pdf_path: Union[str, Path],
    im_path: Union[str, Path] = None,
    scale: pymupdf.Matrix = None,
    margin: tuple[float, float, float, float] = (0, 0, 0, 0),
) -> Optional[pymupdf.Pixmap]:
    """Takes a papermage box containing the bounding box of an entity and returns an image

    Args
    ----
    box: papermage.Box
        Bounding box of the entity.

    pdf_path: Union[str, Path]
        Path to the PDF file containing the entity.

    im_path: Union[str, Path] = None
        If None this method returns the pymupdf.Pixmap of the image if given image is saved to path.

    scale: pymupdf.Matrix = None
        Scale the pdf the image is taken from by this factor. Can in-, decrease the quality of the image.

    margin: tuple:[float, float, float, float] = None
        Additional margin in percentage added to the bounding box detected by Papermage. Format is (left, top, right, bottom).
    """
    pdf = pymupdf.open(str(pdf_path))
    page = pdf[box.page]
    rect = page.rect
    page_box = page.mediabox

    left = box.l * rect.br.x
    top = box.t * rect.br.y
    right = (box.l + box.w) * rect.br.x
    bottom = (box.t + box.h) * rect.br.y

    if left > right:
        raise ValueError("Left side of bounding box cannot have a greater X coordinate then right side")
    if bottom < top:
        raise ValueError("Bottom side of bounding box cannot have a smaller Y coordinate then top side.")

    width = right - left
    height = bottom - top

    expand_left = width * margin[0]
    expand_top = height * margin[1]
    expand_right = width * margin[2]
    expand_bottom = height * margin[3]

    tl = pymupdf.Point(max(left - expand_left, page_box[0]), max(top - expand_top, page_box[1]))
    br = pymupdf.Point(min(right + expand_right, page_box[2]), min(bottom + expand_bottom, page_box[3]))
    clip = pymupdf.Rect(tl, br)

    pix = page.get_pixmap(matrix=scale, clip=clip)
    if im_path:
        pix.save(im_path)
    else:
        return pix