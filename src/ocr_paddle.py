"""Lab 2: PaddleOCR-based OCR and layout detection.

PaddleOCR represents the deep-learning era of OCR: a text detection model locates
text regions, a text recognition model decodes their content, and a separate
layout detection model identifies region types (text, table, chart, footer, etc.)
before extraction - which fixes reading-order problems on multi-column documents
that plain OCR gets wrong.
"""

import os
from typing import Any

os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

from langchain.tools import tool
from paddleocr import LayoutDetection, PaddleOCR

from src.geometry import bbox_from_polygon, filter_and_sort_regions

_ocr_engine: PaddleOCR | None = None
_layout_engine: LayoutDetection | None = None


def get_ocr_engine() -> PaddleOCR:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(lang="en")
    return _ocr_engine


def get_layout_engine() -> LayoutDetection:
    global _layout_engine
    if _layout_engine is None:
        _layout_engine = LayoutDetection()
    return _layout_engine


@tool
def paddle_ocr_read_document(image_path: str) -> list[dict[str, Any]]:
    """
    Reads an image from the given path and returns extracted text with bounding boxes.

    Returns a list of dictionaries, each containing:
    - 'text': the recognized text string
    - 'bbox': bounding box coordinates [x_min, y_min, x_max, y_max]
    - 'confidence': recognition confidence score (if available)
    """
    try:
        result = get_ocr_engine().predict(image_path)
        page = result[0]

        texts = page["rec_texts"]
        boxes = page["dt_polys"]
        scores = page.get("rec_scores", [None] * len(texts))

        return [
            {"text": text, "bbox": bbox_from_polygon(box), "confidence": score}
            for text, box, score in zip(texts, boxes, scores)
        ]
    except Exception as e:
        return [{"error": f"Error reading image: {e}"}]


def detect_layout_regions(image_path: str, min_confidence: float = 0.5) -> list[dict[str, Any]]:
    """Detect document regions (text, table, chart, footer, ...) sorted by confidence."""
    layout_result = get_layout_engine().predict(image_path)

    regions = [
        {"label": box["label"], "score": box["score"], "bbox": box["coordinate"]}
        for box in layout_result[0]["boxes"]
    ]
    return filter_and_sort_regions(regions, min_confidence)
