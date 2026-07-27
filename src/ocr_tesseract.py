"""Lab 1: Tesseract-based OCR tool for document text extraction.

Tesseract represents the traditional computer vision era of OCR - hand-engineered
features and procedural rules. It handles clean printed text well but struggles
with tables, handwriting, and low-quality scans (see README for examples).
"""

from langchain.tools import tool
from PIL import Image
import pytesseract


@tool
def ocr_read_document(image_path: str) -> str:
    """Reads an image from the given path and returns extracted text using OCR."""
    try:
        return pytesseract.image_to_string(Image.open(image_path))
    except Exception as e:
        return f"Error reading image: {e}"
