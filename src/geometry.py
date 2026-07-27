"""Pure bounding-box geometry helpers, kept free of any OCR library import.

Split out from src/ocr_paddle.py so this logic stays testable even in
environments where paddlepaddle (a native dependency with limited Python
version support) cannot be installed.
"""


def bbox_from_polygon(polygon) -> list[float]:
    """Convert a 4-point polygon [[x,y], ...] into an axis-aligned [x_min, y_min, x_max, y_max] box."""
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


def filter_and_sort_regions(regions: list[dict], min_confidence: float) -> list[dict]:
    """Keep regions at or above min_confidence, sorted highest-confidence first."""
    kept = [region for region in regions if region["score"] >= min_confidence]
    return sorted(kept, key=lambda r: r["score"], reverse=True)
