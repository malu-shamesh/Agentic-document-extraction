"""Draw OCR and layout-detection bounding boxes onto images, saving to an output directory."""

from pathlib import Path

import cv2
import numpy as np
from matplotlib import colormaps


def draw_ocr_boxes(image_bgr: np.ndarray, ocr_items: list[dict], show_text: bool = True) -> np.ndarray:
    """Draw bounding boxes and (optionally) recognized text for each OCR item."""
    output = image_bgr.copy()
    for item in ocr_items:
        x_min, y_min, x_max, y_max = (int(c) for c in item["bbox"])
        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        if show_text:
            cv2.putText(output, item["text"], (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return output


def draw_layout_regions(image_bgr: np.ndarray, regions: list[dict]) -> np.ndarray:
    """Draw color-coded bounding boxes per layout region label."""
    output = image_bgr.copy()

    labels = sorted({region["label"] for region in regions})
    cmap = colormaps.get_cmap("tab20")
    colors = {
        label: tuple(int(c * 255) for c in cmap(i % 20)[:3][::-1])
        for i, label in enumerate(labels)
    }

    for region in regions:
        color = colors[region["label"]]
        x1, y1, x2, y2 = (int(c) for c in region["bbox"])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text = f"{region['label']} ({region['score']:.2f})"
        cv2.putText(output, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output


def save_image(image_bgr: np.ndarray, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), image_bgr)
    return output_path
