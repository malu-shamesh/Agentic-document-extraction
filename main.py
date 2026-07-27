"""CLI for agentic document extraction (Lab 1: Tesseract, Lab 2: PaddleOCR).

Usage:
    python main.py --image invoice.png --task "extract tax and total in JSON" --engine tesseract
    python main.py --image receipt.jpg --task "evaluate if the total is correct" --engine paddle
    python main.py --image report.png --engine paddle --layout-only  # just show detected regions
"""

import argparse
from pathlib import Path

import cv2
from dotenv import load_dotenv

from src.agent import build_agent, run_extraction
from src.ocr_paddle import detect_layout_regions, paddle_ocr_read_document
from src.ocr_tesseract import ocr_read_document
from src.visualize import draw_layout_regions, draw_ocr_boxes, save_image

TOOL_DESCRIPTIONS = {
    "tesseract": "OCR tool to extract raw text from images",
    "paddle": "PaddleOCR tool to extract raw text, bounding boxes for each text, and confidence score from images",
}
TOOLS = {"tesseract": ocr_read_document, "paddle": paddle_ocr_read_document}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic document extraction")
    parser.add_argument("--image", required=True, help="Path to the document image.")
    parser.add_argument("--task", default=None, help="Extraction task for the agent (required unless --layout-only).")
    parser.add_argument("--engine", choices=["tesseract", "paddle"], default="paddle")
    parser.add_argument(
        "--layout-only", action="store_true",
        help="Skip the agent and just run PaddleOCR layout detection, saving an annotated image.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.layout_only:
        if args.engine != "paddle":
            raise SystemExit("--layout-only requires --engine paddle")
        print(f"Detecting layout regions in {args.image}...")
        regions = detect_layout_regions(args.image)
        for region in regions:
            print(f"{region['label']:20} score: {region['score']:.3f}  bbox: {[int(c) for c in region['bbox']]}")

        image = cv2.imread(args.image)
        annotated = draw_layout_regions(image, regions)
        output_path = save_image(annotated, args.output_dir, f"{Path(args.image).stem}_layout.png")
        print(f"Saved annotated layout image to {output_path}")
        return

    if not args.task:
        raise SystemExit("--task is required unless --layout-only is set")

    tool = TOOLS[args.engine]
    print(f"Building {args.engine} agent...")
    agent_executor = build_agent(tool, TOOL_DESCRIPTIONS[args.engine])

    print(f"Running task: {args.task!r}")
    result = run_extraction(agent_executor, f"Please process the document at '{args.image}'. {args.task}")
    print("\nResult:\n", result)


if __name__ == "__main__":
    main()
