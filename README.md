# Agentic Document Extraction

Extracts structured information from document images using OCR tools wired into a LangChain agent. Originally two Colab lesson notebooks (`ocr.ipynb`: Tesseract OCR, `L4.ipynb`: PaddleOCR + layout detection); rewritten as a script-based CLI. Four further lesson notebooks (LayoutReader reading-order resolution, LandingAI's commercial Agentic Document Extraction API, loan-document extraction, and RAG over parsed chunks) existed in this repo's git history but were deleted before this conversion and are out of scope here - see **Scope** below.

## Pipeline

1. **Lab 1 - Tesseract OCR** ([src/ocr_tesseract.py](src/ocr_tesseract.py)) - wraps `pytesseract` as a LangChain tool. Tesseract represents the traditional computer-vision era of OCR (hand-engineered rules): good on clean printed text, but misreads numbers, garbles tables, and fails on handwriting.
2. **Lab 2 - PaddleOCR + layout detection** ([src/ocr_paddle.py](src/ocr_paddle.py)) - wraps PaddleOCR (a deep-learning OCR engine returning text, bounding boxes, and confidence scores) as a LangChain tool, plus a separate layout-detection model that identifies region types (text/table/chart/footer) before extraction - fixing the reading-order errors plain OCR produces on multi-column documents.
3. **Bounding-box geometry** ([src/geometry.py](src/geometry.py)) - pure polygon-to-bbox conversion and confidence filtering/sorting, kept free of any OCR library import so it stays testable independent of whether PaddleOCR's native dependencies are installable (see **Known environment limitation** below).
4. **Agent** ([src/agent.py](src/agent.py)) - builds a LangChain tool-calling agent (`gpt-5-mini` by default) around either OCR tool. The LLM decides when to call the OCR tool, reasons over its raw output, and answers the extraction task - correcting some OCR noise (e.g. distinguishing subtotal from total) but still bounded by what the OCR tool actually read.
5. **Visualization** ([src/visualize.py](src/visualize.py)) - draws OCR/layout bounding boxes onto images and saves them, replacing the notebooks' inline matplotlib display calls.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Tesseract must also be installed as a system binary (`brew install tesseract` / `apt-get install tesseract-ocr` / [Windows installer](https://github.com/UB-Mannheim/tesseract/wiki)) - `pytesseract` is just a wrapper around it.

Set `OPENAI_API_KEY` (via `.env` or your shell environment) for the agent step.

## Usage

```bash
# Lab 1: Tesseract OCR + agent
python main.py --image invoice.png --task "extract tax and total in JSON format" --engine tesseract

# Lab 2: PaddleOCR + agent
python main.py --image receipt.jpg --task "evaluate if the total is correct" --engine paddle

# Lab 2: layout detection only, no LLM call - saves an annotated image
python main.py --image report.png --engine paddle --layout-only
```

Annotated images and other output land in `outputs/`.

## Scope

This repo's git history shows four more lesson notebooks that were added and later deleted: Lab 3 (LayoutReader for reading-order resolution + GPT-4o-mini vision tools for charts/tables), Lab 4 parts 1-2 (LandingAI's commercial Agentic Document Extraction API, plus a loan-document categorization/extraction/validation pipeline built on it), and Lab 5 (RAG over ADE-parsed chunks via ChromaDB). These were not rebuilt: Lab 4's core dependency, LandingAI's ADE API, requires a paid API key (`VISION_AGENT_API_KEY`) not available in this environment, and rebuilding Lab 3/5 without also covering Lab 4 would leave gaps in the arc. This conversion covers only what the repo currently contains: Lab 1 and Lab 2.

## Tests

```bash
pytest
```

## Known environment limitation

`paddlepaddle` (PaddleOCR's backing framework) has no published wheel for Python 3.14, so `src/ocr_paddle.py` cannot be imported on this machine's Python - the same class of issue as several other repos converted this session where a native ML dependency lags newer Python releases. Separately, and independent of that: with `langchain<1.0` installed on Python 3.14, `langchain.agents.agent` fails to import through a `pydantic`/`typing.Optional` forward-reference evaluation error inside `langchain.chains.base.Chain` - a real incompatibility between LangChain's legacy `Chain` base class and this Python version's `typing` module, not a bug in this codebase. As a result, `src/agent.py` (and therefore the full CLI) could not be smoke-tested end-to-end here either.

What *was* verified locally: `src/geometry.py` (bounding-box conversion, confidence filtering/sorting - the only PaddleOCR-adjacent logic with no native dependency) and `src/ocr_tesseract.py` (imports and tool registration succeed; `pytesseract`/`Pillow` are pure-Python-installable). Verify a full run - ideally on Python 3.11/3.12, where LangChain and PaddleOCR are both known to work - before relying on this code.
