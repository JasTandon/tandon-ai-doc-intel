# Tandon AI Document Intelligence

**An Unstructured Document Analytics Framework**

This library implements a modular, end-to-end pipeline for processing unstructured documents (PDFs). It moves beyond simple OCR by integrating automated classification, structured extraction (text & tables), LLM-powered enrichment (risk analysis, summarization), and quality validation.

Designed for high-compliance environments (Engineering, Legal, Finance) where data accuracy and semantic understanding are critical.

## Key Features

1.  **Intelligent Ingestion**: Automatically detects if a PDF is **Digital** (selectable text) or **Scanned** (image-based).
2.  **Hybrid Extraction**:
    *   **Digital**: Uses `PyMuPDF` for high-fidelity text extraction and `Camelot` for structured tables.
    *   **Scanned**: Routes through `Tesseract OCR` (pluggable with AWS/Azure) for image-to-text conversion.
3.  **LLM Enrichment**: Uses OpenAI (or other providers) to:
    *   Summarize content.
    *   Extract key entities (People, Orgs, Dates).
    *   Analyze potential Risks (Legal/Financial).
4.  **Quality Validation Loop**: Automatically scores extraction quality based on text density, OCR noise, and table confidence.
5.  **Vector Store Ready**: Generates embeddings (OpenAI) and stores chunked text in `ChromaDB` for semantic search.

---

## Installation

### Prerequisites
*   Python 3.9+
*   **System Dependencies**:
    *   `tesseract` (for OCR)
    *   `ghostscript` (required by Camelot)
    *   `tk` (required by Camelot)

#### macOS (Homebrew)
```bash
brew install tesseract ghostscript python-tk
```

#### Ubuntu/Debian
```bash
sudo apt-get install tesseract-ocr ghostscript python3-tk
```

### Install the Library
Clone the repository and install in editable mode:

```bash
pip install -e .
```

---

## Quick Start

### 1. Run the Web UI (No Coding Required)
The easiest way to explore the framework is using the included **Streamlit Dashboard**.

```bash
# Export your OpenAI API Key (or enter it in the UI sidebar)
export OPENAI_API_KEY="sk-..."

# Run the app
streamlit run app.py
```

This will launch a browser window where you can:
*   Upload a PDF.
*   See the **Validation Score** and **Risk Level**.
*   View interactive charts of extracted **Entities**.
*   Inspect extracted **Tables** and **Summaries**.

### 2. Use in Python Code

```python
import os
from tandon_ai_doc_intel import DocumentPipeline

# 1. Initialize Pipeline
pipeline = DocumentPipeline(openai_api_key="sk-...")

# 2. Process a Document
result = pipeline.process("invoice.pdf")

# 3. Access Insights
print(f"Validation Score: {result.validation_score}")
print(f"Risk Level: {result.risk_analysis['risk_level']}")
print(f"Summary: {result.summary}")

# 4. Access Structured Data
if result.tables:
    print(f"Found {len(result.tables)} tables.")
```

---

## Architecture Overview

The pipeline follows a strict "Production-Grade" flow:

1.  **Ingestion**: Normalizes input (bytes/path).
2.  **Classification**: `DocumentClassifier` checks text layer density.
3.  **Extraction**:
    *   Digital path -> `DigitalPDFExtractor` (Text + Tables).
    *   Scanned path -> `ScannedPDFExtractor` (OCR).
4.  **Enrichment**: `LLMEnricher` performs chunking, summarization, entity extraction, and risk analysis.
5.  **Validation**: `Validator` assigns a quality score and flags potential issues (e.g., "High OCR noise").
6.  **Storage**: Embeddings are generated and stored in a local `ChromaDB` vector store.

---

## Testing

Run the unit tests to verify your installation:

```bash
python -m unittest discover tests
```
