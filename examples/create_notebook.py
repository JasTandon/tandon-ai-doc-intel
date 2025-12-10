import json
import os

# Notebook content structure
notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Tandon AI Document Intelligence - Pipeline Demo\n",
    "\n",
    "This notebook demonstrates how to use the `tandon_ai_doc_intel` library to process a PDF document, extract insights, and explore the results programmatically."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "from pprint import pprint\n",
    "\n",
    "# Ensure we can import the local library\n",
    "sys.path.append(os.path.abspath(os.path.join('..', 'src')))\n",
    "\n",
    "from tandon_ai_doc_intel import DocumentPipeline, PipelineConfig"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Initialize the Pipeline\n",
    "\n",
    "We configure the pipeline to enable all features: OCR, LLM enrichment, and Analytics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set your API Key here or in environment variables\n",
    "api_key = os.getenv(\"OPENAI_API_KEY\", \"sk-...\")\n",
    "\n",
    "config = PipelineConfig(\n",
    "    enable_ocr=True,\n",
    "    enable_llm_enrichment=True,\n",
    "    enable_analytics=True,\n",
    "    use_caching=True\n",
    ")\n",
    "\n",
    "pipeline = DocumentPipeline(openai_api_key=api_key, config=config)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Process a Document\n",
    "\n",
    "Run the pipeline on a sample PDF."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Replace with path to your PDF\n",
    "pdf_path = \"../sample.pdf\" \n",
    "\n",
    "if os.path.exists(pdf_path):\n",
    "    result = pipeline.process(pdf_path)\n",
    "    print(\"Processing complete!\")\n",
    "else:\n",
    "    print(f\"File not found: {pdf_path}. Please place a sample PDF in the root directory.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Explore Results\n",
    "\n",
    "View the extracted text, summary, and analytics metrics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if 'result' in locals():\n",
    "    print(f\"--- Document Summary ---\\n{result.summary}\\n\")\n",
    "    print(f\"--- Key Metrics ---\")\n",
    "    print(f\"Readability Score: {result.readability_score}\")\n",
    "    print(f\"Factuality Score: {result.factuality_score}\")\n",
    "    print(f\"Est. Cost: ${result.cost_estimate_usd}\")\n",
    "    \n",
    "    if result.tables:\n",
    "        print(f\"\\n--- Tables Found: {len(result.tables)} ---\")\n",
    "        pprint(result.tables[0]['data'][:3]) # Show first 3 rows of first table"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

# Write to file
with open('examples/01_pipeline_demo.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

