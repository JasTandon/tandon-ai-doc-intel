from __future__ import annotations

import argparse
import json
import csv
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path to ensure we can import the library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from tandon_ai_doc_intel.pipeline import DocumentPipeline
from tandon_ai_doc_intel.metrics import (
    compute_cer,
    compute_wer,
    aggregate_table_accuracy,
    evaluate_retrieval,
    compute_throughput,
)
from tandon_ai_doc_intel.embeddings import OpenAIEmbeddings, VectorStore

def load_ground_truth(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run document pipeline benchmarks.")
    parser.add_argument("--data-dir", type=str, required=True, help="Folder with PDFs/docs.")
    parser.add_argument("--output-csv", type=str, default="benchmark_results.csv")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--k", type=int, default=5, help="k for retrieval metrics")
    parser.add_argument("--api-key", type=str, help="OpenAI API Key", default=os.getenv("OPENAI_API_KEY"))
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_csv = Path(args.output_csv)
    
    # Initialize pipeline
    pipeline = DocumentPipeline(openai_api_key=args.api_key)
    
    rows: List[Dict[str, Any]] = []
    start_all = time.time()
    
    # Simple convention: PDFs in data_dir, ground truth in data_dir/gt/{stem}.json
    files = sorted(list(data_dir.glob("*.pdf")))
    
    if args.max_docs:
        files = files[: args.max_docs]
        
    print(f"Found {len(files)} documents to process.")
    
    for pdf_path in files:
        doc_id = pdf_path.stem
        gt_path = data_dir / "gt" / f"{doc_id}.json"
        gt = load_ground_truth(gt_path)
        
        print(f"Processing {pdf_path.name} ...")
        try:
            result = pipeline.process(str(pdf_path))
            
            # Basic fields
            row: Dict[str, Any] = {
                "doc_id": doc_id,
                "file_name": pdf_path.name,
                "source_size": result.metadata.get("source_size"),
                "validation_score": result.validation_score,
                "factuality_score": result.factuality_score,
                "est_cost_usd": result.cost_estimate_usd
            }
            
            # Timings
            for stage, secs in (result.processing_time_seconds or {}).items():
                row[f"time_{stage}"] = secs
                
            # Analytics (flatten key metrics)
            row["readability"] = result.readability_score
            row["sentiment"] = result.sentiment_polarity
            row["lexical_diversity"] = result.lexical_diversity
            row["info_density"] = result.info_density
            
            # Ground-truth-based metrics
            ref_text = gt.get("text")
            if ref_text:
                row["cer"] = compute_cer(ref_text, result.text or "")
                row["wer"] = compute_wer(ref_text, result.text or "")
            
            # Digital/scanned classification accuracy if label is present
            true_is_digital = gt.get("is_digital")
            if true_is_digital is not None:
                pred_is_digital = result.metadata.get("is_digital_pdf")
                row["is_digital_true"] = bool(true_is_digital)
                row["is_digital_pred"] = bool(pred_is_digital)
                row["classification_correct"] = (
                    bool(true_is_digital) == bool(pred_is_digital)
                )
                
            rows.append(row)
            
        except Exception as e:
            print(f"Failed to process {pdf_path.name}: {e}")
            
    total_time = time.time() - start_all
    throughput = compute_throughput(len(rows), total_time)
    
    print(f"\nProcessed {len(rows)} documents in {total_time:.2f}s "
          f"({throughput:.3f} docs/s)")
          
    if not rows:
        print("No results to write.")
        return

    # Write per-document CSV
    fieldnames = sorted({k for r in rows for k in r.keys()})
    
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Wrote results to {output_csv}")
    
    # Retrieval evaluation (if you have a query file)
    queries_path = data_dir / "queries.json"
    
    if queries_path.exists() and args.api_key:
        print("\nRunning retrieval evaluation...")
        with queries_path.open("r", encoding="utf-8") as f:
            qdata = json.load(f)
            
        queries = [q["query"] for q in qdata]
        relevant_ids = [q["relevant_ids"] for q in qdata]
        
        # We need to access the pipeline's components
        embedder = pipeline.embedding_provider
        # IMPORTANT: This assumes the vector store already has the documents indexed
        # from the processing loop above.
        # Since pipeline.process() calls embed_and_store(), they should be in there.
        vs = pipeline.vector_store
        
        try:
            retrieval_metrics = evaluate_retrieval(queries, relevant_ids, vs, embedder, k=args.k)
            print("Retrieval metrics:", retrieval_metrics)
            
            # Save retrieval metrics
            retrieval_csv = output_csv.with_name(f"{output_csv.stem}_retrieval.csv")
            with retrieval_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=retrieval_metrics.keys())
                writer.writeheader()
                writer.writerow(retrieval_metrics)
            print(f"Wrote retrieval metrics to {retrieval_csv}")
            
        except Exception as e:
            print(f"Retrieval evaluation failed: {e}")

if __name__ == "__main__":
    main()

