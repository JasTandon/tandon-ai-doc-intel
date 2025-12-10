from __future__ import annotations

import argparse
import json
import csv
import time
import os
import sys
import yaml
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

def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as f:
        if manifest_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        return json.load(f)

def load_ground_truth(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run document pipeline benchmarks.")
    parser.add_argument("--data-dir", type=str, required=True, help="Folder with PDFs/docs.")
    parser.add_argument("--manifest", type=str, help="Path to dataset manifest (JSON/YAML).")
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
    
    # Files to process
    files_to_process = []
    manifest_data = {}
    
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_data = load_manifest(manifest_path)
        # Assuming manifest has list of documents
        if 'documents' in manifest_data:
            for doc_entry in manifest_data['documents']:
                fpath = data_dir / doc_entry['file']
                if fpath.exists():
                    files_to_process.append((fpath, doc_entry))
        else:
             print("Manifest missing 'documents' key. Falling back to glob.")
             files = sorted(list(data_dir.glob("*.pdf")))
             files_to_process = [(f, {}) for f in files]
    else:
        files = sorted(list(data_dir.glob("*.pdf")))
        files_to_process = [(f, {}) for f in files]
    
    if args.max_docs:
        files_to_process = files_to_process[: args.max_docs]
        
    print(f"Found {len(files_to_process)} documents to process.")
    
    total_cost = 0.0
    
    for pdf_path, manifest_entry in files_to_process:
        doc_id = manifest_entry.get('id', pdf_path.stem)
        
        # Try to find ground truth: 
        # 1. In manifest entry ('gt_text')
        # 2. In separate file data_dir/gt/{doc_id}.json
        
        gt = {}
        if 'gt_text' in manifest_entry:
            gt['text'] = manifest_entry['gt_text']
        else:
            gt_path = data_dir / "gt" / f"{doc_id}.json"
            gt = load_ground_truth(gt_path)
        
        if 'is_digital' in manifest_entry:
            gt['is_digital'] = manifest_entry['is_digital']

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
                "est_cost_usd": result.cost_estimate_usd,
                "input_tokens": result.token_usage.get("llm_input", 0),
                "output_tokens": result.token_usage.get("llm_output", 0)
            }
            
            total_cost += result.cost_estimate_usd
            
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
    
    # --- Aggregated Statistics ---
    agg_stats = {
        "total_docs": len(rows),
        "total_time_s": total_time,
        "throughput_docs_s": throughput,
        "total_cost_usd": total_cost,
        "mean_cost_per_doc": total_cost / len(rows) if rows else 0,
        "mean_validation_score": sum(r.get("validation_score", 0) for r in rows) / len(rows) if rows else 0,
        "mean_factuality_score": sum(r.get("factuality_score", 0) for r in rows) / len(rows) if rows else 0,
        "mean_readability": sum(r.get("readability", 0) for r in rows) / len(rows) if rows else 0,
        "mean_input_tokens": sum(r.get("input_tokens", 0) for r in rows) / len(rows) if rows else 0,
        "mean_output_tokens": sum(r.get("output_tokens", 0) for r in rows) / len(rows) if rows else 0,
    }
    
    # Classification Accuracy
    cls_correct = [r["classification_correct"] for r in rows if "classification_correct" in r]
    if cls_correct:
        agg_stats["classification_accuracy"] = sum(cls_correct) / len(cls_correct)

    # Mean Latency per Stage
    stages = ["time_Ingestion", "time_Classification", "time_Extraction", "time_Enrichment", "time_Embedding"]
    for stage in stages:
        values = [r.get(stage, 0) for r in rows if stage in r]
        if values:
            agg_stats[f"mean_{stage}"] = sum(values) / len(values)
    
    # Mean CER/WER if available
    cers = [r["cer"] for r in rows if "cer" in r]
    if cers:
        agg_stats["mean_cer"] = sum(cers) / len(cers)
        
    wers = [r["wer"] for r in rows if "wer" in r]
    if wers:
        agg_stats["mean_wer"] = sum(wers) / len(wers)
        
    # Write Aggregated Stats
    agg_csv = output_csv.with_name(f"{output_csv.stem}_summary.csv")
    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        # Pre-sort keys for consistent output
        fieldnames = sorted(agg_stats.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(agg_stats)
    print(f"Wrote summary stats to {agg_csv}")
    
    # Retrieval evaluation (if you have a query file)
    # Check manifest first
    queries = []
    relevant_ids = []
    
    if 'queries' in manifest_data:
        for q in manifest_data['queries']:
            queries.append(q['query'])
            relevant_ids.append(q['relevant_ids'])
    else:
        queries_path = data_dir / "queries.json"
        if queries_path.exists():
             with queries_path.open("r", encoding="utf-8") as f:
                qdata = json.load(f)
                queries = [q["query"] for q in qdata]
                relevant_ids = [q["relevant_ids"] for q in qdata]

    if queries and args.api_key:
        print("\nRunning retrieval evaluation...")
        embedder = pipeline.embedding_provider
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
