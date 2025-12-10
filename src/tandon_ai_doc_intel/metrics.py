from __future__ import annotations

from typing import List, Dict, Iterable, Any
import math
import numpy as np

# --- Basic Levenshtein distance (no extra dependency) ---
def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur_row = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = cur_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (ca != cb)
            cur_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = cur_row
    return prev_row[-1]

# --- Text extraction metrics ---
def compute_cer(reference: str, extracted: str) -> float:
    """Character Error Rate (0–1)."""
    if not reference:
        return 0.0
    distance = _levenshtein(reference, extracted)
    return distance / len(reference)

def compute_wer(reference: str, extracted: str) -> float:
    """Word Error Rate (0–1)."""
    if not reference or not extracted:
        return 0.0
        
    ref_words = reference.split()
    ext_words = extracted.split()
    
    if not ref_words:
        return 0.0
        
    distance = _levenshtein(" ".join(ref_words), " ".join(ext_words))
    return distance / len(ref_words)

# --- Table accuracy metrics ---
def aggregate_table_accuracy(tables: List[Dict]) -> float:
    """Returns mean of 'accuracy' field across tables, or 0.0 if none."""
    accuracies = [
        float(t.get("accuracy", 0.0))
        for t in tables
        if "accuracy" in t
    ]
    if not accuracies:
        return 0.0
    return float(sum(accuracies) / len(accuracies))

# --- Retrieval metrics ---
def recall_at_k(results: List[List[str]], ground_truth: List[List[str]], k: int) -> float:
    """
    results[i] = list of doc_ids returned for query i (in rank order)
    ground_truth[i] = list of relevant doc_ids for query i
    """
    assert len(results) == len(ground_truth)
    recalls = []
    for res, rel in zip(results, ground_truth):
        if not rel:
            continue
        retrieved_k = set(res[:k])
        rel_set = set(rel)
        hit = len(retrieved_k & rel_set) / len(rel_set)
        recalls.append(hit)
    return sum(recalls) / len(recalls) if recalls else 0.0

def precision_at_k(results: List[List[str]], ground_truth: List[List[str]], k: int) -> float:
    assert len(results) == len(ground_truth)
    precisions = []
    for res, rel in zip(results, ground_truth):
        retrieved_k = res[:k]
        if not retrieved_k:
            continue
        rel_set = set(rel)
        hit = len([d for d in retrieved_k if d in rel_set]) / len(retrieved_k)
        precisions.append(hit)
    return sum(precisions) / len(precisions) if precisions else 0.0

def mrr(results: List[List[str]], ground_truth: List[List[str]]) -> float:
    """Mean Reciprocal Rank."""
    assert len(results) == len(ground_truth)
    rr = []
    for res, rel in zip(results, ground_truth):
        rel_set = set(rel)
        rank = None
        for idx, doc_id in enumerate(res, start=1):
            if doc_id in rel_set:
                rank = idx
                break
        if rank is not None:
            rr.append(1.0 / rank)
    return sum(rr) / len(rr) if rr else 0.0

def ndcg_at_k(results: List[List[str]], ground_truth: List[List[str]], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    Assumes binary relevance (1 if in ground_truth, 0 otherwise).
    """
    assert len(results) == len(ground_truth)
    scores = []
    
    for res, rel in zip(results, ground_truth):
        if not rel:
            continue
            
        rel_set = set(rel)
        dcg = 0.0
        idcg = 0.0
        
        # Calculate DCG
        for i, doc_id in enumerate(res[:k]):
            if doc_id in rel_set:
                # Binary relevance = 1
                dcg += 1.0 / math.log2(i + 2)
                
        # Calculate IDCG (Ideal DCG)
        # In ideal case, all relevant docs are at the top
        num_relevant = len(rel_set)
        for i in range(min(num_relevant, k)):
            idcg += 1.0 / math.log2(i + 2)
            
        if idcg > 0:
            scores.append(dcg / idcg)
        else:
            scores.append(0.0)
            
    return sum(scores) / len(scores) if scores else 0.0

def evaluate_retrieval(
    queries: List[str],
    relevant_ids: List[List[str]],
    vector_store,
    embedder,
    k: int = 5,
) -> Dict[str, float]:
    """
    Simple wrapper that:
      - embeds each query,
      - runs vector_store.query(),
      - computes Recall@k, Precision@k, MRR, and nDCG@k.
    """
    all_results: List[List[str]] = []
    for q in queries:
        q_vec = embedder.embed_query(q)
        hits = vector_store.query(q_vec, k=k) 
        
        if isinstance(hits, dict) and 'ids' in hits:
             ids = hits['ids'][0] if hits['ids'] else []
             all_results.append(ids)
        elif isinstance(hits, list):
             all_results.append([h.get("id") for h in hits])
        else:
             all_results.append([])

    return {
        "recall_at_k": recall_at_k(all_results, relevant_ids, k),
        "precision_at_k": precision_at_k(all_results, relevant_ids, k),
        "mrr": mrr(all_results, relevant_ids),
        "ndcg_at_k": ndcg_at_k(all_results, relevant_ids, k)
    }

# --- Cost & throughput (optional) ---
def estimate_cost(token_counts: Dict[str, int], pricing_per_1k: Dict[str, float]) -> float:
    """
    token_counts: {"llm_input": 1234, "llm_output": 567, "embeddings": 9999}
    pricing_per_1k: {"llm_input": 0.0005, "llm_output": 0.0015, "embeddings": 0.0001}
    """
    cost = 0.0
    for key, count in token_counts.items():
        price = pricing_per_1k.get(key, 0.0)
        cost += (count / 1000.0) * price
    return cost

def compute_throughput(num_docs: int, total_seconds: float) -> float:
    """Documents per second."""
    if total_seconds <= 0:
        return 0.0
    return num_docs / total_seconds
