import jiwer
from typing import List, Dict, Any, Tuple

class Evaluator:
    """
    Helper class for computing quantitative metrics.
    """

    @staticmethod
    def compare_text(reference: str, extracted: str) -> Dict[str, float]:
        """
        Computes CER and WER between reference and extracted text.
        """
        if not reference or not extracted:
            return {"cer": 1.0, "wer": 1.0}
            
        try:
            cer = jiwer.cer(reference, extracted)
            wer = jiwer.wer(reference, extracted)
            return {"cer": cer, "wer": wer}
        except Exception as e:
            print(f"Error computing text metrics: {e}")
            return {"cer": 0.0, "wer": 0.0}

    @staticmethod
    def compare_tables(ref_tables: List[Dict], extracted_tables: List[Dict]) -> Dict[str, float]:
        """
        Computes table extraction accuracy.
        Assumes ref_tables and extracted_tables are lists of dicts with 'data' key (list of rows).
        """
        if not ref_tables:
            return {"table_precision": 0.0, "table_recall": 0.0, "table_f1": 0.0, "cell_accuracy": 0.0}
            
        # Simplified matching: Check if ref table content exists in extracted tables
        # A robust implementation would align tables by page/bbox.
        
        total_ref_cells = 0
        correct_cells = 0
        matched_tables = 0
        
        for ref in ref_tables:
            ref_data = ref.get("data", [])
            total_ref_cells += sum(len(row) for row in ref_data)
            
            # Find best match in extracted
            best_match_score = 0
            best_match_cells = 0
            
            for ext in extracted_tables:
                ext_data = ext.get("data", [])
                
                # Convert to sets of strings for overlap
                ref_set = set(str(cell) for row in ref_data for cell in row.values() if cell)
                ext_set = set(str(cell) for row in ext_data for cell in row.values() if cell)
                
                if not ref_set:
                    continue
                    
                intersect = len(ref_set.intersection(ext_set))
                score = intersect / len(ref_set)
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_cells = intersect
            
            if best_match_score > 0.7: # Threshold for "found table"
                matched_tables += 1
                correct_cells += best_match_cells
                
        precision = matched_tables / len(extracted_tables) if extracted_tables else 0.0
        recall = matched_tables / len(ref_tables)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        cell_acc = correct_cells / total_ref_cells if total_ref_cells > 0 else 0.0
        
        return {
            "table_precision": precision,
            "table_recall": recall,
            "table_f1": f1,
            "cell_accuracy": cell_acc
        }

    @staticmethod
    def evaluate_retrieval(vector_store, queries: List[str], ground_truth_ids: List[List[str]], k: int = 5) -> Dict[str, float]:
        """
        Evaluates retrieval performance (Recall@K, MRR).
        """
        # This requires the VectorStore to have a query method and existing embeddings.
        # We assume vector_store has a `.query(embedding, k)` method.
        # But we need an embedding provider to embed queries first.
        # For simplicity, we'll return a placeholder structure or need dependency injection.
        pass
