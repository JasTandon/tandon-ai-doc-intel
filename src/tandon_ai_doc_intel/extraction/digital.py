import fitz
import tempfile
import os
import shutil
import logging
import multiprocessing
import queue
from typing import Tuple, List, Dict, Any
from .base import BaseExtractor

logger = logging.getLogger(__name__)

def _run_camelot_process(file_path: str, result_queue: multiprocessing.Queue):
    """
    Worker function to run Camelot in a separate process.
    """
    import camelot
    try:
        # Run Camelot
        tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
        
        # Serialize data
        data = []
        for t in tables:
            data.append({
                "page": t.page,
                "accuracy": t.accuracy,
                "whitespace": t.whitespace,
                "data": t.df.to_dict(orient="records"),
                "_bbox": getattr(t, "_bbox", None)
            })
        result_queue.put({"success": True, "data": data})
        
    except Exception as e:
        result_queue.put({"success": False, "error": str(e)})

class DigitalPDFExtractor(BaseExtractor):
    """
    Extracts text from digital PDFs using PyMuPDF and tables using Camelot.
    """
    
    def extract(self, file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]]]:
        logger.info("    [DigitalExtractor] Starting text extraction...")
        full_text = []
        tables = [] 
        
        # Text Extraction with PyMuPDF
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text = page.get_text()
                full_text.append(text)
        logger.info(f"    [DigitalExtractor] Text extraction done. Found {len(full_text)} pages.")
        
        # Table Extraction with Camelot (in separate process with timeout)
        logger.info("    [DigitalExtractor] Starting table extraction (Camelot)...")
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(file_bytes)
            temp_path = temp_pdf.name
        
        try:
            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=_run_camelot_process, args=(temp_path, q))
            p.start()
            
            # Wait for 30 seconds max
            p.join(timeout=30)
            
            if p.is_alive():
                logger.warning("    [DigitalExtractor] Camelot timed out! Terminating process.")
                p.terminate()
                p.join()
                # Proceed without tables
            else:
                if not q.empty():
                    res = q.get()
                    if res["success"]:
                        raw_tables = res["data"]
                        logger.info(f"    [DigitalExtractor] Camelot found {len(raw_tables)} tables.")
                        
                        # Deduplicate / Clean Tables
                        cleaned_tables = self._post_process_tables(raw_tables)
                        
                        for i, table in enumerate(cleaned_tables):
                            tables.append({
                                "index": i,
                                "page": table["page"],
                                "accuracy": table["accuracy"],
                                "whitespace": table["whitespace"],
                                "data": table["data"]
                            })
                    else:
                         logger.error(f"    [DigitalExtractor] Camelot worker error: {res['error']}")
                else:
                    logger.warning("    [DigitalExtractor] Camelot finished but returned no data.")

        except Exception as e:
            logger.error(f"    [DigitalExtractor] Table extraction wrapper failed: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
        return "\n".join(full_text), tables

    def _post_process_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters out subset tables or merges overlapping ones.
        Simple heuristic: If Table A is a subset of rows of Table B, discard A.
        """
        if not tables:
            return []
            
        # Group by page
        pages = {}
        for t in tables:
            p = t["page"]
            if p not in pages:
                pages[p] = []
            pages[p].append(t)
            
        final_tables = []
        
        for p, page_tables in pages.items():
            if len(page_tables) == 1:
                final_tables.extend(page_tables)
                continue
                
            # Check for overlaps/subsets
            # We convert each table's content to a set of stringified rows to check subset
            # This is robust against minor coordinate shifts
            
            # Sort by number of rows (descending), so we keep the largest likely superset
            page_tables.sort(key=lambda x: len(x["data"]), reverse=True)
            
            kept = []
            for i, current in enumerate(page_tables):
                is_subset = False
                curr_rows = set(str(row) for row in current["data"])
                
                for other in kept:
                    other_rows = set(str(row) for row in other["data"])
                    
                    # If current rows are mostly contained in other rows
                    intersection = curr_rows.intersection(other_rows)
                    if len(intersection) / len(curr_rows) > 0.8: # 80% overlap
                        is_subset = True
                        break
                
                if not is_subset:
                    kept.append(current)
            
            final_tables.extend(kept)
            
        return final_tables
