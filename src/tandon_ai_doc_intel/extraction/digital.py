import fitz
import tempfile
import os
import shutil
from typing import Tuple, List, Dict, Any
from .base import BaseExtractor

class DigitalPDFExtractor(BaseExtractor):
    """
    Extracts text from digital PDFs using PyMuPDF and tables using Camelot.
    """
    
    def extract(self, file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]]]:
        full_text = []
        tables = [] 
        
        # Text Extraction with PyMuPDF
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text = page.get_text()
                full_text.append(text)
        
        # Table Extraction with Camelot
        # Camelot requires a file path, so we write to a temp file
        try:
            import camelot
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf.write(file_bytes)
                temp_path = temp_pdf.name
            
            try:
                # 'stream' flavor is good for whitespace-separated tables
                # 'lattice' is good for tables with grid lines
                # We try 'lattice' first as it is more precise for forms.
                camelot_tables = camelot.read_pdf(temp_path, pages='all', flavor='lattice')
                
                raw_tables = []
                for table in camelot_tables:
                    # Capture bbox for overlap detection: (x1, y1, x2, y2)
                    # Camelot table._bbox is usually [x1, y1, x2, y2] in PDF coords (bottom-left origin)
                    # We store it for post-processing.
                    raw_tables.append({
                        "page": table.page,
                        "accuracy": table.accuracy,
                        "whitespace": table.whitespace,
                        "data": table.df.to_dict(orient="records"),
                        "df": table.df,
                        "_bbox": getattr(table, "_bbox", None) 
                    })
                
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
                    
            except Exception as e:
                print(f"Camelot table extraction failed: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except ImportError:
            print("Camelot not installed. Skipping table extraction.")
            
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
