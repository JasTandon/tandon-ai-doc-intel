import fitz
import tempfile
import os
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
                # We default to 'lattice' as it's more common in formal docs, or try both.
                # For this MVP, let's try 'lattice'.
                camelot_tables = camelot.read_pdf(temp_path, pages='all', flavor='lattice')
                
                for i, table in enumerate(camelot_tables):
                    tables.append({
                        "index": i,
                        "page": table.page,
                        "accuracy": table.accuracy,
                        "whitespace": table.whitespace,
                        "data": table.df.to_dict(orient="records"), # Convert DataFrame to list of dicts
                        "html": table.df.to_html()
                    })
                    
            except Exception as e:
                print(f"Camelot table extraction failed: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except ImportError:
            print("Camelot not installed. Skipping table extraction.")
            
        return "\n".join(full_text), tables
