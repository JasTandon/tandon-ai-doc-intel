import fitz
from typing import Tuple, List, Dict, Any
from .base import BaseExtractor

class DigitalPDFExtractor(BaseExtractor):
    """
    Extracts text from digital PDFs using PyMuPDF.
    """
    
    def extract(self, file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]]]:
        full_text = []
        tables = [] # Placeholder for future table extraction logic
        
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text = page.get_text()
                full_text.append(text)
                
        return "\n".join(full_text), tables

