import fitz
import pytesseract
from PIL import Image
import io
from typing import Tuple, List, Dict, Any
from .base import BaseExtractor

class ScannedPDFExtractor(BaseExtractor):
    """
    Extracts text from scanned PDFs using OCR (Tesseract).
    """
    
    def extract(self, file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]]]:
        full_text = []
        tables = []
        
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                # Convert page to image (pixmap)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                
                # Use PIL to load image for pytesseract
                image = Image.open(io.BytesIO(img_bytes))
                
                # Perform OCR
                text = pytesseract.image_to_string(image)
                full_text.append(text)
                
        return "\n".join(full_text), tables

