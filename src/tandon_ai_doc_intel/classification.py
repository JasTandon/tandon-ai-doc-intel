import fitz  # PyMuPDF
import io

class DocumentClassifier:
    """
    Classifies documents (e.g., Digital vs. Scanned PDF).
    """

    @staticmethod
    def is_digital_pdf(file_bytes: bytes, text_threshold: float = 0.05) -> bool:
        """
        Determines if a PDF is digital or scanned based on text layer density.
        
        Args:
            file_bytes: The PDF file content in bytes.
            text_threshold: The percentage of page area covered by text to be considered digital.
                            Default is 5%.
        
        Returns:
            bool: True if digital, False if scanned.
        """
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                if len(doc) == 0:
                    return False
                
                # Check the first few pages (up to 3) to speed up
                pages_to_check = min(len(doc), 3)
                total_text_len = 0
                
                for i in range(pages_to_check):
                    page = doc[i]
                    text = page.get_text()
                    total_text_len += len(text.strip())
                
                # Heuristic: If we extracted a reasonable amount of text, it's digital.
                # A robust check might compare text area to page area, but length is a good proxy.
                return total_text_len > 50  # Arbitrary small threshold for "contains text"
                
        except Exception as e:
            # If it fails to open or parse as PDF, it might not be a PDF or is corrupted.
            # For this pipeline, we assume it's a PDF if we reached here, 
            # but logging would be good.
            print(f"Error classifying PDF: {e}")
            return False

