from typing import List, Dict, Any
from .models import DocumentResult

class Validator:
    """
    Validates the quality of document extraction and enrichment.
    """

    def validate(self, result: DocumentResult) -> DocumentResult:
        """
        Analyzes the DocumentResult and assigns a validation score and issues list.
        """
        score = 1.0
        issues = []

        # 1. Text Density Check
        # If text is very short for a PDF, it might be a failed OCR or empty page.
        text_len = len(result.text.strip())
        if text_len < 100:
            score -= 0.3
            issues.append("Low text content detected (< 100 chars). Possible blank page or failed extraction.")
        
        # 2. OCR Garbage Detection (Simple Heuristic)
        # Check for high ratio of non-alphanumeric characters which often indicates bad OCR
        if text_len > 0:
            import re
            non_alnum = len(re.sub(r'[a-zA-Z0-9\s]', '', result.text))
            ratio = non_alnum / text_len
            if ratio > 0.3: # > 30% garbage characters
                score -= 0.2
                issues.append(f"High noise ratio detected ({ratio:.2f}). Possible poor OCR quality.")

        # 3. Validation of Structured Data
        # If entities were requested but none found, flag it (context dependent)
        if result.metadata.get("is_digital_pdf") is False:
             # Scanned PDFs often have lower confidence
             pass # Could lower base score slightly?

        # 4. Table Validation
        # If tables exist, check their accuracy score (from Camelot)
        if result.tables:
            low_acc_tables = [t for t in result.tables if t.get("accuracy", 100) < 80]
            if low_acc_tables:
                score -= 0.1 * len(low_acc_tables)
                issues.append(f"Detected {len(low_acc_tables)} tables with low accuracy (< 80%).")

        # Clamp score
        result.validation_score = max(0.0, min(1.0, score))
        result.validation_issues = issues
        
        return result

