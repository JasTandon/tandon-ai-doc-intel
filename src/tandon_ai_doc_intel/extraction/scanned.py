import fitz
import pytesseract
from PIL import Image
import io
import cv2
import numpy as np
import tempfile
import os
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Dict, Any
from .base import BaseExtractor

logger = logging.getLogger(__name__)

class ScannedPDFExtractor(BaseExtractor):
    """
    Extracts text from scanned PDFs using OCR (Tesseract).
    Includes optional preprocessing (deskewing, denoising).
    """
    
    def __init__(self, enable_preprocessing: bool = True):
        self.enable_preprocessing = enable_preprocessing

    def _preprocess_image(self, image_path: str) -> str:
        """
        Applies deskewing and denoising to the image using OpenCV.
        Returns path to processed image.
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return image_path
            
            # 1. Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. Denoise
            denoised = cv2.fastNlMeansDenoising(gray, h=10, searchWindowSize=21, templateWindowSize=7)
            
            # 3. Binarization (Otsu's thresholding)
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 4. Deskewing
            coords = np.column_stack(np.where(binary > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            (h, w) = binary.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            # Write processed image
            processed_path = image_path + "_processed.png"
            cv2.imwrite(processed_path, rotated)
            
            return processed_path
            
        except Exception as e:
            logger.warning(f"Preprocessing failed for {image_path}: {e}")
            return image_path

    def _ocr_page(self, page_num: int, pix: fitz.Pixmap) -> str:
        """
        Helper to process a single page safely.
        """
        logger.info(f"[ScannedExtractor] Processing page {page_num}...")
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            tmp_img.write(pix.tobytes("png"))
            tmp_img_path = tmp_img.name
            
        try:
            # Preprocess if enabled
            if self.enable_preprocessing:
                final_img_path = self._preprocess_image(tmp_img_path)
            else:
                final_img_path = tmp_img_path
                
            output_base = tmp_img_path + "_out"
            
            # Run Tesseract via CLI for stability
            # Added basic timeout and retry logic
            cmd = ["tesseract", final_img_path, output_base]
            
            # Retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"[ScannedExtractor] Running Tesseract CLI: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                    break
                except subprocess.TimeoutExpired:
                     logger.warning(f"[ScannedExtractor] Tesseract timeout on page {page_num} (attempt {attempt+1})")
                     if attempt == max_retries - 1:
                         return ""
                except subprocess.CalledProcessError as e:
                     logger.warning(f"[ScannedExtractor] Tesseract error on page {page_num}: {e}")
                     if attempt == max_retries - 1:
                         return ""

            # Read result
            output_file = output_base + ".txt"
            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    text = f.read()
                os.remove(output_file)
                return text
            return ""
            
        finally:
            if os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)
            if self.enable_preprocessing and os.path.exists(tmp_img_path + "_processed.png"):
                os.remove(tmp_img_path + "_processed.png")

    def extract(self, file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]]]:
        logger.info("[ScannedExtractor] Starting OCR extraction...")
        full_text = []
        tables = []
        
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            num_pages = len(doc)
            logger.info(f"[ScannedExtractor] Document has {num_pages} pages.")
            
            # Use ThreadPool for parallel page processing
            # Limit workers to prevent CPU saturation
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i, page in enumerate(doc):
                    # Lower DPI for speed, 150 is usually enough for readable text
                    pix = page.get_pixmap(dpi=150)
                    futures.append(executor.submit(self._ocr_page, i+1, pix))
                
                # Collect results in order
                for future in futures:
                    full_text.append(future.result())
                
        return "\n".join(full_text), tables
