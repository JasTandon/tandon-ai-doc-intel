import fitz
import pytesseract
from PIL import Image, ImageOps, ImageFilter
import io
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
    Includes optional preprocessing using Pillow (safer than OpenCV for multiprocessing).
    """
    
    def __init__(self, enable_preprocessing: bool = True):
        self.enable_preprocessing = enable_preprocessing

    def _preprocess_image(self, image_path: str) -> str:
        """
        Applies basic denoising/grayscale using Pillow.
        Avoids OpenCV to prevent multiprocessing crashes on macOS.
        """
        try:
            with Image.open(image_path) as img:
                # Convert to grayscale
                gray = ImageOps.grayscale(img)
                
                # Apply slight median filter for denoising
                denoised = gray.filter(ImageFilter.MedianFilter(size=3))
                
                # Binarize (simple threshold)
                fn = lambda x : 255 if x > 200 else 0
                binary = denoised.point(fn, mode='1')
                
                # Save processed image
                processed_path = image_path + "_processed.png"
                binary.save(processed_path)
                return processed_path
                
        except Exception as e:
            logger.warning(f"Preprocessing failed for {image_path}: {e}")
            return image_path

    def _ocr_page(self, page_num: int, pix: fitz.Pixmap) -> str:
        """
        Helper to process a single page safely.
        """
        logger.info(f"[ScannedExtractor] Processing page {page_num}...")
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                tmp_img.write(pix.tobytes("png"))
                tmp_img_path = tmp_img.name
            logger.debug(f"[ScannedExtractor] Wrote temp image for page {page_num} to {tmp_img_path}")
        except Exception as e:
            logger.error(f"[ScannedExtractor] Failed to write temp image for page {page_num}: {e}")
            return ""
            
        try:
            # Preprocess if enabled
            if self.enable_preprocessing:
                logger.debug(f"[ScannedExtractor] Preprocessing page {page_num}...")
                final_img_path = self._preprocess_image(tmp_img_path)
            else:
                final_img_path = tmp_img_path
                
            output_base = tmp_img_path + "_out"
            
            # Run Tesseract via CLI for stability
            cmd = ["tesseract", final_img_path, output_base]
            
            # Retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"[ScannedExtractor] Running Tesseract CLI (Attempt {attempt+1}): {' '.join(cmd)}")
                    # Increased timeout to 60s
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                    break
                except subprocess.TimeoutExpired:
                     logger.warning(f"[ScannedExtractor] Tesseract timeout on page {page_num} (attempt {attempt+1})")
                     if attempt == max_retries - 1:
                         return ""
                except subprocess.CalledProcessError as e:
                     logger.warning(f"[ScannedExtractor] Tesseract error on page {page_num}: {e}")
                     if attempt == max_retries - 1:
                         return ""
                except Exception as e:
                     logger.error(f"[ScannedExtractor] Unexpected error running Tesseract on page {page_num}: {e}")
                     return ""

            # Read result
            output_file = output_base + ".txt"
            if os.path.exists(output_file):
                logger.debug(f"[ScannedExtractor] Reading Tesseract output for page {page_num} from {output_file}")
                with open(output_file, "r", encoding="utf-8") as f:
                    text = f.read()
                os.remove(output_file)
                return text
            else:
                logger.warning(f"[ScannedExtractor] No output file found for page {page_num}: {output_file}")
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
            
            if num_pages == 1:
                logger.info("[ScannedExtractor] Processing single page sequentially.")
                pix = doc[0].get_pixmap(dpi=150)
                full_text.append(self._ocr_page(1, pix))
            else:
                # Use ThreadPool for parallel page processing
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = []
                    for i, page in enumerate(doc):
                        pix = page.get_pixmap(dpi=150)
                        futures.append(executor.submit(self._ocr_page, i+1, pix))
                    
                    for future in futures:
                        full_text.append(future.result())
                
        return "\n".join(full_text), tables
