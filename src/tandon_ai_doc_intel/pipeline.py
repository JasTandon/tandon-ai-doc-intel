import uuid
from typing import Optional, Union, BinaryIO, List, Dict, Any
from pathlib import Path

from .models import DocumentResult
from .ingestion import DocumentIngestor
from .classification import DocumentClassifier
from .extraction import DigitalPDFExtractor, ScannedPDFExtractor, BaseExtractor
from .enrichment.llm import LLMEnricher
from .embeddings import OpenAIEmbeddings, VectorStore, EmbeddingsProvider
from .validation import Validator
from .analytics import AdvancedAnalytics
from .config import PipelineConfig
from .evaluation import Evaluator
import time
import logging

logger = logging.getLogger(__name__)

class DocumentPipeline:
    """
    Core pipeline for processing documents.
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 embedding_provider: Optional[EmbeddingsProvider] = None,
                 vector_store: Optional[VectorStore] = None,
                 config: Optional[PipelineConfig] = None):
        
        self.config = config or PipelineConfig()
        
        # Pass caching preference to Enricher
        self.enricher = LLMEnricher(api_key=openai_api_key, use_caching=self.config.use_caching)
        
        self.embedding_provider = embedding_provider or OpenAIEmbeddings(api_key=openai_api_key)
        self.vector_store = vector_store or VectorStore()
        self.validator = Validator()
        self.analytics = AdvancedAnalytics()
        self.evaluator = Evaluator()
        
        # Extractors
        self.digital_extractor = DigitalPDFExtractor()
        # Enable preprocessing for scanned PDF extractor if needed (can be added to config)
        self.scanned_extractor = ScannedPDFExtractor(enable_preprocessing=True)

    def process(self, source: Union[str, Path, bytes, BinaryIO]) -> DocumentResult:
        """
        Runs the full pipeline on a document source.
        """
        timings = {}
        t_start = time.time()
        
        logger.info(f"  [Pipeline] Starting processing...")

        # Initialize result early for soft-fails
        result = DocumentResult()

        try:
            # 1. Ingestion
            t0 = time.time()
            file_bytes = DocumentIngestor.ingest(source)
            result.metadata["source_size"] = len(file_bytes)
            timings["Ingestion"] = time.time() - t0
            logger.info(f"  [Pipeline] Ingestion done ({timings['Ingestion']:.2f}s)")
            
            # 2. Classification
            t0 = time.time()
            if self.config.extractor_mode == "digital_only":
                is_digital = True
            elif self.config.extractor_mode == "scanned_only":
                is_digital = False
            else:
                is_digital = DocumentClassifier.is_digital_pdf(file_bytes)
            
            result.metadata["is_digital_pdf"] = is_digital
            
            extractor = self.digital_extractor if is_digital else self.scanned_extractor
            timings["Classification"] = time.time() - t0
            logger.info(f"  [Pipeline] Classification done ({timings['Classification']:.2f}s) - {'Digital' if is_digital else 'Scanned'}")
            
            # 3. Extraction
            if self.config.enable_ocr:
                t0 = time.time()
                try:
                    text, tables = extractor.extract(file_bytes)
                    result.text = text
                    result.tables = tables
                    timings["Extraction"] = time.time() - t0
                    logger.info(f"  [Pipeline] Extraction done ({timings['Extraction']:.2f}s) - {len(text)} chars, {len(tables)} tables")
                except Exception as e:
                    logger.error(f"  [Pipeline] Extraction failed: {e}")
                    # Soft fail: continue with empty text if allowed, else raise?
                    # For pipeline continuity, we continue but mark error
                    result.metadata["extraction_error"] = str(e)
            else:
                timings["Extraction"] = 0.0

            # 4. Enrichment (includes Chunking)
            if self.config.enable_llm_enrichment:
                t0 = time.time()
                try:
                    self.enrich(result)
                    # Pull token usage from enricher
                    if hasattr(self.enricher, 'token_usage'):
                        result.token_usage.update(self.enricher.token_usage)
                except Exception as e:
                    logger.error(f"  [Pipeline] Enrichment failed: {e}")
                    result.metadata["enrichment_error"] = str(e)

                timings["Enrichment"] = time.time() - t0
                logger.info(f"  [Pipeline] Enrichment done ({timings['Enrichment']:.2f}s)")
            else:
                timings["Enrichment"] = 0.0
                # Ensure chunks are generated even if enrichment is disabled, for embeddings
                result.chunks = self.enricher.chunk_text(result.text)
            
            # 5. Validation
            if self.config.use_validation:
                t0 = time.time()
                self.validator.validate(result)
                timings["Validation"] = time.time() - t0
                logger.info(f"  [Pipeline] Validation done ({timings['Validation']:.2f}s)")
            else:
                timings["Validation"] = 0.0

            # 6. Advanced Analytics (ML & Metrics)
            if self.config.use_analytics:
                t0 = time.time()
                try:
                    self.analytics.analyze(result)
                    
                    # Calculate Factuality Score (Proxy)
                    if result.summary and result.chunks:
                        overlap_score = 0.0
                        summary_words = set(result.summary.lower().split())
                        chunk_words = set(" ".join(result.chunks).lower().split())
                        if chunk_words:
                            overlap_score = len(summary_words.intersection(chunk_words)) / len(summary_words) if len(summary_words) > 0 else 0.0
                        result.factuality_score = overlap_score

                except Exception as e:
                    logger.error(f"  [Pipeline] Analytics module failed completely: {e}")
                timings["Analytics"] = time.time() - t0
                logger.info(f"  [Pipeline] Analytics done ({timings['Analytics']:.2f}s)")
            else:
                timings["Analytics"] = 0.0
            
            # 7. Embedding & Storage
            if self.config.enable_embeddings:
                t0 = time.time()
                try:
                    self.embed_and_store(result)
                except Exception as e:
                    logger.error(f"  [Pipeline] Embedding failed: {e}")
                timings["Embedding"] = time.time() - t0
                logger.info(f"  [Pipeline] Embedding done ({timings['Embedding']:.2f}s)")
            else:
                timings["Embedding"] = 0.0
        
        except Exception as e:
            logger.critical(f"  [Pipeline] Critical failure: {e}", exc_info=True)
            result.metadata["critical_error"] = str(e)
            
        finally:
            timings["Total"] = time.time() - t_start
            result.processing_time_seconds = timings
            result.runtime_metrics = timings
            
            # Cost Estimation
            # Re-calculate based on actual tracked tokens + estimates
            
            # 1. LLM tokens from Enricher
            input_tokens = result.token_usage.get("llm_input", 0)
            output_tokens = result.token_usage.get("llm_output", 0)
            
            # 2. Embedding tokens (estimate if not tracked by provider, but OpenAI provider usually doesn't return usage easily in this simple impl)
            # Estimate: 1 token ~= 4 chars
            embedding_tokens = len(result.text) / 4 if result.text else 0
            result.token_usage["embedding"] = int(embedding_tokens)
            
            # Pricing (gpt-3.5-turbo & ada-002)
            cost = (input_tokens / 1_000_000 * 0.50) + (output_tokens / 1_000_000 * 1.50)
            if self.config.enable_embeddings:
                cost += (embedding_tokens / 1_000_000 * 0.10)
                
            result.cost_estimate_usd = round(cost, 6)
            
            logger.info(f"  [Pipeline] Finished. Total time: {timings.get('Total', 0):.2f}s. Est. Cost: ${cost:.6f}")

        return result

    def extract(self, source: Union[str, Path, bytes, BinaryIO]) -> DocumentResult:
        """
        Only runs ingestion and extraction.
        """
        file_bytes = DocumentIngestor.ingest(source)
        is_digital = DocumentClassifier.is_digital_pdf(file_bytes)
        extractor = self.digital_extractor if is_digital else self.scanned_extractor
        text, tables = extractor.extract(file_bytes)
        
        return DocumentResult(
            text=text,
            tables=tables,
            metadata={"is_digital_pdf": is_digital}
        )

    def enrich(self, result: DocumentResult):
        """
        Enriches an existing DocumentResult with summaries, entities, and risk analysis.
        """
        if not result.text:
            return

        # Generate chunks
        result.chunks = self.enricher.chunk_text(result.text)
        
        # Summarize
        result.summary = self.enricher.summarize(result.text)
        
        # Entities
        result.entities = self.enricher.extract_entities(result.text)
        
        # Risk Analysis
        result.risk_analysis = self.enricher.analyze_risk(result.text)

    def embed_and_store(self, result: DocumentResult):
        """
        Generates embeddings for chunks and stores them in the vector store.
        """
        if not result.chunks:
            return

        # Generate embeddings
        embeddings_list = self.embedding_provider.embed(result.chunks)
        
        if not embeddings_list:
            # Fallback if no embeddings generated (e.g. no API key)
            return

        if embeddings_list:
            # Simple average for doc level
            doc_embedding = [sum(x) / len(embeddings_list) for x in zip(*embeddings_list)]
            result.embeddings = doc_embedding
            
            # Store in Vector DB
            base_id = str(uuid.uuid4())
            ids = [f"{base_id}_{i}" for i in range(len(result.chunks))]
            metadatas = [{"source_id": base_id, "chunk_index": i} for i in range(len(result.chunks))]
            
            self.vector_store.add_documents(
                ids=ids,
                documents=result.chunks,
                embeddings=embeddings_list,
                metadatas=metadatas
            )

    def store(self, result: DocumentResult):
        """
        Explicitly call store if needed separate from processing.
        """
        pass

    def evaluate(self, result: DocumentResult, reference_text: Optional[str] = None, reference_tables: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Evaluates the processed document against ground truth.
        Updates the result object with evaluation metrics.
        """
        metrics = {}
        
        if reference_text:
            text_metrics = self.evaluator.compare_text(reference_text, result.text)
            result.ocr_cer = text_metrics["cer"]
            result.ocr_wer = text_metrics["wer"]
            metrics.update(text_metrics)
            
        if reference_tables:
            table_metrics = self.evaluator.compare_tables(reference_tables, result.tables)
            result.table_accuracy = table_metrics["cell_accuracy"]
            metrics.update(table_metrics)
            
        return metrics
