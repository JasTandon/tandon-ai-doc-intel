import uuid
from typing import Optional, Union, BinaryIO, List
from pathlib import Path

from .models import DocumentResult
from .ingestion import DocumentIngestor
from .classification import DocumentClassifier
from .extraction import DigitalPDFExtractor, ScannedPDFExtractor, BaseExtractor
from .enrichment.llm import LLMEnricher
from .embeddings import OpenAIEmbeddings, VectorStore, EmbeddingsProvider
from .validation import Validator
from .analytics import AdvancedAnalytics
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
                 vector_store: Optional[VectorStore] = None):
        
        self.enricher = LLMEnricher(api_key=openai_api_key)
        self.embedding_provider = embedding_provider or OpenAIEmbeddings(api_key=openai_api_key)
        self.vector_store = vector_store or VectorStore()
        self.validator = Validator()
        self.analytics = AdvancedAnalytics()
        
        # Extractors
        self.digital_extractor = DigitalPDFExtractor()
        self.scanned_extractor = ScannedPDFExtractor()

    def process(self, source: Union[str, Path, bytes, BinaryIO]) -> DocumentResult:
        """
        Runs the full pipeline on a document source.
        """
        timings = {}
        t_start = time.time()
        
        logger.info(f"  [Pipeline] Starting processing...")

        # 1. Ingestion
        t0 = time.time()
        file_bytes = DocumentIngestor.ingest(source)
        timings["Ingestion"] = time.time() - t0
        logger.info(f"  [Pipeline] Ingestion done ({timings['Ingestion']:.2f}s)")
        
        # 2. Classification
        t0 = time.time()
        is_digital = DocumentClassifier.is_digital_pdf(file_bytes)
        extractor = self.digital_extractor if is_digital else self.scanned_extractor
        timings["Classification"] = time.time() - t0
        logger.info(f"  [Pipeline] Classification done ({timings['Classification']:.2f}s) - {'Digital' if is_digital else 'Scanned'}")
        
        # 3. Extraction
        t0 = time.time()
        text, tables = extractor.extract(file_bytes)
        timings["Extraction"] = time.time() - t0
        logger.info(f"  [Pipeline] Extraction done ({timings['Extraction']:.2f}s) - {len(text)} chars, {len(tables)} tables")
        
        result = DocumentResult(
            text=text,
            tables=tables,
            metadata={
                "is_digital_pdf": is_digital,
                "source_size": len(file_bytes)
            }
        )
        
        # 4. Enrichment (includes Chunking)
        t0 = time.time()
        self.enrich(result)
        timings["Enrichment"] = time.time() - t0
        logger.info(f"  [Pipeline] Enrichment done ({timings['Enrichment']:.2f}s)")
        
        # 5. Validation
        t0 = time.time()
        self.validator.validate(result)
        timings["Validation"] = time.time() - t0
        logger.info(f"  [Pipeline] Validation done ({timings['Validation']:.2f}s)")

        # 6. Advanced Analytics (ML & Metrics)
        t0 = time.time()
        try:
            # Run in a separate thread/process with timeout if needed, but for now just try-except
            # If it hangs, it might be NMF on sparse data.
            # Let's add a timeout wrapper around analytics if it's the culprit.
            self.analytics.analyze(result)
        except Exception as e:
            logger.error(f"  [Pipeline] Analytics module failed completely: {e}")
        timings["Analytics"] = time.time() - t0
        logger.info(f"  [Pipeline] Analytics done ({timings['Analytics']:.2f}s)")
        
        # 7. Embedding & Storage
        t0 = time.time()
        try:
            self.embed_and_store(result)
        except Exception as e:
            logger.error(f"  [Pipeline] Embedding failed: {e}")
            # Continue even if embedding fails
        timings["Embedding"] = time.time() - t0
        logger.info(f"  [Pipeline] Embedding done ({timings['Embedding']:.2f}s)")
        
        timings["Total"] = time.time() - t_start
        result.processing_time_seconds = timings
        logger.info(f"  [Pipeline] Finished. Total time: {timings['Total']:.2f}s")

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

        # Store flat list of embeddings? 
        # The result object has `embeddings` field which implies maybe one per document or list of them.
        # Let's store the MEAN embedding or just the list. The Type is List[float] (single vector) or List[List[float]]?
        # In models.py: embeddings: Optional[List[float]] = None. This usually implies a single vector for the doc.
        # But for search we often want chunk embeddings.
        # I'll average them for the document-level embedding field, but store individual chunks in Vector DB.
        
        if embeddings_list:
            # Simple average for doc level
            doc_embedding = [sum(x) / len(embeddings_list) for x in zip(*embeddings_list)]
            result.embeddings = doc_embedding
            
            # Store in Vector DB
            # We need IDs for chunks
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
        # Already handled in embed_and_store for this flow, but exposed for API.
        pass

