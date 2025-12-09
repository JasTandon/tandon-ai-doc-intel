import uuid
from typing import Optional, Union, BinaryIO, List
from pathlib import Path

from .models import DocumentResult
from .ingestion import DocumentIngestor
from .classification import DocumentClassifier
from .extraction import DigitalPDFExtractor, ScannedPDFExtractor, BaseExtractor
from .enrichment.llm import LLMEnricher
from .embeddings import OpenAIEmbeddings, VectorStore, EmbeddingsProvider

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
        
        # Extractors
        self.digital_extractor = DigitalPDFExtractor()
        self.scanned_extractor = ScannedPDFExtractor()

    def process(self, source: Union[str, Path, bytes, BinaryIO]) -> DocumentResult:
        """
        Runs the full pipeline on a document source.
        """
        # 1. Ingestion
        file_bytes = DocumentIngestor.ingest(source)
        
        # 2. Classification
        is_digital = DocumentClassifier.is_digital_pdf(file_bytes)
        extractor = self.digital_extractor if is_digital else self.scanned_extractor
        
        # 3. Extraction
        text, tables = extractor.extract(file_bytes)
        
        result = DocumentResult(
            text=text,
            tables=tables,
            metadata={
                "is_digital_pdf": is_digital,
                "source_size": len(file_bytes)
            }
        )
        
        # 4. Enrichment
        self.enrich(result)
        
        # 5. Embedding & Storage
        self.embed_and_store(result)
        
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

