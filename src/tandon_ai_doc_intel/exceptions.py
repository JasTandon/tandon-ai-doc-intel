class DocIntelError(Exception):
    """Base exception for the library."""
    pass

class IngestionError(DocIntelError):
    """Failed to ingest document."""
    pass

class ExtractionError(DocIntelError):
    """Failed to extract text or tables."""
    pass

class EnrichmentError(DocIntelError):
    """Failed to generate LLM insights."""
    pass

class EmbeddingError(DocIntelError):
    """Failed to generate or store embeddings."""
    pass

class ConfigurationError(DocIntelError):
    """Invalid configuration."""
    pass

