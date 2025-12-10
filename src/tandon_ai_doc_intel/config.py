from dataclasses import dataclass

@dataclass
class PipelineConfig:
    """
    Configuration for the DocumentPipeline.
    Allows enabling/disabling components for ablation studies.
    """
    enable_ocr: bool = True
    enable_llm_enrichment: bool = True
    enable_embeddings: bool = True
    enable_validation: bool = True
    enable_analytics: bool = True
    extractor_mode: str = "auto"  # "auto", "digital_only", "scanned_only"
    
    # Advanced
    use_caching: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300
