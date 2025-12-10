from dataclasses import dataclass

@dataclass
class PipelineConfig:
    """
    Configuration for the DocumentPipeline.
    Allows enabling/disabling components for ablation studies.
    """
    use_analytics: bool = True
    use_validation: bool = True
    enable_embeddings: bool = True
    enable_enrichment: bool = True
    extractor_mode: str = "auto"  # "auto", "digital_only", "scanned_only"

