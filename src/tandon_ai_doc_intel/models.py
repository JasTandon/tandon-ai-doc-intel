from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class DocumentResult:
    """
    Structured result returned by the document processing pipeline.
    """
    text: str = ""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: Optional[List[float]] = None
    risk_analysis: Optional[Dict[str, Any]] = None
    chunks: List[str] = field(default_factory=list)
    
    # Validation
    validation_score: float = 1.0
    validation_issues: List[str] = field(default_factory=list)

    # Advanced Analytics & ML
    readability_score: float = 0.0
    processing_time_seconds: Dict[str, float] = field(default_factory=dict)
    ml_keywords: List[str] = field(default_factory=list) # TF-IDF/RAKE
    topics: List[str] = field(default_factory=list) # NMF/LDA Topics
    
    # Semantic Metrics
    sentiment_polarity: float = 0.0 # -1.0 (Negative) to 1.0 (Positive)
    sentiment_subjectivity: float = 0.0 # 0.0 (Objective) to 1.0 (Subjective)
    lexical_diversity: float = 0.0 # Unique words / Total words
    
    # Advanced Readability
    gunning_fog: float = 0.0 
    automated_readability_index: float = 0.0

