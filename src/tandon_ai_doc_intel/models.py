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

