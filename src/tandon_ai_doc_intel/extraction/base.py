from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

class BaseExtractor(ABC):
    """
    Abstract base class for document extractors.
    """
    
    @abstractmethod
    def extract(self, file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extracts text and tables from document bytes.
        
        Args:
            file_bytes: The document content.
            
        Returns:
            Tuple[str, List[Dict[str, Any]]]: A tuple containing the full text and a list of tables.
        """
        pass

