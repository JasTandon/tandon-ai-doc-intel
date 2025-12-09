import os
import io
from pathlib import Path
from typing import Union, BinaryIO

class DocumentIngestor:
    """
    Handles ingestion of documents from file paths, bytes, or file-like objects.
    """

    @staticmethod
    def ingest(source: Union[str, Path, bytes, BinaryIO]) -> bytes:
        """
        Ingests a document and returns its bytes content.
        
        Args:
            source: File path (str/Path), raw bytes, or file-like object.
            
        Returns:
            bytes: The raw content of the document.
            
        Raises:
            ValueError: If the source format is not supported or unreadable.
        """
        if isinstance(source, (str, Path)):
            file_path = Path(source)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            return file_path.read_bytes()
        
        elif isinstance(source, bytes):
            return source
        
        elif hasattr(source, 'read'):
            # File-like object
            content = source.read()
            if isinstance(content, str):
                return content.encode('utf-8') # normalize to bytes
            return content
            
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

