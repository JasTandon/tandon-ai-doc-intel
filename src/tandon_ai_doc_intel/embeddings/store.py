import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional

class VectorStore:
    """
    Manages storage and retrieval of embeddings using ChromaDB.
    """
    
    def __init__(self, collection_name: str = "documents", path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, 
                      ids: List[str], 
                      documents: List[str], 
                      embeddings: List[List[float]], 
                      metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Adds documents and their embeddings to the store.
        """
        if not ids or not documents or not embeddings:
            return

        try:
            # ChromaDB batch size limit is often 5461. 
            # Safe batch size: 100
            batch_size = 100
            total_docs = len(ids)
            
            for i in range(0, total_docs, batch_size):
                end = min(i + batch_size, total_docs)
                self.collection.add(
                    ids=ids[i:end],
                    documents=documents[i:end],
                    embeddings=embeddings[i:end],
                    metadatas=metadatas[i:end] if metadatas else None
                )
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            # If it fails, maybe IDs duplicate?
            # Re-try with upsert or just log.
            pass

    def query(self, query_embeddings: List[float], n_results: int = 5):
        """
        Queries the vector store for similar documents.
        """
        return self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=n_results
        )

