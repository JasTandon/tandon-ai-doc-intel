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
            pass

    def query(self, query_embeddings: List[float], n_results: int = 5):
        """
        Queries the vector store for similar documents.
        """
        return self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=n_results
        )

    def hybrid_search(self, query: str, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search combining vector similarity and basic keyword filtering.
        Note: Since ChromaDB doesn't support full-text search native hybrid ranking easily, 
        this implementation fetches a larger candidate set via vector search and re-ranks 
        based on exact keyword presence in the chunk.
        """
        # 1. Fetch larger candidate set (2x n_results)
        candidates = self.query(query_embedding, n_results=n_results * 2)
        
        if not candidates or not candidates['ids'] or not candidates['ids'][0]:
            return []
            
        ids = candidates['ids'][0]
        distances = candidates['distances'][0]
        documents = candidates['documents'][0]
        metadatas = candidates['metadatas'][0] if candidates['metadatas'] else [{}] * len(ids)
        
        # 2. Score candidates
        # Base score = 1.0 - distance (Vector similarity)
        # Bonus = 0.2 if query terms appear in text
        
        query_terms = set(query.lower().split())
        scored_results = []
        
        for i in range(len(ids)):
            text = documents[i].lower()
            base_score = 1.0 - distances[i]
            
            # Simple term overlap bonus
            term_matches = sum(1 for term in query_terms if term in text)
            keyword_bonus = 0.1 * term_matches
            
            final_score = base_score + keyword_bonus
            
            scored_results.append({
                "id": ids[i],
                "score": final_score,
                "text": documents[i],
                "metadata": metadatas[i]
            })
            
        # 3. Sort and truncate
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:n_results]
