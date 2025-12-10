import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import numpy as np
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

class VectorStore:
    """
    Manages storage and retrieval of embeddings using ChromaDB.
    Supports Hybrid Search (Vector + BM25) via Reciprocal Rank Fusion (RRF).
    """
    
    def __init__(self, collection_name: str = "documents", path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # In-memory BM25 index
        self.bm25 = None
        self.bm25_doc_ids = []
        self.bm25_corpus = []

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
            
            # Invalidate BM25 index so it rebuilds on next query
            self.bm25 = None
            
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            pass

    def _build_bm25_index(self):
        """
        Fetches all documents from ChromaDB and builds an in-memory BM25 index.
        Note: This is expensive for large datasets.
        """
        if not BM25Okapi:
            print("rank_bm25 not installed. Hybrid search will fallback to vector only.")
            return

        try:
            # Fetch all documents
            result = self.collection.get()
            if not result or not result['documents']:
                return

            self.bm25_doc_ids = result['ids']
            self.bm25_corpus = result['documents']
            
            # Simple tokenization
            tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
        except Exception as e:
            print(f"Error building BM25 index: {e}")

    def query(self, query_embeddings: List[float], n_results: int = 5):
        """
        Queries the vector store for similar documents (Vector Only).
        """
        return self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=n_results
        )

    def hybrid_search(self, query: str, query_embedding: List[float], n_results: int = 5, k: int = 60) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search combining Vector Search and BM25 using Reciprocal Rank Fusion (RRF).
        RRF Score = 1 / (k + rank_vector) + 1 / (k + rank_bm25)
        """
        # 1. Vector Search
        vector_res = self.query(query_embedding, n_results=n_results * 2) # Fetch more for fusion
        
        vector_hits = {} # id -> rank
        vector_data = {} # id -> (doc, metadata, distance)
        
        if vector_res and vector_res['ids'] and vector_res['ids'][0]:
            ids = vector_res['ids'][0]
            docs = vector_res['documents'][0]
            metas = vector_res['metadatas'][0] if vector_res['metadatas'] else [{}] * len(ids)
            dists = vector_res['distances'][0]
            
            for rank, doc_id in enumerate(ids):
                vector_hits[doc_id] = rank
                vector_data[doc_id] = (docs[rank], metas[rank], dists[rank])

        # 2. BM25 Search
        bm25_hits = {} # id -> rank
        
        if not self.bm25:
            self._build_bm25_index()
            
        if self.bm25:
            tokenized_query = query.lower().split()
            # Get scores for all docs in index
            doc_scores = self.bm25.get_scores(tokenized_query)
            # Get top indices
            top_n = min(len(doc_scores), n_results * 2)
            top_indices = np.argsort(doc_scores)[::-1][:top_n]
            
            for rank, idx in enumerate(top_indices):
                doc_id = self.bm25_doc_ids[idx]
                bm25_hits[doc_id] = rank
                # Ensure we have data for BM25-only hits
                if doc_id not in vector_data:
                    vector_data[doc_id] = (self.bm25_corpus[idx], {}, 1.0) # Placeholder distance

        # 3. Reciprocal Rank Fusion
        all_ids = set(vector_hits.keys()) | set(bm25_hits.keys())
        scored_results = []
        
        for doc_id in all_ids:
            rank_v = vector_hits.get(doc_id, float('inf'))
            rank_bm = bm25_hits.get(doc_id, float('inf'))
            
            score = 0.0
            if rank_v != float('inf'):
                score += 1.0 / (k + rank_v + 1)
            if rank_bm != float('inf'):
                score += 1.0 / (k + rank_bm + 1)
                
            if doc_id in vector_data:
                doc_text, meta, dist = vector_data[doc_id]
                scored_results.append({
                    "id": doc_id,
                    "score": score,
                    "text": doc_text,
                    "metadata": meta,
                    "vector_dist": dist
                })

        # 4. Sort and return
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:n_results]
