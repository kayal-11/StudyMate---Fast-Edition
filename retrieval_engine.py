from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict

class RetrievalEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.embeddings = None
    
    def build_index(self, chunks: List[Dict]):
        """Build FAISS index from text chunks"""
        self.chunks = chunks
        
        # Extract text for embedding
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        self.embeddings = self.model.encode(texts)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
    
    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve top-k most relevant chunks for a query"""
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search in index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return relevant chunks with scores
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(scores[0][i])
                relevant_chunks.append(chunk)
        
        return relevant_chunks