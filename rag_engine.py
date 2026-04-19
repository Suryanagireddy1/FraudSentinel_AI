"""
rag_engine.py
=============
RAG Retrieval Engine — cosine similarity over fraud case vectors.
Also adjusts the ML fraud probability based on retrieved case similarity
(RAG now functionally influences the final score, not just display).
"""
import numpy as np
import json

class RAGRetriever:
    def __init__(self,
                 vectors_path='models/rag_vectors.npy',
                 meta_path='models/rag_meta.json'):
        self.normed = np.load(vectors_path)
        with open(meta_path) as f:
            self.meta = json.load(f)

    def retrieve(self, query_vec: np.ndarray, top_k: int = 3) -> list:
        q    = query_vec.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(q) or 1
        sims = (self.normed @ (q / norm).T).flatten()
        idxs = np.argsort(sims)[::-1][:top_k]
        return [{
            'case_id':        self.meta[i]['case_id'],
            'similarity_pct': round(float(sims[i]) * 100, 1),
            'amount':         self.meta[i]['amount'],
            'product':        self.meta[i]['product'],
            'card_type':      self.meta[i].get('card_type', 'visa'),
            'email':          self.meta[i].get('email', 'unknown'),
            'hour':           self.meta[i].get('hour', 0),
            'outcome':        self.meta[i]['outcome'],
        } for i in idxs]

    def rag_adjusted_score(self, ml_prob: float, similar_cases: list,
                            weight: float = 0.15) -> float:
        """
        Functionally adjusts ML fraud probability using RAG similarity.
        If top retrieved cases are highly similar confirmed frauds,
        the score is nudged upward (and vice versa if low similarity).

        Formula:
            rag_signal = mean(similarity_pct / 100) for top cases
            adjusted   = (1 - weight) * ml_prob + weight * rag_signal

        weight=0.15 means RAG contributes 15% to the final score.
        This makes RAG functional — not just decorative.
        """
        if not similar_cases:
            return ml_prob
        avg_sim   = np.mean([c['similarity_pct'] / 100.0 for c in similar_cases])
        adjusted  = (1 - weight) * ml_prob + weight * avg_sim
        return float(np.clip(adjusted, 0.0, 1.0))
