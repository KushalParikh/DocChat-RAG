"""
DocChat — Semantic Cache Module
================================
In-memory semantic cache that matches queries by embedding similarity.
Eliminates redundant LLM calls for semantically identical questions.
"""

import time
import numpy as np


class SemanticCache:
    """
    Cache that stores query embeddings and their answers.
    On new queries, checks cosine similarity against all cached embeddings.
    If similarity > threshold, returns the cached answer (zero LLM cost).
    """

    def __init__(self, embeddings, threshold: float = 0.92):
        """
        Args:
            embeddings: The embedding model (same one used for retrieval).
            threshold: Cosine similarity threshold for cache hits. 
                       0.92 = conservative (fewer false hits).
        """
        self.embeddings = embeddings
        self.threshold = threshold
        self._cache = []  # List of {embedding, query, answer, sources, timestamp}
        self._hits = 0
        self._misses = 0

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        a = np.array(a)
        b = np.array(b)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0

    def get(self, query: str):
        """
        Check cache for a semantically similar query.
        Returns (answer, sources) if found, None otherwise.
        """
        if not self._cache:
            self._misses += 1
            return None

        # Skip caching for time-sensitive queries
        time_keywords = ["latest", "current", "today", "now", "recent", "updated"]
        if any(kw in query.lower() for kw in time_keywords):
            self._misses += 1
            return None

        query_embedding = self.embeddings.embed_query(query)

        best_sim = 0.0
        best_entry = None

        for entry in self._cache:
            sim = self._cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_sim >= self.threshold and best_entry is not None:
            self._hits += 1
            return best_entry["answer"], best_entry.get("sources", [])

        self._misses += 1
        return None

    def set(self, query: str, answer: str, sources: list = None):
        """Store a query-answer pair in the cache."""
        # Don't cache empty or error responses
        if not answer or not answer.strip():
            return
        if "I cannot find the answer" in answer:
            return

        query_embedding = self.embeddings.embed_query(query)

        self._cache.append({
            "embedding": query_embedding,
            "query": query,
            "answer": answer,
            "sources": sources or [],
            "timestamp": time.time(),
        })

    def invalidate(self):
        """Clear all cached entries. Call when new documents are uploaded."""
        self._cache.clear()

    def stats(self) -> dict:
        """Return cache performance statistics."""
        total = self._hits + self._misses
        return {
            "total_entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{(self._hits / total * 100):.1f}%" if total > 0 else "0.0%",
        }
