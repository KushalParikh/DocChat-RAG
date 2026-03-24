"""
DocChat — Query Monitoring Module
===================================
Lightweight query logging and cost estimation.
Tracks cache hits, chunks used, and estimated Gemini API costs.
"""

import time


class QueryMonitor:
    """Tracks query statistics for the admin dashboard."""

    # Gemini 2.5 Flash pricing (per 1M tokens) — updated March 2025
    # Free tier: generous but rate-limited. Paid: $0.15/1M input, $0.60/1M output
    INPUT_COST_PER_TOKEN = 0.15 / 1_000_000
    OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000

    def __init__(self):
        self._queries = []

    def log_query(self, query: str, cache_hit: bool, chunks_used: int,
                  response_length: int, k_used: int, latency_ms: float = 0):
        """Record a single query event."""
        self._queries.append({
            "query": query[:100],  # Truncate for storage
            "cache_hit": cache_hit,
            "chunks_used": chunks_used,
            "response_length": response_length,
            "k_used": k_used,
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        })

    def get_stats(self) -> dict:
        """Return aggregate statistics for the dashboard."""
        total = len(self._queries)
        if total == 0:
            return {
                "total_queries": 0,
                "cache_hits": 0,
                "cache_hit_rate": "0.0%",
                "avg_chunks_per_query": 0,
                "avg_latency_ms": 0,
                "estimated_cost": "$0.00",
            }

        cache_hits = sum(1 for q in self._queries if q["cache_hit"])
        llm_queries = [q for q in self._queries if not q["cache_hit"]]

        avg_chunks = (
            sum(q["chunks_used"] for q in llm_queries) / len(llm_queries)
            if llm_queries else 0
        )

        avg_latency = sum(q["latency_ms"] for q in self._queries) / total

        # Rough cost estimate: ~500 tokens per chunk input, response_length chars ≈ tokens/4
        total_input_tokens = sum(
            q["chunks_used"] * 500 for q in llm_queries
        )
        total_output_tokens = sum(
            q["response_length"] // 4 for q in llm_queries
        )
        estimated_cost = (
            total_input_tokens * self.INPUT_COST_PER_TOKEN +
            total_output_tokens * self.OUTPUT_COST_PER_TOKEN
        )

        return {
            "total_queries": total,
            "cache_hits": cache_hits,
            "cache_hit_rate": f"{(cache_hits / total * 100):.1f}%",
            "avg_chunks_per_query": round(avg_chunks, 1),
            "avg_latency_ms": round(avg_latency),
            "estimated_cost": f"${estimated_cost:.4f}",
            "queries_saved_by_cache": cache_hits,
        }
