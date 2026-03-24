"""
DocChat — Retrieval Pipeline Module
=====================================
Encapsulates adaptive top-k retrieval and cross-encoder reranking 
to minimize tokens sent to the LLM.
"""

import re
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline


# ---------------------------------------------------------------------------
# Query Complexity Classifier
# ---------------------------------------------------------------------------

# Keywords that signal complex/analytical queries
COMPLEX_KEYWORDS = re.compile(
    r"\b(compare|contrast|list all|explain|summarize|describe|analyze|"
    r"what are all|tell me everything|differences between|pros and cons|"
    r"advantages|disadvantages|how does .+ work)\b",
    re.IGNORECASE
)


def classify_query_complexity(query: str) -> int:
    """
    Determine how many chunks to retrieve based on query complexity.
    Simple factual lookups need fewer chunks; analytical questions need more.

    Returns k (number of chunks to retrieve before reranking).
    """
    word_count = len(query.split())

    if COMPLEX_KEYWORDS.search(query):
        return 8

    if word_count > 20:
        return 8
    elif word_count >= 8:
        return 5
    else:
        return 3


# ---------------------------------------------------------------------------
# Cross-Encoder Reranker (loaded once at module level for speed)
# ---------------------------------------------------------------------------

_reranker = None


def _get_reranker():
    """Lazy-load the cross-encoder reranker model (first call only)."""
    global _reranker
    if _reranker is None:
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        _reranker = CrossEncoderReranker(model=model, top_n=3)
    return _reranker


# ---------------------------------------------------------------------------
# Embeddings Filter (Context Compression)
# ---------------------------------------------------------------------------

def _build_compressor_pipeline(embeddings):
    """
    Build a compression pipeline:
    Stage 1: Cross-encoder reranker — re-scores chunks for accurate relevance
    """
    reranker = _get_reranker()

    return DocumentCompressorPipeline(
        transformers=[reranker]
    )


# ---------------------------------------------------------------------------
# Main Retriever Builder
# ---------------------------------------------------------------------------

def build_retriever(vectorstore, embeddings, query: str):
    """
    Build the full retrieval pipeline for a given query:
    1. Adaptive k based on query complexity
    2. Cross-encoder reranking (top 3)

    Returns a ContextualCompressionRetriever ready to call.
    """
    k = classify_query_complexity(query)

    # Base retriever with adaptive k
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    # Compression pipeline: reranker
    compressor = _build_compressor_pipeline(embeddings)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    ), k
