"""
Reranker Agent for Cosine of Thrones
------------------------------------

Uses a Sentence-Transformers CrossEncoder to rerank retrieved chunks
based on semantic similarity between the user question and each chunk.

This is called inside node_reranker OR directly from pipeline code.

Exposes:
- get_reranker(): loads model once, caches globally
- rerank(df, question): returns reranked DataFrame
"""

import os
import pandas as pd
from typing import Optional

from sentence_transformers import CrossEncoder


# -------------------------------------------------------
# Lazy-loaded singleton reranker
# -------------------------------------------------------

_RERANKER = None

def get_reranker() -> Optional[CrossEncoder]:
    """
    Load reranker model if available.
    Returns None if model cannot be loaded.
    """
    global _RERANKER

    if _RERANKER is not None:
        return _RERANKER

    model_name = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    try:
        print(f"Loading reranker model: {model_name}")
        _RERANKER = CrossEncoder(model_name)
        return _RERANKER

    except Exception as e:
        print(f"WARNING: Could not load reranker model: {e}")
        _RERANKER = None
        return None


# -------------------------------------------------------
# Reranking functionality
# -------------------------------------------------------

def rerank(df: pd.DataFrame, question: str) -> pd.DataFrame:
    """
    Apply cross-encoder reranking.
    Returns df sorted by semantic score.

    IMPORTANT:
    - Expects df['text'] to exist
    - Returns df with new column 'rerank_score'

    If reranker model is not available, returns df unchanged.
    """
    if df is None or len(df) == 0:
        return df

    reranker = get_reranker()
    if reranker is None:
        print("Reranker unavailable -> skipping rerank.")
        return df

    # Build input pairs
    pairs = [[question, t] for t in df["text"].tolist()]

    # Predict scores
    scores = reranker.predict(pairs)

    # Create sorted output
    out = df.copy().reset_index(drop=True)
    out["rerank_score"] = scores
    out = out.sort_values("rerank_score", ascending=False).reset_index(drop=True)

    return out