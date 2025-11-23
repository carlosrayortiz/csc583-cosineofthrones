"""
Cosine / RAG Answer Pipeline for RAGThrones

This module:
- Runs hybrid retrieval (FAISS + BM25)
- Builds a context string from top passages
- Calls an LLM to generate an answer
- Returns answer + retrieved evidence
"""

import os
import pandas as pd

from openai import OpenAI

from ragthrones.retrieval.hybrid_search import hybrid_search
from ragthrones.retrieval.evidence_builder import build_evidence_html


# -----------------------------
# LLM client setup
# -----------------------------

GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")


def _get_llm_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=api_key)


# -----------------------------
# Helper: build context string
# -----------------------------

def _build_context_from_hits(hits: pd.DataFrame, max_chunks: int = 8) -> str:
    """
    Build a plain-text context string from top retrieval hits.
    """
    parts = []
    for _, row in hits.head(max_chunks).iterrows():
        text = str(row.get("text", "")).strip()
        score = float(row.get("score", 0.0))
        parts.append(f"[score={score:.3f}] {text}")
    return "\n\n".join(parts)


# -----------------------------
# Helper: call LLM
# -----------------------------

def _generate_answer(query: str, context: str) -> str:
    """
    Call the LLM with the query and retrieved context.
    """
    client = _get_llm_client()

    system_prompt = (
        "You are an expert assistant for Game of Thrones lore. "
        "Use ONLY the provided context to answer the question. "
        "If the context is insufficient, say you are not sure instead of guessing."
    )

    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer concisely in 3–5 sentences."
    )

    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()


# -----------------------------
# Public pipeline function
# -----------------------------

def run_cosine_pipeline(
    query: str,
    topk: int = 10,
    alpha: float = 0.35,
    cand_mult: int = 20,
) -> dict:
    """
    Full RAGThrones Q&A pipeline.

    Steps:
    - Run hybrid retrieval to get hits
    - Build context from hits
    - Generate answer with LLM
    - Build evidence HTML for UI

    Returns a dict with:
        {
            "query": str,
            "answer": str,
            "hits": pd.DataFrame,
            "evidence_html": str,
        }
    """

    # 1) Retrieval
    hits = hybrid_search(
        query=query,
        topk=topk,
        alpha=alpha,
        cand_mult=cand_mult,
    )

    if hits is None or hits.empty:
        return {
            "query": query,
            "answer": "I couldn’t find any relevant passages to answer that.",
            "hits": pd.DataFrame([]),
            "evidence_html": "<p>No evidence found.</p>",
        }

    # 2) Context
    context = _build_context_from_hits(hits)

    # 3) LLM answer
    answer = _generate_answer(query, context)

    # 4) Evidence HTML (for Gradio / UI)
    evidence_html = build_evidence_html(hits)

    return {
        "query": query,
        "answer": answer,
        "hits": hits,
        "evidence_html": evidence_html,
    }