"""
Shared helper functions for Cosine of Thrones
----------------------------------------------

Includes:
- Evidence formatting
- Reranker node adapter
- Synthesizer node adapter
- Lightweight entity extraction
- Question type classification
"""

import os
import json
import re
from typing import Optional

import pandas as pd
from openai import OpenAI
from ragthrones.prompts.answer_prompt import ANSWER_PROMPT
from ragthrones.prompts.answer_prompt import TRIVIA_ANSWER_PROMPT


# -------------------------------------------------------
# OpenAI client helper
# -------------------------------------------------------

GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")

def _get_llm_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set.")
    return OpenAI(api_key=api_key)


# -------------------------------------------------------
# 1. Evidence formatting
# -------------------------------------------------------

def format_evidence_rows(df: pd.DataFrame, k: int = 5) -> str:
    """
    Format reranked evidence rows into clean text lines.
    """
    if df is None or len(df) == 0:
        return "(no evidence)"

    lines = []
    for _, r in df.head(k).iterrows():
        s, e = r.get("season"), r.get("episode")

        # Safe season/episode formatter
        if pd.notna(s) and pd.notna(e):
            try:
                tag = f"S{int(s)}E{int(e)}"
            except Exception:
                tag = "S?E?"
        else:
            tag = "S?E?"

        txt = str(r.get("text", "")).replace("\n", " ").strip()
        spk = r.get("speaker")

        prefix = f"[{tag}] {spk}: " if isinstance(spk, str) and spk.strip() else f"[{tag}] "
        lines.append(prefix + txt)

    return "\n".join(lines)


# -------------------------------------------------------
# 2. Reranker node adapter
# -------------------------------------------------------

def node_reranker(state, reranker_model=None):
    """
    Insert reranked results into state.
    If no reranker model is provided, pass-through.
    """

    if (
        state.retrieved is None 
        or len(state.retrieved) == 0 
        or reranker_model is None
    ):
        state.reranked = state.retrieved
        state.logs["reranker"] = {"used": False, "reason": "no-hits-or-no-model"}
        return state

    pairs = [[state.question, txt] for txt in state.retrieved["text"].tolist()]
    scores = reranker_model.predict(pairs)

    df = state.retrieved.copy().reset_index(drop=True)
    df["rerank_score"] = scores
    df = df.sort_values("rerank_score", ascending=False).reset_index(drop=True)

    state.reranked = df
    state.logs["reranker"] = {
        "used": True,
        "top_score": float(df.iloc[0]["rerank_score"])
    }

    return state


# -------------------------------------------------------
# 3. Synthesizer node adapter (final LLM answer)
# -------------------------------------------------------

def node_synthesizer(
    state,
    answer_prompt_template=None,
    k_evidence: int = 5,
    show_prompt: bool = False
):
    """
    Synthesizer with Trivia Mode:
    - Detects trivia-style questions (short factoid answers)
    - Forces use of TRIVIA_ANSWER_PROMPT when needed
    - Preserves canonical entity merging logic
    """

    client = _get_llm_client()

    # -------------------------------------------------
    # Choose prompt template (auto-detect trivia)
    # -------------------------------------------------
    q = state.question.lower()

    TRIVIA_TRIGGERS = [
        "who", "what", "where", "when",
        "which", "name", "called", "known as",
        "sigil", "bastard", "killed", "father", "mother"
    ]

    is_trivia_question = any(t in q for t in TRIVIA_TRIGGERS)

    # Manual override if you want:
    if getattr(state, "trivia_mode", False):
        is_trivia_question = True

    # Select prompt
    if is_trivia_question:
        prompt_template = TRIVIA_ANSWER_PROMPT
    else:
        prompt_template = answer_prompt_template or ANSWER_PROMPT

    # -------------------------------------------------
    # Evidence selection (same as before)
    # -------------------------------------------------
    hits = state.reranked if state.reranked is not None else state.retrieved
    if hits is None or len(hits) == 0:
        state.answer = "(no evidence)"
        return state

    canonical_entities = (
        state.logs
        .get("decomposer", {})
        .get("canonical_entities", [])
    )
    canonical_entities = [c for c in canonical_entities if isinstance(c, str)]

    top_rows = hits.head(k_evidence)

    if canonical_entities:
        mask = hits["text"].apply(
            lambda t: any(name.lower() in str(t).lower() for name in canonical_entities)
        )
        entity_rows = hits[mask]
        merged = pd.concat([top_rows, entity_rows]).drop_duplicates()
    else:
        merged = top_rows

    # -------------------------------------------------
    # Build evidence text
    # -------------------------------------------------
    ev_text = "\n".join(
        f"[S{r.get('season')}E{r.get('episode')}] {r.get('text')}"
        for _, r in merged.iterrows()
    )

    state.evidence_text = ev_text

    # -------------------------------------------------
    # Build the full prompt
    # -------------------------------------------------
    full_prompt = prompt_template.format(
        question=state.question,
        evidence=ev_text
    )

    if show_prompt:
        print("\n=======================")
        print(full_prompt)
        print("=======================\n")

    # -------------------------------------------------
    # Call LLM
    # -------------------------------------------------
    response = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=0.1 if not is_trivia_question else 0.0,
        messages=[{"role": "user", "content": full_prompt}]
    )

    state.answer = response.choices[0].message.content.strip()

    # -------------------------------------------------
    # Logs
    # -------------------------------------------------
    state.logs["synthesizer"] = {
        "prompt_used": "TRIVIA_ANSWER_PROMPT" if is_trivia_question else "ANSWER_PROMPT",
        "prompt_length_chars": len(full_prompt),
        "evidence_count": len(merged)
    }

    return state

# -------------------------------------------------------
# 4. Heuristic entity extraction
# -------------------------------------------------------

def extract_entities(question: str):
    tokens = question.replace("?", "").split()
    ents, curr = [], []

    for tok in tokens:
        if tok and tok[0].isupper():
            curr.append(tok)
        else:
            if curr:
                ents.append(" ".join(curr))
                curr = []
    if curr:
        ents.append(" ".join(curr))

    return [e.strip() for e in ents if len(e.strip()) > 1]


# -------------------------------------------------------
# 5. Question type classification
# -------------------------------------------------------

def guess_question_type(question: str):
    q = question.lower()

    if q.startswith("why") or "cause" in q or "reason" in q:
        return "causal"

    if q.startswith("when") or "episode" in q or "season" in q or "timeline" in q:
        return "temporal"

    if any(k in q for k in ["who", "what", "where", "mother", "father", "identity"]):
        return "factual"

    return "general"