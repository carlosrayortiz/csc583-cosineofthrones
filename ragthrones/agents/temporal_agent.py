"""
Temporal Agent for Cosine of Thrones
------------------------------------

This agent analyzes a Game of Thrones question and determines:

- likely season range
- likely specific episodes
- timeline-anchored retrieval subqueries

It outputs strict JSON via OpenAI LLM, then parses into a TemporalResult object.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI
import os


# -------------------------------------------------------
# OpenAI client (same model used across your project)
# -------------------------------------------------------

GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")

def _get_llm_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


# -------------------------------------------------------
# Dataclass Result
# -------------------------------------------------------

@dataclass
class TemporalResult:
    season_range: Optional[str]
    episodes: List[str]
    timeline_queries: List[str]


# -------------------------------------------------------
# Prompt
# -------------------------------------------------------

TEMPORAL_PROMPT = """
You are the Temporal Agent for a Game of Thrones RAG system.

Your goal is to determine WHEN in the story the answer to the question likely occurs.

Given a question about Game of Thrones:

1. Identify the approximate season range of the relevant events.
2. Identify specific episode candidates (SxEy).
3. Produce timeline-focused retrieval subqueries WITH explicit temporal anchors.
4. Output STRICT JSON ONLY in this format:

{
  "season_range": "S4-S6" or null,
  "episodes": ["S6E10", "S6E03"] or [],
  "timeline_queries": [
     "...subquery 1...",
     "...subquery 2..."
  ]
}

Rules:
- Make sure all output is valid JSON.
- Episodes must be formatted as SxEy.
- Subqueries should include season or episode hints when possible.
"""


# -------------------------------------------------------
# Agent Function
# -------------------------------------------------------

def temporal_agent(question: str) -> TemporalResult:
    """
    Sends question to LLM, extracts temporal reasoning as JSON,
    returns TemporalResult.
    """

    client = _get_llm_client()

    response = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": TEMPORAL_PROMPT},
            {"role": "user", "content": question},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Try strict JSON first
    try:
        payload = json.loads(raw)
    except Exception:
        # fallback: extract nearest JSON block
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError(
                f"Temporal agent returned unparseable output:\n{raw}"
            )
        payload = json.loads(m.group(0))

    return TemporalResult(
        season_range=payload.get("season_range"),
        episodes=payload.get("episodes", []),
        timeline_queries=payload.get("timeline_queries", []),
    )