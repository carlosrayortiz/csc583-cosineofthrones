# ==========================
# Agent 4 — NARRATIVE CONSISTENCY AGENT
# ==========================

import os
import json, re
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

# Load model name
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")

# Initialize LLM client (LangChain interface)
client = init_chat_model(GEN_MODEL, temperature=0)

NARRATIVE_PROMPT = """
You are the Narrative Consistency Agent for a Game of Thrones RAG system.

Your job is to review retrieved evidence lines and extract:
1. Core factual claims.
2. Character relationships and motivations.
3. Cause → effect chains.
4. Timeline alignment or contradictions.
5. A short "narrative-consistent summary" that merges overlapping facts.

INPUT FORMAT:
{
  "question": "...",
  "evidence": [
      "S6E10 ...",
      "S1E02 ...",
      "S?E? ...",
      ...
  ]
}

OUTPUT STRICT JSON:
{
  "facts": [...],
  "causal_links": [...],
  "character_entities": [...],
  "narrative_summary": "1-3 sentence merged, contradiction-free summary"
}
"""

@dataclass
class NarrativeResult:
    facts: list
    causal_links: list
    character_entities: list
    narrative_summary: str

def narrative_agent(question: str, evidence_list: list) -> NarrativeResult:
    """
    Calls the LLM to merge overlapping evidence into a causal and
    temporally consistent narrative summary.
    """

    payload = {
        "question": question,
        "evidence": evidence_list
    }

    # ---- LangChain invoke() instead of raw OpenAI client ----
    out = client.invoke([
        {"role": "system", "content": NARRATIVE_PROMPT},
        {"role": "user", "content": json.dumps(payload)}
    ])

    # out.content holds the LLM string
    raw = out.content.strip()

    # ---- Robust JSON parsing ----
    try:
        data = json.loads(raw)
    except Exception:
        # fallback: extract JSON substring
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise ValueError(f"Narrative Agent returned unparseable JSON:\n{raw}")
        data = json.loads(match.group(0))

    return NarrativeResult(
        facts=data.get("facts", []),
        causal_links=data.get("causal_links", []),
        character_entities=data.get("character_entities", []),
        narrative_summary=data.get("narrative_summary", "")
    )