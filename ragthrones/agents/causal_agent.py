"""
Causal Agent
------------
Extracts cause-effect structure from retrieved evidence.

This agent:
- receives the question + evidence lines
- identifies discrete CAUSES
- identifies discrete EFFECTS
- builds explicit CAUSE → EFFECT chains
- returns a structured result for UI + downstream synthesis
"""

import json
import re
from dataclasses import dataclass, field

from ragthrones.llm.llm_client import llm_client, GEN_MODEL


# ============================================================
# Prompt
# ============================================================

CAUSAL_PROMPT = """
You are the Causality Analysis Agent inside a Game of Thrones RAG system.

Your goal is to extract cause-effect structure ONLY from the provided evidence.

You must output STRICT JSON with the following fields:

{
  "causes": [ "cause1", "cause2", ... ],
  "effects": [ "effect1", "effect2", ... ],
  "causal_links": [
      "cause → effect",
      ...
  ]
}

Guidelines:
- Use only facts found in the evidence (no outside knowledge).
- Causes should be short, atomic statements describing antecedent conditions.
- Effects should be short, atomic statements describing outcomes or consequences.
- Causal links must be explicit: "<cause> → <effect>".

Avoid interpretation, speculation, or emotional analysis.
"""


# ============================================================
# Result Dataclass
# ============================================================

@dataclass
class CausalResult:
    causes: list = field(default_factory=list)
    effects: list = field(default_factory=list)
    causal_links: list = field(default_factory=list)


# ============================================================
# Main Agent Function
# ============================================================

def causal_agent(question: str, evidence_lines: list) -> CausalResult:
    """
    Run causal extraction over the top reranked evidence.
    """

    payload = {
        "question": question,
        "evidence": evidence_lines
    }

    # ------ LLM CALL ------
    out = llm_client.chat.completions.create(
        model=GEN_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": CAUSAL_PROMPT},
            {"role": "user", "content": json.dumps(payload)}
        ]
    )

    raw = out.choices[0].message.content.strip()

    # ------ Robust JSON Parsing ------
    try:
        data = json.loads(raw)
    except Exception:
        # Try to salvage embedded JSON
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            data = json.loads(match.group(0))
        else:
            # Hard fail – return empty structure with error
            return CausalResult(
                causes=[],
                effects=[],
                causal_links=[]
            )

    return CausalResult(
        causes=data.get("causes", []),
        effects=data.get("effects", []),
        causal_links=data.get("causal_links", [])
    )