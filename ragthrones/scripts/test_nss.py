"""
Standalone NSS scoring test
---------------------------

This script tests your Narrative Scoring System (NSS) agent
without running the full LangGraph pipeline.

It mirrors the lightweight style of the existing scripts in this directory.
"""

import pandas as pd
from types import SimpleNamespace

from ragthrones.agents.nss_agent import scoring_agent

# -------------------------------
# Fake Answer for Testing
# -------------------------------
answer_text = (
    "Cersei destroyed the Sept of Baelor to eliminate her political enemies "
    "and secure her claim to the Iron Throne after Tommen became vulnerable."
)

# -------------------------------
# Fake Evidence DataFrame
# -------------------------------
evidence_rows = [
    {
        "season": 6,
        "episode": 10,
        "text": "Cersei orchestrates the explosion of the Sept using wildfire."
    },
    {
        "season": 6,
        "episode": 10,
        "text": "The wildfire detonation kills the High Sparrow, Margaery Tyrell, "
                "and many major political figures."
    },
]

evidence_df = pd.DataFrame(evidence_rows)

# -------------------------------
# Mock Minimal AgentState
# -------------------------------
state = SimpleNamespace(
    answer=answer_text,
    retrieved=evidence_df,
    reranked=evidence_df,
    nss_score=None,
)

# -------------------------------
# Run the NSS agent
# -------------------------------
updated = scoring_agent(state)

# -------------------------------
# Print Result
# -------------------------------
print("\n=== NSS SCORING OUTPUT ===")
print(updated.nss_score)

# Pretty-print version (optional)
try:
    import json
    print("\n=== Pretty JSON ===")
    print(json.dumps(updated.nss_score, indent=2))
except:
    pass