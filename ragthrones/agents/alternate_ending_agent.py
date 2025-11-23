"""
Alternate Ending Agent
----------------------

Generates a Season 8 final scene using ONLY Seasons 1–7 evidence.

This agent:
- Runs retrieval on user question
- Filters out any Season 8 chunks
- Synthesizes a screenplay-style alternate ending
- Provides justification based on S1–7 canon only
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from ragthrones.llm.llm_client import llm_chat
import pandas as pd


ALT_ENDING_PROMPT = """
You are generating an alternate Season 8 ending for *Game of Thrones*, but you must ONLY use canonical information from Seasons 1–7. Ignore all events, arcs, or outcomes introduced in Season 8.

TASK:
Create a structured alternate final scene for the character: {character}

Rules:
- Your ending **must not** reference or copy any plot points from Season 8.
- Infer plausible future actions ONLY from:
  - their Season 1–7 traits
  - their motivations
  - their unresolved conflicts
  - their alliances and enemies
- If other characters appear, their actions must also be derived ONLY from Seasons 1–7.

OUTPUT FORMAT (DO NOT DEVIATE):

# {LOCATION_OR_ARC_TITLE}
## {character}'s Turning Point
Describe the major decision or realization that pushes the character into their final arc. 
(3–5 sentences)

## {character}'s Final Act
Describe the climactic moment of the character’s final scene. 
(3–5 sentences)

## Symbolic Conclusion
Give a symbolic or thematic moment that concludes the character’s story 
(2–4 sentences)
"""

@dataclass
class AlternateEndingResult:
    scene: str
    justification: str
    variants: str


def alternate_ending_agent(question: str, df: pd.DataFrame) -> AlternateEndingResult:
    """
    Generate an alternate S8 ending using only S1–7 data.
    """

    if df is None or len(df) == 0:
        evidence_text = "No usable evidence from Seasons 1–7."
    else:
        # HARD FILTER: REMOVE SEASON 8 CHUNKS
        filtered = df[df["season"].astype(str) != "8"]
        if len(filtered) == 0:
            evidence_text = "No usable evidence after filtering out Season 8."
        else:
            evidence_text = "\n".join(
                f"[S{r.season}E{r.episode}] {r.text}"
                for _, r in filtered.iterrows()
            )

    prompt = ALT_ENDING_PROMPT + "\n\nEVIDENCE:\n" + evidence_text

    response = llm_chat(prompt)

    # We let the model output a combined block;
    # UI will display it directly as HTML.

    return AlternateEndingResult(
        scene=response,
        justification="(included in model output)",
        variants="(included in model output)",
    )