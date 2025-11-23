"""
Emotion Agent
-------------
Extracts emotional indicators for characters from the RAG evidence.

This agent:
- Reads the retrieved evidence lines
- Identifies characters mentioned
- Extracts emotional states (anger, fear, grief, determination, etc.)
- Extracts overall sentiment orientation (positive / negative / conflicted)
- Returns structured JSON only
"""

import json
import re
from dataclasses import dataclass, field

from ragthrones.llm.llm_client import get_llm_client, GEN_MODEL


# ============================================================
# Prompt
# ============================================================

EMOTION_PROMPT = """
You are the Emotion Analysis Agent in a Game of Thrones RAG system.

Your task is to extract emotional indicators about characters
based ONLY on the provided evidence.

You must output STRICT JSON:

{
  "character_entities": [
      "Daenerys Targaryen - emotion or mental state",
      "Jon Snow - emotion or mental state",
      ...
  ],
  "emotional_state": [
      "anger", "fear", "grief", "determination", ...
  ],
  "sentiment": "positive" | "negative" | "conflicted" | ""
}

Guidelines:
- Use ONLY the provided evidence.
- Identify characters and pair them with the emotional state implied.
- Emotional state must come from the *tone* or *descriptions* in the evidence.
- Do NOT invent events, motives, or unstated emotions.
- Sentiment should summarize the emotional tone of the evidence as a whole.
- Keep entries short and specific.

Return ONLY JSON. No prose.
"""


# ============================================================
# Result Dataclass
# ============================================================

@dataclass
class EmotionResult:
    character_entities: list = field(default_factory=list)
    emotional_state: list = field(default_factory=list)
    sentiment: str = ""


# ============================================================
# Main Agent Function
# ============================================================

def emotion_agent(question: str, evidence_lines: list) -> EmotionResult:
    """
    Run emotional/affect extraction over evidence.
    """

    client = get_llm_client()

    payload = {
        "question": question,
        "evidence": evidence_lines
    }

    # ---- Call LLM (correct OpenAI API format) ----
    try:
        response = client.chat.completions.create(
            model=GEN_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": EMOTION_PROMPT},
                {"role": "user", "content": json.dumps(payload)}
            ]
        )

        #raw = response.choices[0].message["content"].strip()
        raw = response.choices[0].message.content.strip()

    except Exception as e:
        return EmotionResult(
            character_entities=[f"ERROR: {str(e)}"],
            emotional_state=[],
            sentiment=""
        )

    # ---- JSON parsing with fallback ----
    try:
        data = json.loads(raw)
    except Exception:
        # Try to extract a JSON object embedded in text
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            data = json.loads(match.group(0))
        else:
            return EmotionResult(
                character_entities=[],
                emotional_state=[],
                sentiment=""
            )

    return EmotionResult(
        character_entities=data.get("character_entities", []),
        emotional_state=data.get("emotional_state", []),
        sentiment=data.get("sentiment", "")
    )