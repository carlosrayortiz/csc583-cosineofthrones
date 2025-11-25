"""
ANSWER_PROMPT templates for grounded and trivia-style answers
------------------------------------------------------------

This module defines the prompt templates used by the Cosine of Thrones
synthesizer node. You may extend this collection later with:

- strict fact-only mode
- verbose lore mode
- creative narrative mode
- character-perspective mode

Current templates:
- ANSWER_PROMPT: full grounded answer for normal pipeline use
- TRIVIA_ANSWER_PROMPT: short factual answer for FunTrivia evaluation
"""

# ============================================================
# Full grounded answer prompt (default)
# ============================================================

ANSWER_PROMPT = """
You are the Answer Synthesis Agent for a Game of Thrones retrieval system.

Your task is to generate a clear, factual answer to the user's question,
grounded ONLY in the provided evidence. Follow these rules:

1. Do NOT add details not supported by the evidence.
2. If the evidence contradicts itself, explain the contradiction.
3. If evidence is incomplete or ambiguous, make this explicit.
4. Write in a concise narrative style — 3 to 6 sentences.
5. NEVER speculate outside the evidence.
6. Base every claim on the evidence lines provided.

QUESTION:
{question}

EVIDENCE:
{evidence}

Now provide a grounded, contradiction-free answer based strictly on the evidence above.
"""


# ============================================================
# Trivia short-answer prompt (FunTrivia evaluation mode)
# ============================================================

TRIVIA_ANSWER_PROMPT = """
You are an expert Game of Thrones trivia bot.

Return ONLY the short factual answer.
No explanation.
No commentary.
No narrative.
Do NOT mention evidence.
Do NOT give multiple sentences.

Provide the short answer ONLY.

QUESTION:
{question}

EVIDENCE:
{evidence}

Short answer:
"""


# ============================================================
# Export
# ============================================================

__all__ = [
    "ANSWER_PROMPT",
    "TRIVIA_ANSWER_PROMPT",
]