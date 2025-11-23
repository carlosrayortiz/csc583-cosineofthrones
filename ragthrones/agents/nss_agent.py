import json
from dataclasses import dataclass
from typing import Dict, Any

from ragthrones.llm.llm_client import get_llm_client


NSS_SYSTEM_PROMPT = """
You are the Narrative Scoring System (NSS) for the Cosine of Thrones multi-agent RAG project.
Your job is to EVALUATE a narrative answer, not to generate new content.

You score according to an 8-category rubric. Each category receives:
- score: an integer 1–5
- weight: provided below
- explanation: 1–3 sentences describing why the score was assigned

Use ONLY the provided evidence when judging correctness, consistency, or plausibility.
Do NOT invent new events, motivations, history, or details not found in the evidence.

===========================
NSS RUBRIC CATEGORIES
===========================

1. SETTING CONSISTENCY (weight 2)
   Measures whether the answer is grounded in the correct locations, world-state, timeline,
   and historically consistent facts of the GOT universe.
   Examples of errors:
   - placing a character in a location they never reached
   - violating war timelines
   - anachronistic technology or events

2. CHARACTER CONSISTENCY (weight 4)
   Measures whether character behavior matches their established personalities, arcs,
   speaking styles, and behavioral patterns from Seasons 1–7.
   Examples of errors:
   - bravery in place of an established coward
   - emotional tone mismatches
   - ignoring defining character traits

3. CHARACTER MOTIVATION (weight 4)
   Measures whether motivations in the answer align with what the character would plausibly want
   based on canonical desires, loyalties, traumas, and relationships.
   Examples of errors:
   - alliances that contradict history
   - sudden betrayals without cause

4. REFERENCING CONSISTENCY (weight 3)
   Measures whether the answer **accurately uses the provided evidence** and avoids contradictions.
   This includes:
   - quoting events correctly
   - not contradicting retrieved episodes
   - drawing correct causal links between scenes

5. CONFLICT / RESOLUTION LINKAGE (weight 4)
   Measures whether conflicts introduced in the answer logically connect to the resolutions.
   Does cause lead to effect?
   Are there missing steps or jumps?

6. THEME ALIGNMENT (weight 3)
   Measures whether the answer fits the major themes of the GOT world:
   - power corrupts
   - loyalty vs ambition
   - prophecy ambiguity
   - cycles of violence
   - political consequences
   Deviations should be justified by evidence.

7. MACROSTRUCTURE COHESION (weight 4)
   Measures whether the answer has:
   - beginning → escalation → climax → resolution flow
   - coherent scene progression
   - no plot holes or unnatural transitions
   - clear narrative rhythm

8. CREATIVE PLAUSIBILITY (weight 4; alt endings only)
   For alternate endings only:
   - Is the creative direction plausible within the world?
   - Does it feel believable given Seasons 1–7?
   - Does it avoid Season 8–specific canon unless explicitly supported by earlier evidence?

===========================
SCORING RULES  
===========================

Score meaning:
1 = very inconsistent / wrong / contradicts evidence
2 = weak or partially incorrect
3 = acceptable but with issues
4 = strong; minor flaws only
5 = excellent and fully consistent with evidence

===========================
OUTPUT FORMAT (STRICT JSON)
===========================

Return ONLY JSON with this structure:

{
  "scores": {
    "<category>": {
      "score": <int>,
      "weight": <int>,
      "weighted": <int>,
      "explanation": "<short reason>"
    },
    ...
  },
  "total_weighted_score": <int>
}

- weighted = score * weight
- Sum all weighted scores into total_weighted_score
- No extra text, no Markdown, STRICT JSON only.

===========================
EVALUATION PRINCIPLES
===========================

- Base all judgments on the provided evidence.
- If evidence is missing or incomplete, score based on internal consistency.
- Penalize creative hallucinations that contradict GOT canon.
- Be objective and follow the rubric exactly.

You are an evaluator, not an author.
"""

RUBRIC = {
    "setting_consistency": 2,
    "character_consistency": 4,
    "character_motivation": 4,
    "referencing_consistency": 3,
    "conflict_resolution_linkage": 4,
    "theme_alignment": 3,
    "macrostructure_cohesion": 4,
    "creative_plausibility": 4
}

def scoring_agent(state):
    from ragthrones.llm.llm_client import get_llm_client

    client = get_llm_client()  # returns an OpenAI client, not LangChain

    # 1. Answer
    answer = getattr(state, "answer", "") or ""

    # 2. Evidence
    if getattr(state, "reranked", None) is not None and not state.reranked.empty:
        evidence_df = state.reranked
    elif getattr(state, "retrieved", None) is not None and not state.retrieved.empty:
        evidence_df = state.retrieved
    else:
        evidence_df = None

    evidence_records = (
        evidence_df.to_dict(orient="records")
        if evidence_df is not None else []
    )

    payload = {
        "answer": answer,
        "evidence": evidence_records
    }

    messages = [
        {"role": "system",    "content": NSS_SYSTEM_PROMPT},
        {"role": "user",      "content": json.dumps(payload)}
    ]

    # 3. Correct OpenAI v1 call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )

    # 4. Parse JSON
   # 4. Parse JSON safely
    try:
        content = response.choices[0].message.content
        result = json.loads(content)
    except Exception as e:
        result = {"error": "Invalid JSON", "details": str(e)}

    # 5. Save to state
    state.nss_score = result
    return state