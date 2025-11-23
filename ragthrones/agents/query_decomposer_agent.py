"""
Query Decomposer Agent for Cosine of Thrones
-------------------------------------------

This agent:
- Extracts entities via spaCy
- Sends an LLM prompt to analyze GOT questions
- Returns structured JSON needed by downstream agents
- Produces canonicalized entities, subqueries, temporal hints,
  and retrieval-friendly rewritten queries.
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Any

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

import spacy

from dotenv import load_dotenv
load_dotenv()


# -------------------------------------------------------
# Load spaCy NER (same model you used in the notebook)
# -------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not installed. Run: python -m spacy download en_core_web_sm"
    )

# -------------------------------------------------------
# LLM for decomposition
# -------------------------------------------------------
GEN_MODEL = "gpt-4o-mini"
QueryDecomposerLLM = init_chat_model(GEN_MODEL, temperature=0.0)


# -------------------------------------------------------
# ParsedQuery dataclass
# -------------------------------------------------------
@dataclass
class ParsedQuery:
    question: str
    question_type: str
    entities: List[str]
    canonical_entities: List[str]
    subqueries: List[str]
    temporal_hints: Dict[str, Any]
    retrieval_queries: List[str]


# -------------------------------------------------------
# SpaCy Entity Extraction
# -------------------------------------------------------
def extract_spacy_entities(text: str) -> List[str]:
    doc = nlp(text)
    return [ent.text for ent in doc.ents]


# -------------------------------------------------------
# LLM Prompt
# -------------------------------------------------------
DECOMPOSER_PROMPT = """
You are an expert Game of Thrones question analyst.

Your job is to break the user's question into structured components
for retrieval in a RAG pipeline.

Return your output STRICTLY as a JSON dict with the following fields:

{
 "question_type": "...",
 "entities": [...],
 "canonical_entities": [...],
 "subqueries": [...],
 "temporal_hints": {
    "season": null or number,
    "episode": null or number,
    "approx_range": null or "SxEy-SxEz"
 },
 "retrieval_queries": [...]
}

Rules:
- Identify all GOT characters, places, objects, and factions.
- Convert nicknames to canonical names (example: "The Imp" → "Tyrion Lannister")
- Decompose "why/how" questions into multiple sub-questions.
- Add retrieval-friendly phrasing for each subquery.
- If no season/episode is given, infer approximate seasons when possible.
- Retrieval queries should be suitable for the hybrid_retrieve tool.
"""


# -------------------------------------------------------
# Main Decomposer Agent
# -------------------------------------------------------
def query_decomposer_agent(question: str) -> ParsedQuery:
    """Break a GOT question into structured retrieval instructions."""

    # SpaCy baseline entities
    spacy_entities = extract_spacy_entities(question)

    # Prepare messages
    messages = [
        SystemMessage(content=DECOMPOSER_PROMPT),
        HumanMessage(
            content=f"Question: {question}\nDetected entities: {spacy_entities}"
        ),
    ]

    # Execute LLM
    out = QueryDecomposerLLM.invoke(messages).content

    # Parse JSON
    parsed = json.loads(out)

    return ParsedQuery(
        question=question,
        question_type=parsed["question_type"],
        entities=parsed["entities"],
        canonical_entities=parsed["canonical_entities"],
        subqueries=parsed["subqueries"],
        temporal_hints=parsed["temporal_hints"],
        retrieval_queries=parsed["retrieval_queries"],
    )