"""
Multi-Agent LangGraph Orchestrator for Cosine of Thrones
--------------------------------------------------------

Coordinates:
- Query Decomposer Agent
- Temporal Agent
- Narrative Agent
- Causality Agent
- Emotion Agent
- Basic RAG (now uses hybrid_search_aug directly)
- Alternate Ending Agent (creative, S1–S7 only)
- Reranker
- Synthesizer

This version bypasses the tool-based RetrievalAgent and calls
the proven-working hybrid_search_aug() directly for retrieval.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Iterable, List

import pandas as pd
from langgraph.graph import StateGraph, START, END

# --- Agents ---
from ragthrones.agents.query_decomposer_agent import query_decomposer_agent
from ragthrones.agents.temporal_agent import temporal_agent
from ragthrones.agents.narrative_agent import narrative_agent
from ragthrones.agents.causal_agent import causal_agent
from ragthrones.agents.emotion_agent import emotion_agent
from ragthrones.agents.basic_rag_agent import basic_rag_agent  # kept for flexibility
from ragthrones.agents.reranker_agent import get_reranker
from ragthrones.agents.alternate_ending_agent import alternate_ending_agent
from ragthrones.agents.nss_agent import scoring_agent

# --- Shared helpers / prompts / retrieval ---
from ragthrones.shared.helpers import node_reranker, node_synthesizer
from ragthrones.prompts.answer_prompt import ANSWER_PROMPT
from ragthrones.retrieval.hybrid_search import hybrid_search_aug


# ---------------------------------------------------------------
#                    STATE MODEL
# ---------------------------------------------------------------

@dataclass
class AgentState:
    # Core input
    question: str

    # Retrieval + reranking
    retrieved: Optional[pd.DataFrame] = None
    reranked: Optional[pd.DataFrame] = None
    evidence_text: str = ""

    # Final synthesized answer
    answer: Optional[str] = None

    # Agent analysis outputs
    causal: Dict[str, Any] = field(default_factory=dict)
    emotion: Dict[str, Any] = field(default_factory=dict)
    narrative: Dict[str, Any] = field(default_factory=dict)

    # Narrative scoring system (NSS) output
    nss_score: Optional[Dict[str, Any]] = None  # <-- NEW

    # Logs (all agents write here)
    logs: Dict[str, Any] = field(default_factory=dict)

    # Router decision
    route_decision: Optional[str] = None


# ---------------------------------------------------------------
#                    SHARED RETRIEVAL HELPERS
# ---------------------------------------------------------------

def _retrieve_with_hybrid(queries: Iterable[str], topk: int = 10) -> pd.DataFrame:
    """
    Run hybrid_search_aug over one or more queries, merge and dedupe results.
    """
    if isinstance(queries, str):
        queries = [queries]

    dfs = []
    for q in queries:
        q = q.strip()
        if not q:
            continue
        df = hybrid_search_aug(q, topk=topk)
        if df is not None and len(df):
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return merged


def _make_evidence_lines(df: pd.DataFrame, max_lines: int = 12) -> List[str]:
    """
    Turn reranked rows into short evidence lines:
    [SxEy] text...
    """
    lines: List[str] = []
    for _, r in df.head(max_lines).iterrows():
        season = r.get("season", "?")
        episode = r.get("episode", "?")
        tag = f"S{season}E{episode}" if str(season).isdigit() else "S?E?"
        text = str(r.get("text", "")).strip()
        lines.append(f"[{tag}] {text}")
    return lines


def _filter_to_pre_s8(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter retrieval results to use only Seasons 1–7.
    Strategy:
      - Keep rows with numeric season <= 7.
      - For rows without a parseable season, drop anything that *clearly*
        references Season 8 or post-finale meta-content.
    """
    if df is None or len(df) == 0:
        return df

    df = df.copy()

    # Attempt to coerce season to numeric
    season_num = pd.to_numeric(df.get("season"), errors="coerce")
    df["_season_num"] = season_num

    # 1) Keep explicit S1–S7 rows
    mask_pre_s8 = (df["_season_num"].notna()) & (df["_season_num"] <= 7)
    df_pre = df[mask_pre_s8]

    # 2) Handle rows with unknown season, filter by text keywords
    df_unknown = df[df["_season_num"].isna()]

    if len(df_unknown):
        keywords = [
            "season 8",
            "eighth and final season",
            "the iron throne",         # S8 finale title
            "s8e",                     # any S8E?
            "series finale",
            "final season of the fantasy drama television series",
        ]

        def _looks_like_s8(text: str) -> bool:
            t = text.lower()
            return any(k in t for k in keywords)

        keep_rows = []
        for _, row in df_unknown.iterrows():
            text = str(row.get("text", ""))
            if not _looks_like_s8(text):
                keep_rows.append(row)

        df_unknown_filtered = pd.DataFrame(keep_rows) if keep_rows else pd.DataFrame()
    else:
        df_unknown_filtered = pd.DataFrame()

    filtered = pd.concat([df_pre, df_unknown_filtered], ignore_index=True)
    filtered = filtered.drop(columns=["_season_num"], errors="ignore")

    return filtered


def _run_analysis_agents(state: AgentState, evidence_df: Optional[pd.DataFrame]) -> AgentState:
    """
    Run Narrative, Causal, and Emotion agents using reranked evidence.
    Writes outputs into state.narrative / state.causal / state.emotion
    and mirrors them under state.logs[...] for the Gradio UI.
    """
    if evidence_df is None or len(evidence_df) == 0:
        return state

    q = state.question
    evidence_lines = _make_evidence_lines(evidence_df, max_lines=12)

    # --- Narrative Agent ---
    try:
        n = narrative_agent(q, evidence_lines)
        narrative_dict = n.__dict__ if hasattr(n, "__dict__") else dict(n)
    except Exception as e:
        narrative_dict = {"error": str(e)}

    state.narrative = narrative_dict
    state.logs["narrative"] = narrative_dict
    if isinstance(narrative_dict, dict):
        # Use narrative summary as a high-level evidence text if present
        state.evidence_text = narrative_dict.get("narrative_summary", "") or state.evidence_text

    # --- Causality Agent ---
    try:
        c = causal_agent(q, evidence_lines)
        causal_dict = c.__dict__ if hasattr(c, "__dict__") else dict(c)
    except Exception as e:
        causal_dict = {"error": str(e)}

    state.causal = causal_dict
    state.logs["causal"] = causal_dict

    # --- Emotion Agent ---
    try:
        e = emotion_agent(q, evidence_lines)
        emotion_dict = e.__dict__ if hasattr(e, "__dict__") else dict(e)
    except Exception as e:
        emotion_dict = {"error": str(e)}

    state.emotion = emotion_dict
    state.logs["emotion"] = emotion_dict

    return state


# ---------------------------------------------------------------
#                    ROUTER
# ---------------------------------------------------------------

def router_node(state: AgentState) -> AgentState:
    """Classify the question into flows."""

    q = state.question.lower()

    # --- Creative / Alternate Ending must be FIRST ---
    if any(w in q for w in [
        "alternate ending",
        "rewrite",
        "new ending",
        "fix season 8",
        "rewrite season 8",
        "reimagine season 8",
        "alternate finale",
    ]):
        decision = "alternate_ending_flow"

    # --- Narrative Why/Reason ---
    elif any(w in q for w in ["why", "cause", "reason", "because"]):
        decision = "narrative_flow"

    # --- Temporal When/Season ---
    elif any(w in q for w in ["when", "timeline", "season", "episode", "chronology"]):
        decision = "temporal_flow"

    # --- Factual Who/What/Where ---
    elif any(w in q for w in ["who", "what", "where", "mother", "father", "killed", "identity"]):
        decision = "factual_flow"

    # --- Default ---
    else:
        decision = "basic_rag_flow"

    state.route_decision = decision
    return state


def router_route(state: AgentState) -> str:
    """Return the route string for conditional edges."""
    return state.route_decision


# ---------------------------------------------------------------
#                    FACTUAL FLOW
# ---------------------------------------------------------------

def factual_flow(state: AgentState) -> AgentState:
    q = state.question

    parsed = query_decomposer_agent(q)
    state.logs["decomposer"] = parsed.__dict__

    queries = parsed.retrieval_queries or [q]

    hits = _retrieve_with_hybrid(queries, topk=15)
    state.retrieved = hits
    state.logs["retrieval"] = {
        "flow": "factual_flow",
        "queries": queries,
        "hit_count": int(len(hits)),
    }

    state = node_reranker(state, reranker_model=get_reranker())

    # Analysis agents (Narrative / Causal / Emotion)
    state = _run_analysis_agents(
        state,
        state.reranked if state.reranked is not None else state.retrieved,
    )

    state = node_synthesizer(state, answer_prompt_template=ANSWER_PROMPT)

    return state


# ---------------------------------------------------------------
#                    TEMPORAL FLOW
# ---------------------------------------------------------------

def temporal_flow(state: AgentState) -> AgentState:
    q = state.question

    parsed = query_decomposer_agent(q)
    temporal = temporal_agent(q)

    state.logs["decomposer"] = parsed.__dict__
    state.logs["temporal"] = temporal.__dict__

    queries = set(parsed.retrieval_queries or [q])

    for t in temporal.timeline_queries:
        queries.add(t)
    for ep in temporal.episodes:
        queries.add(f"{q} (seen in {ep})")

    queries = list(queries)

    hits = _retrieve_with_hybrid(queries, topk=15)
    state.retrieved = hits
    state.logs["retrieval"] = {
        "flow": "temporal_flow",
        "queries": queries,
        "hit_count": int(len(hits)),
    }

    state = node_reranker(state, reranker_model=get_reranker())

    # Analysis agents
    state = _run_analysis_agents(
        state,
        state.reranked if state.reranked is not None else state.retrieved,
    )

    state = node_synthesizer(state, answer_prompt_template=ANSWER_PROMPT)

    return state


# ---------------------------------------------------------------
#                    NARRATIVE FLOW
# ---------------------------------------------------------------

def narrative_flow(state: AgentState) -> AgentState:
    q = state.question

    parsed = query_decomposer_agent(q)
    state.logs["decomposer"] = parsed.__dict__

    queries = parsed.retrieval_queries or [q]

    hits = _retrieve_with_hybrid(queries, topk=15)
    state.retrieved = hits
    state.logs["retrieval"] = {
        "flow": "narrative_flow",
        "queries": queries,
        "hit_count": int(len(hits)),
    }

    state = node_reranker(state, reranker_model=get_reranker())

    # Analysis agents
    state = _run_analysis_agents(
        state,
        state.reranked if state.reranked is not None else state.retrieved,
    )

    # Correct synthesize call
    state = node_synthesizer(state, answer_prompt_template=ANSWER_PROMPT)

    return state


# ---------------------------------------------------------------
#                    BASIC FLOW
# ---------------------------------------------------------------

def basic_rag_flow(state: AgentState) -> AgentState:
    q = state.question

    # Option A: use original basic_rag_agent
    # df = basic_rag_agent(q)

    # Option B: basic flow uses hybrid retrieval for consistency
    df = _retrieve_with_hybrid(q, topk=15)

    state.retrieved = df
    state.logs["retrieval"] = {
        "flow": "basic_rag_flow",
        "queries": [q],
        "hit_count": int(len(df)),
    }

    state = node_reranker(state, reranker_model=get_reranker())

    # Analysis agents
    state = _run_analysis_agents(
        state,
        state.reranked if state.reranked is not None else state.retrieved,
    )

    state = node_synthesizer(state, answer_prompt_template=ANSWER_PROMPT)

    return state


# ---------------------------------------------------------------
#                    ALTERNATE ENDING FLOW
# ---------------------------------------------------------------

def alternate_ending_flow(state: AgentState) -> AgentState:
    """
    Creative flow: generate an alternate Season 8 ending
    using ONLY Seasons 1–7 evidence.
    """
    q = state.question

    parsed = query_decomposer_agent(q)
    state.logs["decomposer"] = parsed.__dict__

    queries = parsed.retrieval_queries or [q]

    # Retrieve a fairly large pool
    raw_hits = _retrieve_with_hybrid(queries, topk=40)
    raw_count = int(len(raw_hits))

    # Hard filter to S1–S7 only
    filtered_hits = _filter_to_pre_s8(raw_hits)
    filtered_count = int(len(filtered_hits))

    state.retrieved = filtered_hits
    state.logs["retrieval"] = {
        "flow": "alternate_ending_flow",
        "queries": queries,
        "raw_hit_count": raw_count,
        "filtered_pre_s8_count": filtered_count,
    }

    # Hand off to alternate ending agent (it can also do its own internal filtering)
    alt = alternate_ending_agent(q, filtered_hits)

    state.answer = alt.scene
    state.logs["alternate_ending"] = {
        "scene": alt.scene,
        "variants": alt.variants,
    }

    return state

def nss_flow(state: AgentState) -> AgentState:
    return scoring_agent(state)


# ---------------------------------------------------------------
#                    BUILD GRAPH
# ---------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("factual_flow", factual_flow)
workflow.add_node("temporal_flow", temporal_flow)
workflow.add_node("narrative_flow", narrative_flow)
workflow.add_node("basic_rag_flow", basic_rag_flow)
workflow.add_node("alternate_ending_flow", alternate_ending_flow)
workflow.add_node("nss_scoring", scoring_agent)  # <-- NEW


workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    router_route,
    {
        "factual_flow": "factual_flow",
        "temporal_flow": "temporal_flow",
        "narrative_flow": "narrative_flow",
        "basic_rag_flow": "basic_rag_flow",
        "alternate_ending_flow": "alternate_ending_flow",
    },
)

workflow.add_edge("factual_flow", "nss_scoring")
workflow.add_edge("temporal_flow", "nss_scoring")
workflow.add_edge("narrative_flow", "nss_scoring")
workflow.add_edge("basic_rag_flow", "nss_scoring")
workflow.add_edge("alternate_ending_flow", "nss_scoring")

workflow.add_edge("nss_scoring", END) 

app = workflow.compile()

print("Cosine of Thrones multi-agent LangGraph orchestrator ready.")