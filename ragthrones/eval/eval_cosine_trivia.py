# ragthrones/scripts/eval_trivia.py
# =============================================
# COSINE of Thrones — Trivia Evaluation Script
# =============================================
#
# Evaluates the Cosine of Thrones multi-agent LangGraph pipeline
# against the FunTrivia golden Q&A set using:
#   - Exact Match (EM)
#   - F1 (token-level)
#   - Semantic similarity (TF-IDF cosine)
#
# It forces TRIVIA answer style (short, canonical answers)
# by setting state.trivia_mode = True on every question.
#
# Requirements:
#   pip install pandas tqdm scikit-learn
#
# Run:
#   python -m ragthrones.scripts.eval_trivia
#
# Output:
#   eval/funtrivia_cosine_eval.csv
# =============================================

import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ragthrones.pipelines.multi_agent_graph import app, AgentState


# -------------------------------------------
# Helper: run_graph (evaluation-friendly)
# -------------------------------------------
def run_graph(question: str, trivia_mode: bool = True) -> AgentState:
    """
    Executes the full Cosine of Thrones LangGraph pipeline and returns the final AgentState.

    Args:
        question (str): The input question.
        trivia_mode (bool):
            - True  -> force TRIVIA_ANSWER_PROMPT (short factual answer)
            - False -> force normal ANSWER_PROMPT

    Returns:
        AgentState: Final state containing answer, evidence, logs, etc.
    """
    state = AgentState(question=question)

    # Manual override for Trivia style answers
    state.trivia_mode = bool(trivia_mode)

    raw = app.invoke(state)
    final_state = AgentState(**raw)
    return final_state


# -------------------------------------------
# Load FunTrivia Golden Set
# -------------------------------------------
GOLD_PATH = Path("ragthrones/eval/funtrivia_golden_set.csv")

if not GOLD_PATH.exists():
    raise FileNotFoundError(
        f"Golden set not found at {GOLD_PATH}. "
        "Make sure ragthrones/eval/funtrivia_golden_set.csv exists."
    )

dfg = pd.read_csv(GOLD_PATH)
print(f"Loaded {len(dfg)} golden trivia questions from {GOLD_PATH}")


# -------------------------------------------
# Text Normalization & Metrics
# -------------------------------------------
def normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def em(pred: str, gold: str) -> float:
    """Exact Match (after normalization)."""
    return float(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    """Token-level F1 score between prediction and gold answer."""
    p_tokens = normalize(pred).split()
    g_tokens = normalize(gold).split()
    if not p_tokens or not g_tokens:
        return 0.0

    pc, gc = Counter(p_tokens), Counter(g_tokens)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def semantic_sim(pred: str, gold: str) -> float:
    """TF-IDF cosine similarity between pred and gold."""
    pred_n = normalize(pred)
    gold_n = normalize(gold)
    vec = TfidfVectorizer().fit_transform([pred_n, gold_n])
    return float(cosine_similarity(vec)[0, 1])


# -------------------------------------------
# Evaluation Loop
# -------------------------------------------
results = []
error_count = 0

for i, row in tqdm(dfg.iterrows(), total=len(dfg), desc="Evaluating Cosine of Thrones on Trivia"):
    q = str(row["question"])
    gold = str(row["answer_short"])

    try:
        final = run_graph(q, trivia_mode=True)
        pred = final.answer or ""

        # Metrics
        score_em = em(pred, gold)
        score_f1 = f1(pred, gold)
        score_sem = semantic_sim(pred, gold)

        # Evidence snippet (if available)
        evid = getattr(final, "evidence_text", "") or ""
        evid_snippet = evid[:300]

        # NSS (if present)
        nss_score = getattr(final, "nss_score", None) or {}
        nss_total = nss_score.get("total_weighted_score", None)

        results.append({
            "qnum": row.get("qnum", i + 1),
            "question": q,
            "gold": gold,
            "pred": pred,
            "em": score_em,
            "f1": score_f1,
            "semantic": score_sem,
            "nss_total_weighted": nss_total,
            "evidence": evid_snippet,
        })

    except Exception as e:
        error_count += 1
        print(f"[{i+1}] ERROR on question: {q[:80]}... -> {e}")
        results.append({
            "qnum": row.get("qnum", i + 1),
            "question": q,
            "gold": gold,
            "pred": f"ERROR: {e}",
            "em": 0.0,
            "f1": 0.0,
            "semantic": 0.0,
            "nss_total_weighted": None,
            "evidence": "",
        })

    # Gentle rate-limit for OpenAI
    time.sleep(0.25)


# -------------------------------------------
# Aggregate Metrics & Save CSV
# -------------------------------------------
df_eval = pd.DataFrame(results)

avg_em = df_eval["em"].mean()
avg_f1 = df_eval["f1"].mean()
avg_sem = df_eval["semantic"].mean()

print("\n📈 Evaluation Summary (Trivia Mode)")
print(f"Exact Match (EM):       {avg_em:.3f}")
print(f"F1 Score:               {avg_f1:.3f}")
print(f"Semantic Similarity:    {avg_sem:.3f}")
print(f"Total questions:        {len(df_eval)}")
print(f"Errors:                 {error_count}")

OUT_DIR = Path("eval")
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "funtrivia_cosine_eval.csv"

df_eval.to_csv(OUT_PATH, index=False)
print(f"\nSaved detailed trivia evaluation results → {OUT_PATH}")