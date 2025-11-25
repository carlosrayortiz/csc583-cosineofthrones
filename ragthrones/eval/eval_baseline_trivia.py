# ragthrones/scripts/eval_baseline.py
# ==================================================
# COSINE of Thrones — Baseline Evaluation (No RAG)
# ==================================================
#
# Baselines:
#   A) Most Frequent Answer (global unigram baseline)
#   B) Direct LLM Answering (no retrieval)
#   C) Optional: BM25 Retrieval-Only (no reranking, no synthesis)
#
# Evaluated on funtrivia_golden_set.csv
#
# ==================================================

import re
import time
from pathlib import Path
from collections import Counter
import pandas as pd
from tqdm import tqdm

# --- OPTIONAL: Direct LLM baseline ---
from ragthrones.llm.llm_client import llm_client, GEN_MODEL


# ==================================================
# Helper functions
# ==================================================

def normalize(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def em(pred, gold):
    return float(normalize(pred) == normalize(gold))

def f1(pred, gold):
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g:
        return 0.0
    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def semantic_sim(pred, gold):
    v = TfidfVectorizer().fit_transform(
        [normalize(pred), normalize(gold)]
    )
    return float(cosine_similarity(v)[0, 1])


# ==================================================
# Load Golden Set
# ==================================================

GOLD_PATH = Path("ragthrones/eval/funtrivia_golden_set.csv")
if not GOLD_PATH.exists():
    raise FileNotFoundError(f"Missing evaluation file: {GOLD_PATH}")

dfg = pd.read_csv(GOLD_PATH)
print(f"Loaded {len(dfg)} trivia questions.")


# ==================================================
# Baseline A — Most Frequent Answer (Unigram Baseline)
# ==================================================

gold_answers = [normalize(a) for a in dfg["answer_short"].tolist()]
freq_dist = Counter(gold_answers)
most_common_answer = freq_dist.most_common(1)[0][0]

print(f"\n📌 Most frequent answer baseline → '{most_common_answer}'")


# ==================================================
# Baseline B — Direct LLM Answering (no retrieval)
# ==================================================

def llm_direct_answer(q: str) -> str:
    prompt = f"""
You are an expert Game of Thrones trivia answer bot.
Return ONLY the short factual answer.
No explanation.
Question: {q}
"""

    print(f"\nAsking LLM directly:\n{prompt.strip()}\n")

    # ---- ADD THESE DEBUG PRINTS ----
    print("Calling llm_client with model:", GEN_MODEL)

    try:
        response = llm_client.chat.completions.create(
            model=GEN_MODEL,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        print(f"🔥 LLM call FAILED: {e}")
        return f"ERROR: {e}"

    # Print the raw object
    print("🟦 RAW RESPONSE OBJECT:", response)

    try:
        text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"🔥 Could not parse LLM response: {e}")
        return f"ERROR: {e}"

    print(f"LLM response: {text}\n")
    return text


# ==================================================
# Evaluation Loop
# ==================================================

rows = []
for i, row in tqdm(dfg.iterrows(), total=len(dfg), desc="Running baselines"):
    q = row["question"]
    gold = row["answer_short"]

    # Baseline A prediction
    pred_freq = most_common_answer

    # Baseline B prediction
    try:
        pred_llm = llm_direct_answer(q)
    except Exception as e:
        pred_llm = f"ERROR: {e}"

    # Record
    rows.append({
        "qnum": row.get("qnum", i+1),
        "question": q,
        "gold": gold,

        "pred_freq": pred_freq,
        "em_freq": em(pred_freq, gold),
        "f1_freq": f1(pred_freq, gold),
        "semantic_freq": semantic_sim(pred_freq, gold),

        "pred_llm": pred_llm,
        "em_llm": em(pred_llm, gold),
        "f1_llm": f1(pred_llm, gold),
        "semantic_llm": semantic_sim(pred_llm, gold),
    })

    # avoid hammering the LLM endpoint
    time.sleep(0.25)


# ==================================================
# Save + Summary
# ==================================================

df = pd.DataFrame(rows)

print("\n📊 Baseline Results Summary")
print("Most Frequent Answer:")
print(f"  EM:       {df['em_freq'].mean():.3f}")
print(f"  F1:       {df['f1_freq'].mean():.3f}")
print(f"  Semantic: {df['semantic_freq'].mean():.3f}")

print("\nLLM Direct Answering:")
print(f"  EM:       {df['em_llm'].mean():.3f}")
print(f"  F1:       {df['f1_llm'].mean():.3f}")
print(f"  Semantic: {df['semantic_llm'].mean():.3f}")

OUT = Path("ragthrones/eval/funtrivia_baselines.csv")
df.to_csv(OUT, index=False)

print(f"\nSaved baseline results → {OUT}")