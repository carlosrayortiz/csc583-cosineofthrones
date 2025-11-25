import pandas as pd
import numpy as np
import re
from nltk import word_tokenize
from pathlib import Path

BASE = Path(__file__).parent

COSINE_CSV = BASE / "funtrivia_cosine_eval.csv"
BASELINE_CSV = BASE / "funtrivia_baselines.csv"
OUTPUT_COMPARISON_CSV = BASE / "evaluation_comparison.csv"

# ==========================================================
# CONFIG — Strict Thresholds
# ==========================================================
F1_THRESHOLD = 0.75
SEM_THRESHOLD = 0.70

# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize(t):
    if not isinstance(t, str):
        return ""
    return re.sub(r"\s+", " ", t.strip().lower())


# ==========================================================
# METRICS
# ==========================================================

def exact_match(gold, pred):
    return 1 if normalize(gold) == normalize(pred) else 0


def f1(gold, pred):
    gold_tokens = word_tokenize(normalize(gold))
    pred_tokens = word_tokenize(normalize(pred))

    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0

    common = set(gold_tokens) & set(pred_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def hybrid_correct(em, f1_score, semantic, f1_thr=F1_THRESHOLD, sem_thr=SEM_THRESHOLD):
    if em == 1:
        return 1
    if f1_score >= f1_thr:
        return 1
    if semantic >= sem_thr:
        return 1
    return 0


# ==========================================================
# BAR-MODE CORRECTNESS
# ==========================================================

def bar_mode_correct(gold, pred, semantic, f1_score):
    gold_norm = normalize(gold)
    pred_norm = normalize(pred)

    # 1. Exact match
    if gold_norm == pred_norm:
        return 1

    # 2. Substring containment
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return 1

    # 3. Semantic similarity bar-level threshold
    if semantic >= 0.55:
        return 1

    # 4. F1 lexical threshold
    if f1_score >= 0.60:
        return 1

    # Otherwise incorrect
    return 0


# ==========================================================
# EVALUATE ANY SYSTEM COLUMNS INSIDE ONE CSV
# ==========================================================

def evaluate_system(df, pred_col, gold_col="gold", sem_col=None):
    ema = []
    f1a = []
    strict_corr = []
    bar_corr = []

    for _, r in df.iterrows():
        gold = r[gold_col]
        pred = r[pred_col]

        emv = exact_match(gold, pred)
        f1v = f1(gold, pred)
        sem = r[sem_col] if sem_col and sem_col in df.columns else 0.0

        ema.append(emv)
        f1a.append(f1v)

        strict_corr.append(hybrid_correct(emv, f1v, sem))
        bar_corr.append(bar_mode_correct(gold, pred, sem, f1v))

    return {
        "EM": np.mean(ema),
        "F1": np.mean(f1a),
        "Semantic": float(df[sem_col].mean()) if sem_col and sem_col in df.columns else None,
        "Correct_Strict": np.mean(strict_corr),
        "Correct_BarMode": np.mean(bar_corr)
    }


# ==========================================================
# MAIN COMPARE SCRIPT
# ==========================================================

def main():

    print("\n=== Loading Baselines ===")
    baseline = pd.read_csv(BASELINE_CSV)

    print("=== Loading Cosine Results ===")
    cosine = pd.read_csv(COSINE_CSV)

    print(f"\n=== USING STRICT THRESHOLDS ===")
    print(f"F1 Threshold:       {F1_THRESHOLD}")
    print(f"Semantic Threshold: {SEM_THRESHOLD}")

    # ------------------------------------------------------
    # Evaluate each system
    # ------------------------------------------------------
    print("\nEvaluating Most Frequent baseline…")
    freq_scores = evaluate_system(
        baseline,
        pred_col="pred_freq",
        sem_col="semantic_freq"
    )

    print("Evaluating LLM baseline…")
    llm_scores = evaluate_system(
        baseline,
        pred_col="pred_llm",
        sem_col="semantic_llm"
    )

    print("Evaluating Cosine-of-Thrones…")
    cosine_scores = evaluate_system(
        cosine,
        pred_col="pred",
        sem_col="semantic"
    )

    # ------------------------------------------------------
    # CREATE COMPARISON TABLE
    # ------------------------------------------------------
    metrics = ["EM", "F1", "Semantic", "Correct_Strict", "Correct_BarMode"]
    rows = []

    for m in metrics:
        row = {
            "Metric": m,
            "Freq_Baseline": freq_scores[m],
            "LLM_Baseline": llm_scores[m],
            "Cosine": cosine_scores[m],
            "Δ Cosine-LLM": cosine_scores[m] - llm_scores[m] if llm_scores[m] is not None else None,
            "%Δ Cosine-LLM": ((cosine_scores[m] - llm_scores[m]) / llm_scores[m]
                               if llm_scores[m] not in [None, 0] else None),
            "F1_Threshold": F1_THRESHOLD,
            "Semantic_Threshold": SEM_THRESHOLD,
        }
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_COMPARISON_CSV, index=False)

    # ------------------------------------------------------
    # PRINT SUMMARY
    # ------------------------------------------------------
    print("\n=== FINAL COMPARISON SUMMARY ===")
    print(df_out.to_string(index=False))

    print(f"\nSaved comparison CSV → {OUTPUT_COMPARISON_CSV}")


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()