import pandas as pd
import numpy as np
import re
from nltk import word_tokenize
from pathlib import Path

BASE = Path(__file__).parent

def normalize(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())

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

def hybrid_correct(em, f1_score, semantic, f1_thr=0.90, sem_thr=0.80):
    if em == 1:
        return 1
    if f1_score >= f1_thr:
        return 1
    if semantic >= sem_thr:
        return 1
    return 0

def evaluate(csv_path, output_path):
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    df["EM_new"] = df.apply(lambda r: exact_match(r["gold"], r["pred"]), axis=1)
    df["F1_new"] = df.apply(lambda r: f1(r["gold"], r["pred"]), axis=1)

    df["Correct"] = df.apply(
        lambda r: hybrid_correct(
            r["EM_new"],
            r["F1_new"],
            r.get("semantic", 0.0)
        ),
        axis=1,
    )

    summary = {
        "EM": df["EM_new"].mean(),
        "F1": df["F1_new"].mean(),
        "Semantic (avg)": df["semantic"].mean() if "semantic" in df.columns else None,
        "Accuracy": df["Correct"].mean(),
        "Total": len(df),
    }

    df.to_csv(output_path, index=False)
    print(f"Saved updated CSV → {output_path}")

    print("\n=== Summary ===")
    for k, v in summary.items():
        if v is None:
            print(f"{k}: None")
        else:
            print(f"{k}: {v:.4f}")

    return df, summary


if __name__ == "__main__":
    CSV_PATH = BASE / "funtrivia_cosine_eval.csv"
    OUT_PATH = BASE / "funtrivia_cosine_eval_with_metrics.csv"
    evaluate(CSV_PATH, OUT_PATH)