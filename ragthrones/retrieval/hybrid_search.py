"""
Hybrid Search for RAGThrones
---------------------------
Fixes: ensures hybrid_search_aug uses the same global vectorstore
instance as the RetrievalAgent, avoiding stale globals.
"""

import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi

from ragthrones.retrieval.load_vectorstore import load_all_vectorstore

# ------------------------------------------------------------
# GLOBAL SINGLETON (REAL FIX)
# ------------------------------------------------------------
# ensure every agent call uses the SAME vectorstore instance
_VSTORE = None

def _get_store():
    global _VSTORE
    if _VSTORE is None:
        _VSTORE = load_all_vectorstore()
    return _VSTORE


# ------------------------------------------------------------
# Main Hybrid Retrieval Function
# ------------------------------------------------------------
def hybrid_search_aug(
    query: str,
    topk: int = 10,
    alpha: float = 0.35,
    cand_mult: int = 20,
):
    """
    Runtime loads the ACTIVE vectorstore (FAISS + BM25 + df_aug).
    Fixes stale-global bug that caused zero-hit retrieval inside agents.
    """

    # Load store FIRST
    store = _get_store()

    df_aug = store["df_aug"]
    faiss_index = store["faiss"]
    bm25_obj = store["bm25"]
    embed_client = store["embed_client"]

    # BM25 wrapper
    if isinstance(bm25_obj, list):
        bm25 = BM25Okapi(bm25_obj)
    else:
        bm25 = bm25_obj

    # --- SAFE DEBUG PRINTS ---
    print("\nDEBUG hybrid_search_aug:")
    print("df_aug rows:", len(df_aug))
    print("faiss_index.ntotal:", faiss_index.ntotal)
    print("bm25 token count:", len(bm25_obj))
    print("embed_client:", embed_client)
    print("query:", query)
    print("---------------------------\n")

    df_aug = store["df_aug"]
    faiss_index = store["faiss"]
    bm25_obj = store["bm25"]
    embed_client = store["embed_client"]

    # BM25 is a list of tokens → wrap with BM25Okapi
    if isinstance(bm25_obj, list):
        bm25 = BM25Okapi(bm25_obj)
    else:
        bm25 = bm25_obj

    # ------------------------------
    # 1. Embed query
    # ------------------------------
    try:
        q_emb = embed_client.embed(query)
    except TypeError:
        q_emb = embed_client.embed(query, model="text-embedding-3-large")

    qv = np.asarray(q_emb, dtype="float32")[None, :]
    faiss.normalize_L2(qv)

    # ------------------------------
    # 2. FAISS vector search
    # ------------------------------
    D, I = faiss_index.search(qv, topk * cand_mult)
    vec_scores = D[0].tolist()
    vec_idx = I[0].tolist()

    # ------------------------------
    # 3. BM25 lexical search
    # ------------------------------
    q_tokens = query.lower().split()
    bm_scores = bm25.get_scores(q_tokens)

    bm_top = np.argsort(bm_scores)[::-1][: topk * cand_mult]

    # ------------------------------
    # 4. Merge
    # ------------------------------
    max_valid = len(df_aug)
    valid_vec_pairs = [
        (int(idx), float(score))
        for idx, score in zip(vec_idx, vec_scores)
        if 0 <= int(idx) < max_valid
    ]

    v_map = {i: s for i, s in valid_vec_pairs}
    vec_idx_valid = set(v_map.keys())
    bm_top_valid = set(int(i) for i in bm_top if 0 <= int(i) < max_valid)

    cand = list(vec_idx_valid | bm_top_valid)
    if not cand:
        return pd.DataFrame([])

    # ------------------------------
    # 5. Score blending
    # ------------------------------
    bm_max = max(bm_scores) if len(bm_scores) else 1.0

    scored = []
    for i in cand:
        v = float(v_map.get(i, 0.0))
        b = float(bm_scores[i]) / (bm_max + 1e-6)
        final = alpha * v + (1 - alpha) * b
        scored.append((i, final))

    scored.sort(key=lambda x: x[1], reverse=True)

    # ------------------------------
    # 6. Build DF
    # ------------------------------
    rows = []
    for i, sc in scored[:topk]:
        row = df_aug.iloc[int(i)].to_dict()
        row["score"] = float(sc)
        rows.append(row)

    return pd.DataFrame(rows)