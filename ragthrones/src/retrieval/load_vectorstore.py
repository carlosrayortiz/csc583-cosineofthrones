"""
Load FAISS, BM25, and supporting artifacts for RAGThrones.
"""

import os
import pickle
import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from ragthrones.embeddings.embed_client import EmbedClient


ARTIFACT_DIR = "data/artifacts"


def load_df_aug(path=f"{ARTIFACT_DIR}/df_aug.pkl"):
    """Load df_aug dataframe."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"df_aug not found at {path}")
    return pd.read_pickle(path)


def load_faiss_index(path=f"{ARTIFACT_DIR}/faiss.index"):
    """Load FAISS index."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index missing at {path}")

    index = faiss.read_index(path)
    return index


def load_bm25(path=f"{ARTIFACT_DIR}/bm25.pkl"):
    """Load pickled BM25 object."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"BM25 object missing at {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_vectorstore():
    """
    Load FAISS, BM25, df_aug, and embedding client.
    Returns a dictionary for convenience.
    """

    df_aug = load_df_aug()
    index = load_faiss_index()
    bm25 = load_bm25()

    embed_client = EmbedClient()

    return {
        "df_aug": df_aug,
        "faiss": index,
        "bm25": bm25,
        "embed_client": embed_client,
    }