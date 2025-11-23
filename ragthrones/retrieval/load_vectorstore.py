import os
import pickle
import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from ragthrones.embeddings.embed_client import EmbedClient

# -------------------------------------------
# FIX: Absolute path to ragthrones/data/artifacts
# -------------------------------------------
# Path to THIS file: .../ragthrones/retrieval/load_vectorstore.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to package root: .../ragthrones
PACKAGE_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Path to artifacts: .../ragthrones/data/artifacts
ARTIFACT_DIR = os.path.join(PACKAGE_ROOT, "data", "artifacts")

def load_df_aug(path=None):
    if path is None:
        path = os.path.join(ARTIFACT_DIR, "df_aug.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"df_aug not found at {path}")
    return pd.read_pickle(path)


def load_faiss_index(path=None):
    if path is None:
        path = os.path.join(ARTIFACT_DIR, "faiss.index")
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index missing at {path}")
    return faiss.read_index(path)


def load_bm25(path=None):
    if path is None:
        path = os.path.join(ARTIFACT_DIR, "bm25.pkl")
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