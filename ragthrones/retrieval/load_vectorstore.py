import os
import pickle
import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from ragthrones.embeddings.embed_client import EmbedClient

from google.cloud import storage

# -------------------------------------------
# Local artifact paths
# -------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
ARTIFACT_DIR = os.path.join(PACKAGE_ROOT, "data", "artifacts")

# -------------------------------------------
# Cloud Run fallback path
# -------------------------------------------
GCS_BUCKET = "cosinify-artifacts"
GCS_PREFIX = ""            # files stored at root of bucket
CLOUD_TMP_DIR = "/tmp/artifacts"   # writeable in Cloud Run


def ensure_gcs_artifacts():
    """
    If local artifacts do not exist, fetch them from GCS into /tmp/artifacts.
    Minimal change: keeps your existing load_*() functions untouched.
    """
    if os.path.exists(ARTIFACT_DIR):
        return ARTIFACT_DIR  # Local dev

    os.makedirs(CLOUD_TMP_DIR, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    # expected files
    files = ["df_aug.pkl", "faiss.index", "bm25.pkl"]

    for fname in files:
        blob = bucket.blob(f"{GCS_PREFIX}{fname}")
        local_path = os.path.join(CLOUD_TMP_DIR, fname)

        if not os.path.exists(local_path):
            print(f"[GCS] Downloading {fname} → {local_path}")
            blob.download_to_filename(local_path)

    return CLOUD_TMP_DIR



# ------------------------------------------------------------
# Existing loader functions (unchanged)
# ------------------------------------------------------------
def load_df_aug(path=None):
    if path is None:
        path = os.path.join(ensure_gcs_artifacts(), "df_aug.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"df_aug not found at {path}")
    return pd.read_pickle(path)


def load_faiss_index(path=None):
    if path is None:
        path = os.path.join(ensure_gcs_artifacts(), "faiss.index")
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index missing at {path}")
    return faiss.read_index(path)


def load_bm25(path=None):
    if path is None:
        path = os.path.join(ensure_gcs_artifacts(), "bm25.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"BM25 object missing at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_vectorstore():
    """
    Minimal modified: now works with local OR GCS.
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