# scripts/rebuild_artifacts.py

import pandas as pd
import pickle
import faiss
from pathlib import Path

# Path to original artifacts created during preprocessing
ART_DIR = Path("ragthrones/data/artifacts")

def main():
    print("=== Rebuilding Vectorstore Artifacts ===")

    # ----------------------------------------------------
    # 1. Load original files (must exist locally)
    # ----------------------------------------------------
    df_path = ART_DIR / "df_aug.pkl"
    bm25_path = ART_DIR / "bm25.pkl"
    faiss_path = ART_DIR / "faiss.index"

    print(f"Loading {df_path} ...")
    df_aug = pd.read_pickle(df_path)

    print(f"Loading {bm25_path} ...")
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    print(f"Loading {faiss_path} ...")
    index = faiss.read_index(str(faiss_path))

    # ----------------------------------------------------
    # 2. Re-save in Cloud Run–compatible formats
    # ----------------------------------------------------
    print("Saving df_aug.pkl with protocol=5")
    df_aug.to_pickle("df_aug.pkl", protocol=5)

    print("Saving bm25.pkl with protocol=5")
    with open("bm25.pkl", "wb") as f:
        pickle.dump(bm25, f, protocol=5)

    print("Saving faiss.index (FAISS binary)")
    faiss.write_index(index, "faiss.index")

    # ----------------------------------------------------
    # 3. Done
    # ----------------------------------------------------
    print("\n=== Finished! Upload these files to GCS ===")
    print("  • df_aug.pkl")
    print("  • bm25.pkl")
    print("  • faiss.index")
    print("\nAfter uploading, redeploy Cloud Run and your service should boot normally.")


if __name__ == "__main__":
    main()