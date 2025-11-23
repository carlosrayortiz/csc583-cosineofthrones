from ragthrones.retrieval.load_vectorstore import load_all_vectorstore
from ragthrones.retrieval.hybrid_search import hybrid_search_aug
# If you still want to test the old pipeline, make sure cosine_pipeline imports hybrid_search_aug
# from ragthrones.pipelines.cosine_pipeline import run_cosine_pipeline

# Load store
store = load_all_vectorstore()

print("df_aug shape:", store["df_aug"].shape)
print("FAISS index:", store["faiss"].ntotal)
print("BM25 object loaded:", type(store["bm25"]))
print("Embed client:", store["embed_client"])

# -------------------------------
# Test hybrid search directly
# -------------------------------
query = "What happened at the Red Wedding?"
hits = hybrid_search_aug(query, topk=5)

print("\n=== Hybrid Search Results ===")
if hits is None or len(hits) == 0:
    print("(no hits)")
else:
    print(hits[["text", "score"]])

# -------------------------------
# (Optional) Test old cosine pipeline
# -------------------------------
"""
If you still want this and have updated cosine_pipeline to use
hybrid_search_aug instead of hybrid_search, uncomment:

result = run_cosine_pipeline(query, topk=5)
print("\nANSWER:")
print(result["answer"])
print("\nTOP EVIDENCE ROWS:")
print(result["hits"][["text", "score"]].head(3))
"""