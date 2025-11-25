from fastapi import APIRouter
from ragthrones.retrieval.hybrid_search import hybrid_search_aug
# from ragthrones.agents.synth import synth_answer

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/answer")
def answer(q: str):
    reranked = hybrid_search_aug(q)
    # answer = synth_answer(q, reranked)
    return {
        "query": q,
        "answer": "Synthesis not implemented yet",
        "top_chunks": reranked[:3]
    }