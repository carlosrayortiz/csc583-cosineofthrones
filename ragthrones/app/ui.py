# ragthrones/app/ui.py
import gradio as gr
from ragthrones.retrieval.hybrid_search import hybrid_search_aug
from ragthrones.agents.synth import synth_answer

def go(q):
    reranked = hybrid_search_aug(q)
    return synth_answer(q, reranked)

def build_interface():
    return gr.Interface(
        fn=go,
        inputs="text",
        outputs="text",
        title="Cosinify",
        allow_flagging="never"
    )