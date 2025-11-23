# ragthrones/app/gradio_ui.py
import json
import gradio as gr
import pandas as pd

from ragthrones.pipelines.multi_agent_graph import app, AgentState
from ragthrones.retrieval.load_vectorstore import load_all_vectorstore

VS = load_all_vectorstore()


# ------------------------------------------------------------
# SAFE COLLAPSIBLE EVIDENCE HTML (CSS-only, Gradio-safe)
# ------------------------------------------------------------
def build_evidence_html(df, k=10):
    """Render collapsible evidence cards using CSS-only accordions (Gradio-safe)."""
    if df is None or len(df) == 0:
        return "<p>No evidence retrieved.</p>"

    html = """
    <style>
        .accordion-item {
            border: 1px solid #ccc !important;
            border-radius: 6px !important;
            margin-bottom: 10px !important;
            background-color: #f7f7f7 !important;
        }

        .accordion-item input {
            display: none !important;
        }

        .accordion-header {
            padding: 10px !important;
            font-weight: bold !important;
            cursor: pointer !important;
            color: #333 !important;
            background-color: #e6e6e6 !important;
            display: block !important;
        }

        .accordion-body {
            padding: 10px !important;
            display: none !important;
            color: #222 !important;
            background-color: #fff !important;
            border-top: 1px solid #ccc !important;
        }

        .accordion-item input:checked ~ .accordion-body {
            display: block !important;
        }
    </style>
    """

    html += "<div class='accordion'>"

    for idx, (_, row) in enumerate(df.head(k).iterrows()):
        season = row.get("season", "?")
        episode = row.get("episode", "?")
        tag = f"S{season}E{episode}" if str(season).isdigit() else "S?E?"
        kind = row.get("chunk_kind", "chunk")
        text = str(row.get("text", "")).replace("\n", " ").strip()

        html += f"""
        <div class="accordion-item">
            <input type="checkbox" id="acc-{idx}">
            <label class="accordion-header" for="acc-{idx}">
                {tag} • {kind}
            </label>
            <div class="accordion-body">{text}</div>
        </div>
        """

    html += "</div>"
    return html


# ------------------------------------------------------------
# Pipeline runner
# ------------------------------------------------------------
def run_cosine(question: str):
    if not question.strip():
        return (
            "Please enter a question.",
            "<p>No analysis.</p>",
            "<p>No evidence.</p>",
            "",
            ""
        )

    init_state = AgentState(question=question)
    raw = app.invoke(init_state)
    final_state = AgentState(**raw)

    # -------------------------------
    # Final Answer
    # -------------------------------
    answer = final_state.answer or "(no answer)"

    # -------------------------------
    # Analysis Cards (using new state fields)
    # -------------------------------
    causal_info = final_state.causal.get("causal_links", [])
    emotion_info = final_state.emotion.get("character_entities", [])
    narrative_summary = final_state.narrative.get("narrative_summary", "")

    causal_html = f"""
    <div style='padding:16px; border:1px solid #ddd; border-radius:8px; background:#fff8e8;'>
        <h3 style='margin:0;'>🧠 Causality Agent Report</h3>
        <p style='margin-top:8px; font-size:15px;'>{'<br>'.join(causal_info) if causal_info else 'No causal links found.'}</p>
    </div>
    """

    emotion_html = f"""
    <div style='padding:16px; border:1px solid #ddeaff; border-radius:8px; background:#f5faff;'>
        <h3 style='margin:0;'>🔥 Emotion Agent Report</h3>
        <p style='margin-top:8px; font-size:15px;'>{'<br>'.join(emotion_info) if emotion_info else 'No emotional indicators found.'}</p>
    </div>
    """

    analysis_cards = causal_html + "<br>" + emotion_html

    # -------------------------------
    # Evidence Accordion
    # -------------------------------
    evidence_html = build_evidence_html(final_state.reranked)

    # -------------------------------
    # Debug Logs
    # -------------------------------
    try:
        logs = json.dumps(final_state.logs, indent=2)
    except Exception:
        logs = str(final_state.logs)

    return (
        answer,
        analysis_cards,
        evidence_html,
        narrative_summary,
        logs,
    )


# ------------------------------------------------------------
# Build UI
# ------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="Cosine of Thrones") as demo:

        gr.Markdown("""
        # 🧙‍♂️ Cosine of Thrones — Multi-Agent RAG System  
        Ask a Game of Thrones question and watch the agents work.
        """)

        question = gr.Textbox(
            label="Ask a Question",
            placeholder="Why did the Red Wedding happen?",
            lines=1
        )

        run_btn = gr.Button("Run", variant="primary")

        answer_out = gr.Textbox(label="Final Answer", lines=4)
        analysis_out = gr.HTML(label="Analysis")
        evidence_out = gr.HTML(label="Top Evidence (Reranked)")
        narrative_out = gr.Textbox(label="Narrative Summary", lines=6)
        logs_out = gr.Textbox(label="Debug Logs", lines=15)

        run_btn.click(
            fn=run_cosine,
            inputs=[question],
            outputs=[
                answer_out,
                analysis_out,
                evidence_out,
                narrative_out,
                logs_out
            ]
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()