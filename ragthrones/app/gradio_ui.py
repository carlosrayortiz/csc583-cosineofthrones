# ragthrones/app/gradio_ui.py
import json
import gradio as gr
import pandas as pd

from ragthrones.pipelines.multi_agent_graph import app, AgentState
from ragthrones.retrieval.load_vectorstore import load_all_vectorstore

VS = load_all_vectorstore()

def build_nss_panel(nss: dict) -> str:
    """Return a styled HTML panel for the Narrative Scoring System results."""
    if not nss or "scores" not in nss:
        return "<p>No NSS score available.</p>"

    scores = nss["scores"]
    total = nss.get("total_weighted_score", 0)

    # Compute normalized final score (0–5)
    max_total = sum(v["weight"] * 5 for v in scores.values())
    final_score = round((total / max_total) * 5, 2)

    html = f"""
    <style>
        .nss-card {{
            border: 1px solid #d0dce7;
            border-radius: 10px;
            background: #f7fbff;
            padding: 20px;
            font-family: 'Inter', sans-serif;
            color: #1a1a1a;
        }}
        .nss-title {{
            font-size: 24px;
            font-weight: 700;
            color: #0a3d62;
            margin-bottom: 6px;
        }}
        .nss-final {{
            font-size: 20px;
            font-weight: 600;
            color: #0a3d62;
            margin-bottom: 20px;
        }}
        .nss-row {{
            margin-bottom: 18px;
            padding-bottom: 12px;
            border-bottom: 1px solid #dbe7f3;
        }}
        .nss-row:last-child {{
            border-bottom: none;
        }}
        .nss-row-title {{
            font-weight: 600;
            color: #0a3d62;
            font-size: 16px;
        }}
        .nss-row-expl {{
            margin-top: 4px;
            font-size: 14px;
            color: #555;
        }}
    </style>

    <div class="nss-card">
        <div class="nss-title">🏛 Narrative Structure Score (NSS)</div>
        <div class="nss-final">Final Score: {final_score} / 5.00</div>
    """

    for name, info in scores.items():
        title = name.replace("_", " ").title()
        score = info.get("score", "?")
        weight = info.get("weight", "?")
        explanation = info.get("explanation", "")

        html += f"""
        <div class="nss-row">
            <div class="nss-row-title">{title} — Score {score} (Weight {weight})</div>
            <div class="nss-row-expl">{explanation}</div>
        </div>
        """

    html += "</div>"
    return html


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
    # Analysis Cards
    # -------------------------------
    causal_info = final_state.causal.get("causal_links", [])
    emotion_info = final_state.emotion.get("character_entities", [])
    narrative_summary = final_state.narrative.get("narrative_summary", "")
    nss_score = final_state.nss_score or {}

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

    # Evidence
    evidence_html = build_evidence_html(final_state.reranked)

    # Debug logs
    try:
        logs = json.dumps(final_state.logs, indent=2)
    except Exception:
        logs = str(final_state.logs)

    # NSS Scoring (proper JSON)
    try:
        nss_json = json.dumps(nss_score, indent=2)
    except Exception:
        nss_json = str(nss_score)

    nss_panel = build_nss_panel(nss_score)

    return (
        answer,
        analysis_cards,
        evidence_html,
        narrative_summary,
        nss_panel,   # <-- updated to use the HTML panel
        logs        # <-- correct
    )

def build_nss_panel(nss: dict) -> str:
    if not nss or "scores" not in nss:
        return "<p>No NSS data.</p>"

    total = nss.get("total_weighted_score", 0)
    max_total = sum(cat["weight"] * 5 for cat in nss["scores"].values())
    final_score = round(total / max_total * 5, 2)

    html = f"""
    <div style="padding:20px; border:1px solid #d0d7de; border-radius:10px; 
                background:#f7f9fc; box-shadow:0 1px 3px rgba(0,0,0,0.1);">
        
        <h2 style="margin-top:0; color:#0f3c76;">
            🏛️ Narrative Structure Score (NSS)
        </h2>
        
        <h3 style="margin-top:4px; color:#0b3262;">
            Final Score: {final_score} / 5.00
        </h3>

        <hr style="margin:16px 0;">
    """

    for cat_name, cat in nss["scores"].items():
        score = cat["score"]
        weight = cat["weight"]
        exp = cat["explanation"]

        html += f"""
        <div style="margin-bottom:18px;">
            <h4 style="margin:0; color:#0b3262;">
                {cat_name.replace('_',' ').title()} — Score {score} (Weight {weight})
            </h4>
            <p style="margin:2px 0 4px; color:#333; font-size:14px;">
                {exp}
            </p>
        </div>
        """

    html += "</div>"
    return html

# ------------------------------------------------------------
# Build UI
# ------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="Cosine of Thrones") as demo:

        gr.Markdown("""
        # 🧙‍♂️ Cosine of Thrones — Multi-Agent RAG System  
        Ask a Game of Thrones question and watch the agents work.
        """)

        TEST_PROMPTS = {
            "Factual (Who/What)": "Who killed Tywin Lannister?",
            "Causality Agent": "Why did Arya decide to leave Winterfell in Season 1?",
            "Emotion Agent": "How did Jon Snow feel after killing Daenerys?",
            "Temporal Agent": "When did Brienne get knighted?",
            "Narrative Agent": "Why did the Red Wedding happen?",
            "Alternate Ending (S1–7 only)": "Rewrite the ending of Season 8 based only on Seasons 1–7.",
            "Hybrid Search Test": "What happened at Hardhome?",
            "Deep Lore (Complex Retrieval)": "What were Littlefinger’s motives for starting the War of the Five Kings?"
        }

        with gr.Row():

            # ================= LEFT COLUMN =================
            with gr.Column(scale=1):

                gr.Markdown("### 🧪 Test Prompts")

                # this placeholder will be filled later
                test_buttons = []

                # store the button objects to link after defining question
                for name, prompt in TEST_PROMPTS.items():
                    test_buttons.append((gr.Button(name), prompt))

                gr.Markdown("### 🧩 Agent Graph")
                gr.Image("ragthrones/assets/langgraph.png", interactive=False)

            # ================= CENTER COLUMN =================
            with gr.Column(scale=2):

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

            # ================= RIGHT COLUMN =================
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Narrative Scoring System (NSS)")
                nss_out = gr.HTML("<p>NSS panel will appear here.</p>")

        # ====================================================
        # RUN BUTTON HANDLER
        # ====================================================
        run_btn.click(
            run_cosine,
            inputs=[question],
            outputs=[
                answer_out,
                analysis_out,
                evidence_out,
                narrative_out,
                nss_out,
                logs_out
            ]
        )

        # ====================================================
        # TEST PROMPTS BUTTON HANDLERS
        # ====================================================
        for btn, prompt in test_buttons:
            btn.click(lambda p=prompt: p, inputs=None, outputs=question)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()