"""
Build simple HTML evidence blocks for displaying retrieved passages.
Used by the cosine pipeline and UI layers.
"""

import pandas as pd


def build_evidence_html(df: pd.DataFrame) -> str:
    """
    Given a DataFrame of retrieved hits (df_aug rows),
    build simple collapsible HTML to display evidence.
    """

    if df is None or df.empty:
        return "<p>No evidence found.</p>"

    html = "<div style='font-family: sans-serif;'>"

    for _, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        score = float(row.get("score", 0.0))

        html += f"""
        <div style="
            margin-bottom: 12px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 6px;
            cursor: pointer;
        " onclick="this.classList.toggle('expanded')">
            <div style="font-weight: bold;">
                Score: {score:.3f}
            </div>
            <div class="evidence-body" style="margin-top: 6px;">
                {text}
            </div>
        </div>
        """

    html += "</div>"
    return html