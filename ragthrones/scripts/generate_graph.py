"""
Generate LangGraph Diagram (PNG)
--------------------------------
This script loads the compiled multi-agent LangGraph orchestrator,
renders the graph as a Mermaid PNG, and saves it to:

    ragthrones/assets/langgraph.png

Run:
    python -m ragthrones.scripts.generate_graph
"""

import os
from pathlib import Path

from ragthrones.pipelines.multi_agent_graph import app

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "langgraph.png"


def main():
    print("Generating LangGraph diagram...")

    try:
        graph = app.get_graph()

        # Render PNG bytes (LangGraph built-in)
        png_bytes = graph.draw_mermaid_png()

        # Save file
        with open(OUTPUT_FILE, "wb") as f:
            f.write(png_bytes)

        print(f"✔ LangGraph diagram generated at: {OUTPUT_FILE}")

    except Exception as e:
        print("❌ Error generating diagram:", e)


if __name__ == "__main__":
    main()