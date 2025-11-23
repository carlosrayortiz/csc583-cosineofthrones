#!/usr/bin/env python3
"""
CLI runner for the Cosine of Thrones LangGraph multi-agent system.

Usage:
    python -m ragthrones.scripts.run_agent --question "Why did Robb marry Talisa?"

OR interactive mode:
    python -m ragthrones.scripts.run_agent
"""

import argparse
import sys
import json

from ragthrones.pipelines.multi_agent_graph import app, AgentState
from ragthrones.retrieval.load_vectorstore import load_all_vectorstore

from dotenv import load_dotenv
load_dotenv()


# -----------------------------------------------------------
# Bootstrap: load vectorstore globally ONCE
# -----------------------------------------------------------
VSTORE = None

def ensure_vectorstore_loaded():
    global VSTORE
    if VSTORE is None:
        print("🔧 Loading vectorstore artifacts...")
        VSTORE = load_all_vectorstore()
        print("✅ Vectorstore loaded.")
    return VSTORE


# -----------------------------------------------------------
# Run a single query through the LangGraph app
# -----------------------------------------------------------
def run_query(question: str):
    ensure_vectorstore_loaded()

    print("\n===================================")
    print(f"❓ QUESTION: {question}")
    print("===================================\n")

    init_state = AgentState(question=question)

    # NOTE: LangGraph returns a **dict**, not an AgentState instance.
    final_state = app.invoke(init_state)

    print("\n===================================")
    print("🧠 FINAL ANSWER")
    print("===================================\n")

    print(final_state.get("answer", "(no answer returned)"))

    print("\n===================================")
    print("📚 TOP EVIDENCE")
    print("===================================\n")

    print(final_state.get("evidence_text", "(no evidence)"))

    print("\n===================================")
    print("📝 DEBUG LOGS")
    print("===================================\n")

    logs = final_state.get("logs", {})
    if isinstance(logs, dict):
        try:
            print(json.dumps(logs, indent=2))
        except Exception:
            print(logs)
    else:
        print("(no logs)")

    print("\nDone.")
    return final_state


# -----------------------------------------------------------
# Interactive mode
# -----------------------------------------------------------
def interactive_loop():
    print("\nCosine of Thrones — Interactive CLI Mode")
    print("Type 'exit' or Ctrl-C to quit.\n")

    while True:
        try:
            q = input("🧿 Ask a question: ").strip()
            if q.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            if not q:
                continue
            run_query(q)
            print("\n-----------------------------------\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cosine of Thrones CLI Runner")
    parser.add_argument("--question", "-q", type=str, help="Question to ask")

    args = parser.parse_args()

    if args.question:
        run_query(args.question)
    else:
        interactive_loop()


if __name__ == "__main__":
    main()