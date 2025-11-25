"""
Basic RAG Agent for Cosine of Thrones
-------------------------------------

This agent wraps the RetrievalAgent tool-calling graph
and returns a clean Pandas DataFrame of retrieved chunks.

It is the simplest RAG path, used when the question is not
factual / temporal / narrative-specific.

Outputs:
- retrieved DataFrame (text, season, episode, etc.)
- logs of tool invocations
"""

import json
import pandas as pd

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from ragthrones.agents.retrieval_agent import RetrievalAgent


def basic_rag_agent(question: str) -> pd.DataFrame:
    """
    Runs the simple RetrievalAgent graph on the question.
    Returns a DataFrame of retrieval results.
    """

    # Run the RetrievalAgent graph
    ret = RetrievalAgent.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "llm_calls": 0
        }
    )

    # Extract the last tool message (contains JSON result)
    msgs = ret["messages"]
    tool_msgs = [m for m in msgs if m.type == "tool"]

    if not tool_msgs:
        # No structured retrieval results -> empty dataframe
        return pd.DataFrame()

    # Parse tool JSON payload
    payload = json.loads(tool_msgs[-1].content)

    # Convert into DataFrame
    results = payload.get("results", [])
    df = pd.DataFrame(results)

    return df