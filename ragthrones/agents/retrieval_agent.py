# ------------------------------------------------------------
# RetrievalAgent with real hybrid_search_aug() — FULL WORKING VERSION
# ------------------------------------------------------------

import os
import json

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
import operator

from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------
# MODEL CONFIG
# ------------------------------------------------------------
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")

model = init_chat_model(
    GEN_MODEL,
    temperature=0
)

# ------------------------------------------------------------
# LOAD VECTORSTORE + SET GLOBALS FOR hybrid_search_aug
# ------------------------------------------------------------
from ragthrones.retrieval.load_vectorstore import load_all_vectorstore
V = load_all_vectorstore()

# Bind globals that hybrid_search_aug expects
df_aug = V["df_aug"]
index = V["faiss"]
bm25 = V["bm25"]
client = V["embed_client"]

# spaCy model (used in hybrid_search_aug)
import spacy
nlp = spacy.load("en_core_web_sm")

# The embedding model used when computing query embeddings
EMBED_MODEL = "text-embedding-3-large"

# Import real retrieval function
from ragthrones.retrieval.hybrid_search import hybrid_search_aug


# ------------------------------------------------------------
# 1. RETRIEVAL TOOL — Always return VALID JSON
# ------------------------------------------------------------
@tool("hybrid_retrieve", return_direct=True)
def hybrid_retrieve(query: str, topk: int = 10) -> str:
    """
    Execute hybrid FAISS + BM25 retrieval and return JSON payload:
    {
        "results": [ {...chunk...}, ... ]
    }
    """
    try:
        df = hybrid_search_aug(query=query, topk=topk)

        if df is None or len(df) == 0:
            return json.dumps({"results": []})

        return json.dumps({"results": df.to_dict(orient="records")})

    except Exception as e:
        return json.dumps({"error": str(e)})


# Register tools
tools = [hybrid_retrieve]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)


# ------------------------------------------------------------
# 2. AGENT STATE
# ------------------------------------------------------------
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# ------------------------------------------------------------
# 3. LLM Node — issues tool calls
# ------------------------------------------------------------
def llm_call(state: dict):
    """
    LLM decides how to call hybrid_retrieve.
    """

    sys_prompt = """
    You are the RetrievalAgent for the Cosine of Thrones system.

    Your tasks:
    - Rewrite the user's question into retrieval-friendly subqueries.
    - Call the hybrid_retrieve tool with those subqueries.
    - Do NOT keep looping forever. One round of retrieval is enough.
    """

    response = model_with_tools.invoke(
        [SystemMessage(content=sys_prompt)] + state["messages"]
    )

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ------------------------------------------------------------
# 4. TOOL NODE — executes the JSON-returning tool
# ------------------------------------------------------------
def tool_node(state: dict):
    """
    Execute any tool calls issued by the LLM.
    """
    outputs = []
    last = state["messages"][-1]

    for tc in last.tool_calls:
        tool = tools_by_name[tc["name"]]
        result = tool.invoke(tc["args"])

        outputs.append(
            ToolMessage(
                content=result,           # JSON already
                tool_call_id=tc["id"]
            )
        )

    return {"messages": outputs}


# ------------------------------------------------------------
# 5. Routing Logic — prevents infinite recursion
# ------------------------------------------------------------
from typing import Literal
from langgraph.graph import StateGraph, START, END

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    last = state["messages"][-1]

    # No tool calls → stop
    if not last.tool_calls:
        return END

    # If LLM has already issued tool calls once → stop
    if state.get("llm_calls", 0) >= 1:
        return END

    return "tool_node"


# ------------------------------------------------------------
# 6. Build RetrievalAgent Graph
# ------------------------------------------------------------
agent = StateGraph(MessagesState)

agent.add_node("llm_call", llm_call)
agent.add_node("tool_node", tool_node)

agent.add_edge(START, "llm_call")

agent.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)

agent.add_edge("tool_node", "llm_call")

RetrievalAgent = agent.compile()