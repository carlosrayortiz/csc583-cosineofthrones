from ragthrones.pipelines.multi_agent_graph import app, AgentState


def run_graph(question: str, trivia_mode: bool = None):
    """
    Executes the full Cosine of Thrones LangGraph pipeline and returns the final AgentState.

    Args:
        question (str): The input question.
        trivia_mode (bool | None):
            - True  -> force TRIVIA_ANSWER_PROMPT
            - False -> force normal ANSWER_PROMPT
            - None  -> auto-detect inside synthesizer (default)

    Returns:
        AgentState: The final structured state containing:
            - answer
            - evidence_text
            - reranked retrieval
            - causal/emotion/narrative agent outputs
            - NSS scoring (if hooked in)
            - logs
    """

    # Initialize state
    state = AgentState(question=question)

    # Optional manual override
    if trivia_mode is not None:
        state.trivia_mode = trivia_mode

    # Run the graph
    raw = app.invoke(state)

    # Reconstruct into AgentState (LangGraph returns a dict)
    final_state = AgentState(**raw)

    return final_state