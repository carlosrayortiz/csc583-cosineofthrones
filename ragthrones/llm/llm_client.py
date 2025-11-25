"""
Unified LLM client for all Cosine of Thrones agents.
Exports:
- GEN_MODEL
- get_llm_client()
- llm_client   (singleton OpenAI client)
- llm_chat(prompt)
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------
# Default model used everywhere unless overridden
# ----------------------------------------------------------
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")


# ----------------------------------------------------------
# Constructor for fresh OpenAI client
# ----------------------------------------------------------
def get_llm_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=api_key)


# ----------------------------------------------------------
# Singleton OpenAI client instance
# ----------------------------------------------------------
_llm_instance = get_llm_client()

# IMPORTANT:
# llm_client is NOW a real OpenAI client, NOT a function.
llm_client = _llm_instance


# ----------------------------------------------------------
# Convenience wrapper for simple chat completions
# ----------------------------------------------------------
def llm_chat(prompt: str,
             model: str = None,
             temperature: float = 0.1) -> str:
    """
    Simple helper for quick chat calls:
    llm_chat("Hello") → returns just text.
    """

    if model is None:
        model = GEN_MODEL

    response = llm_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()