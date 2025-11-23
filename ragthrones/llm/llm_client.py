"""
Unified LLM client for all Cosine of Thrones agents.
Exports:
- GEN_MODEL
- get_llm_client()
- llm_client  (singleton)
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
# Construct new OpenAI client
# ----------------------------------------------------------
def get_llm_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=api_key)


# ----------------------------------------------------------
# Singleton instance used by all modules
# ----------------------------------------------------------
_llm_instance = None

def llm_client():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = get_llm_client()
    return _llm_instance


# ----------------------------------------------------------
# Minimal wrapper around Chat Completions API
# ----------------------------------------------------------
def llm_chat(prompt: str,
             model: str = None,
             temperature: float = 0.1) -> str:
    """
    Simple helper for:
    llm_chat("Hello") → returns text only.
    """

    if model is None:
        model = GEN_MODEL

    client = llm_client()

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()