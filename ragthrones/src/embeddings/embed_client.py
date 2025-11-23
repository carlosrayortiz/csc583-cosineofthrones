"""
EmbedClient: thin wrapper around OpenAI embedding API.
Matches the behavior of your original notebook code.
"""

import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


class EmbedClient:
    """
    Wrapper for embedding creation using OpenAI client.

    Usage:
        client = EmbedClient()
        vec = client.embed("What happened in King's Landing?")
    """

    def __init__(self, model_name: str = "text-embedding-3-large"):
        self.model = model_name

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment. Export it first.")

        self.client = OpenAI(api_key=api_key)

    def embed(self, text: str):
        """
        Compute an embedding for a single string.
        Returns a Python list of floats.
        """

        resp = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )

        # original notebook used: .data[0].embedding
        return resp.data[0].embedding

    def embed_batch(self, texts):
        """
        Batch embedding support (list of strings).
        Returns a list of embeddings, one per string.
        """

        resp = self.client.embeddings.create(
            model=self.model,
            input=texts
        )

        return [item.embedding for item in resp.data]