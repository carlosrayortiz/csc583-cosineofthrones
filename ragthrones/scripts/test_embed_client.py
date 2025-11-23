import sys, os
sys.path.append(os.path.abspath("ragthrones/src"))

from ragthrones.embeddings.embed_client import EmbedClient

client = EmbedClient()

vec = client.embed("Winter is coming.")
print("Vector length:", len(vec))
print("First 5 dims:", vec[:5])