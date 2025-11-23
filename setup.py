import os

dirs = [
    "ragthrones/data/raw",
    "ragthrones/data/processed",
    "ragthrones/data/artifacts",
    "ragthrones/config",
    "ragthrones/src/ragthrones/ingest",
    "ragthrones/src/ragthrones/embeddings",
    "ragthrones/src/ragthrones/retrieval",
    "ragthrones/src/ragthrones/pipelines",
    "ragthrones/src/ragthrones/app",
    "ragthrones/src/ragthrones/utils",
    "ragthrones/scripts",
    "ragthrones/notebooks"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

open("ragthrones/README.md", "w").close()
open("ragthrones/requirements.txt", "w").close()

print("Project directories created.")