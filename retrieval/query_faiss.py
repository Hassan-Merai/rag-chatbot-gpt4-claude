import logging
import numpy as np
import pickle
import faiss

from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Tuple

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
INDEX_PATH = Path("retrieval/index/faiss_index.idx")
CHUNKS_PATH = Path("retrieval/cache_data/chunks.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load files
def load_chunks(path: Path) -> List[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)

def load_index(path: Path, dim: int) -> faiss.IndexFlatL2:
    index = faiss.read_index(str(path))
    if index.d != dim:
        raise ValueError(f"Index dim mismatch: expected {dim}, got {index.d}")
    return index

# Search
def search(query: str, model: SentenceTransformer, index: faiss.IndexFlatL2, chunks: List[dict], k: int = 5) -> List[Tuple[str, float]]:
    logger.info(f"Searching for: \"{query}\"")

    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)

    results = []
    for i, dist in zip(indices[0], distances[0]):
        chunk = chunks[i]
        results.append((chunk["text"], dist))
    return results

def main():
    # Step 1: Load artifacts
    chunks = load_chunks(CHUNKS_PATH)
    model = SentenceTransformer(MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    index = load_index(INDEX_PATH, dim=embedding_dim)

    # Step 2: Run a query
    query = input("🔎 Enter your search query: ").strip()
    results = search(query, model, index, chunks)

    # Step 3: Display results
    print("\n📚 Top Results:\n" + "-" * 40)
    for i, (text, dist) in enumerate(results, 1):
        print(f"{i}. (Score: {dist:.4f})\n{text[:300]}...\n")

if __name__ == "__main__":
    main()
