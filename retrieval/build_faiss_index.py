# build_faiss_index.py

import logging
import numpy as np
import faiss
from pathlib import Path
from embed_utils import load_chunks

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()

    # Load cached chunks and embeddings
    chunks_path = Path("retrieval/cache_data/chunks.pkl")
    embeddings_path = Path("retrieval/cache_data/embeddings.npy")

    if not chunks_path.exists() or not embeddings_path.exists():
        logger.error("Missing precomputed chunks or embeddings. Run process_real_data.py first.")
        return

    chunks = load_chunks(chunks_path)
    embeddings = np.load(embeddings_path)

    if len(chunks) != len(embeddings):
        logger.error("Mismatch between number of chunks and embeddings.")
        return

    d = embeddings.shape[1]  # dimension of embedding vectors
    logger.info(f"Embedding dimension: {d}")

    # Step 1: Create FAISS index (Flat L2 for now)
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    # Step 2: Save index to disk
    faiss.write_index(index, "retrieval/index/faiss_index.idx")
    logger.info(f"✅ FAISS index built and saved with {index.ntotal} vectors.")

if __name__ == "__main__":
    main()
