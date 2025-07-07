# process_real_data.py

import logging
from pathlib import Path
from retrieval.embed_utils import DocumentLoader, TextSplitter, Embedder, save_chunks

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()

    data_folder = Path("data")
    if not data_folder.exists():
        logger.error(f"Data folder not found: {data_folder.resolve()}")
        return

    # Step 1: Load documents
    loader = DocumentLoader(data_folder)
    documents = loader.load()

    if not documents:
        logger.warning("No documents loaded. Add .pdf or .txt files to /data/")
        return

    # Step 2: Chunk documents
    splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split(documents)
    if not chunks:
        logger.warning("Text splitting returned 0 chunks.")
        return

    # Step 3: Save chunks
    save_chunks(chunks, "retrieval/cache_data/chunks.pkl")

    # Step 4: Embed
    embedder = Embedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32,
        cache_path="retrieval/cache_data/embeddings.npy"
    )
    embeddings = embedder.embed(chunks)

    logger.info(f"✅ Embedding complete: {len(embeddings)} vectors cached.")

if __name__ == "__main__":
    main()
