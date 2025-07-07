import os
from pathlib import Path
import numpy as np

from retrieval.embed_utils import (
    DocumentLoader,
    TextSplitter,
    Embedder,
    save_chunks,
    load_chunks
)

# Define test input folder (make sure test_data/ contains small .txt and .pdf files)
TEST_DATA_DIR = Path("test_data")
CACHE_EMBEDDINGS_PATH = Path("test_outputs/embeddings.npy")
CACHE_CHUNKS_PATH = Path("test_outputs/chunks.pkl")

# Ensure output dir exists
CACHE_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

def test_document_loader():
    loader = DocumentLoader(TEST_DATA_DIR)
    docs = loader.load()
    assert isinstance(docs, list)
    assert all("text" in doc and "source" in doc and "type" in doc for doc in docs)
    print(f"[✓] Loaded {len(docs)} documents")

def test_text_splitter():
    loader = DocumentLoader(TEST_DATA_DIR)
    docs = loader.load()
    splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split(docs)
    assert len(chunks) > 0
    assert "chunk_index" in chunks[0]
    print(f"[✓] Split into {len(chunks)} chunks")

def test_embedder():
    loader = DocumentLoader(TEST_DATA_DIR)
    docs = loader.load()
    splitter = TextSplitter()
    chunks = splitter.split(docs)

    embedder = Embedder(batch_size=16, cache_path=CACHE_EMBEDDINGS_PATH)
    embeddings = embedder.embed(chunks)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(chunks)
    print(f"[✓] Computed embeddings for {len(chunks)} chunks")

def test_chunk_save_load():
    loader = DocumentLoader(TEST_DATA_DIR)
    docs = loader.load()
    splitter = TextSplitter()
    chunks = splitter.split(docs)

    save_chunks(chunks, CACHE_CHUNKS_PATH)
    loaded_chunks = load_chunks(CACHE_CHUNKS_PATH)

    assert len(chunks) == len(loaded_chunks)
    assert chunks[0]["text"] == loaded_chunks[0]["text"]
    print(f"[✓] Saved and reloaded {len(chunks)} chunks")

if __name__ == "__main__":
    print("🧪 Running Embed Utils Tests")
    test_document_loader()
    test_text_splitter()
    test_embedder()
    test_chunk_save_load()
    print("✅ All tests passed!")
