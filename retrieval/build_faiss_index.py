import logging
from pathlib import Path
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
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

    chunks_path = Path("retrieval/cache_data/chunks.pkl")
    if not chunks_path.exists():
        logger.error(f"Chunks file not found at {chunks_path}")
        return

    chunks = load_chunks(chunks_path)
    logger.info(f"✅ Loaded {len(chunks)} chunks.")

    # Convert chunks to LangChain Documents
    documents = []
    for chunk in chunks:
        # Assuming chunk is dict with 'text' key; adjust if your chunks differ
        if isinstance(chunk, dict) and "text" in chunk:
            content = chunk["text"]
        else:
            content = str(chunk)  # fallback to string conversion
        documents.append(Document(page_content=content))

    logger.info(f"Converted chunks to {len(documents)} LangChain Documents.")

    # Initialize HuggingFace Embeddings (note the deprecation warning, consider updating package later)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build FAISS index from documents
    vectorstore = FAISS.from_documents(documents, embedding_model)
    index_dir = Path("retrieval/index")
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    logger.info(f"✅ FAISS index built and saved to {index_dir}")

if __name__ == "__main__":
    main()
