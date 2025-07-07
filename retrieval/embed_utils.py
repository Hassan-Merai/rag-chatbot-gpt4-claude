import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF (fitz) is not installed. Please install it using 'pip install PyMuPDF'.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
Document = Dict[str, Any]
Embedding = np.ndarray

class DocumentLoader:
    """
    Unified loader for .txt, .pdf, and other supported document types.
    Returns documents with metadata for provenance.
    """
    def __init__(self, folder_path: Union[str, Path]):
        self.folder_path = Path(folder_path)

    def load(self) -> List[Document]:
        docs: List[Document] = []
        for path in self.folder_path.glob("**/*"):
            if path.suffix.lower() == ".txt":
                try:
                    docs.append(self._load_txt(path))
                except Exception as e:
                    logger.warning(f"Failed to load TXT {path.name}: {e}")
            elif path.suffix.lower() == ".pdf":
                try:
                    docs.append(self._load_pdf(path))
                except Exception as e:
                    logger.warning(f"Failed to load PDF {path.name}: {e}")
            else:
                logger.debug(f"Skipping unsupported file type: {path.name}")
        logger.info(f"Loaded {len(docs)} documents from {self.folder_path}")
        return docs

    def _load_txt(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8")
        return {"text": text, "source": str(path), "type": "txt"}

    def _load_pdf(self, path: Path) -> Document:
        doc = fitz.open(path)
        text = []
        for page in doc:
            text.append(page.get_text())
        full_text = "\n".join(text)
        return {"text": full_text, "source": str(path), "type": "pdf"}

class TextSplitter:
    """
    Splits documents into chunks, preserving metadata.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def split(self, documents: List[Document]) -> List[Document]:
        chunks: List[Document] = []
        for doc in documents:
            texts = self.splitter.split_text(doc["text"])
            for i, chunk in enumerate(texts):
                chunks.append({
                    "text": chunk,
                    "source": doc["source"],
                    "type": doc["type"],
                    "chunk_index": i
                })
        logger.info(f"Split into {len(chunks)} chunks from {len(documents)} documents")
        return chunks

class Embedder:
    """
    Lazy initialization of embedding model with batching and optional caching.
    """
    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        batch_size: int = 32,
        cache_path: Optional[Union[str, Path]] = None
    ):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.cache_path = Path(cache_path) if cache_path else None
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def embed(self, chunks: List[Document]) -> np.ndarray:
        texts = [c["text"] for c in chunks]
        if self.cache_path and self.cache_path.exists():
            logger.info(f"Loading embeddings from cache at {self.cache_path}")
            return np.load(self.cache_path)

        logger.info(f"Computing embeddings for {len(texts)} chunks")
        embeddings: Embedding = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        if self.cache_path:
            np.save(self.cache_path, embeddings)
            logger.info(f"Saved embeddings cache to {self.cache_path}")
        return embeddings

    def clear_cache(self) -> None:
        if self.cache_path and self.cache_path.exists():
            self.cache_path.unlink()
            logger.info(f"Cleared embeddings cache at {self.cache_path}")

# Optional persistence utilities

def save_chunks(chunks: List[Document], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(chunks, f)
    logger.info(f"Saved {len(chunks)} chunks to {path}")


def load_chunks(path: Union[str, Path]) -> List[Document]:
    with open(path, 'rb') as f:
        chunks = pickle.load(f)
    logger.info(f"Loaded {len(chunks)} chunks from {path}")
    return chunks
