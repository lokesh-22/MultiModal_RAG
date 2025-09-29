"""
Centralized model definitions and helpers used across the project.

Exposes:
- get_embedding_model(), get_embedding_dimension()
- load_faiss_index(create_if_missing=True), save_faiss_index(index)
- load_metadata_store(), save_metadata_store(store)
- get_whisper_model(model_size="base")
- OLLAMA_VL_MODEL, INDEX_FILE, METADATA_FILE
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict
from dotenv import load_dotenv
from config import VECTOR_DIR

load_dotenv()

# Configurable via env var; default VL model name for Ollama
OLLAMA_VL_MODEL = os.getenv("OLLAMA_VL_MODEL", "qwen2.5vl:7b")

# Ensure vectorstore folder exists
os.makedirs(VECTOR_DIR, exist_ok=True)
INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
METADATA_FILE = os.path.join(VECTOR_DIR, "metadata.json")

# Singletons
_embedding_model: Any = None
_whisper_model: Any = None


def get_embedding_model() -> Any:
    """Return a singleton SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "sentence-transformers is not installed. Please install it to use embeddings."
            ) from e
        _embedding_model = SentenceTransformer("local_models/all-MiniLM-L6-v2")
    return _embedding_model


def get_embedding_dimension() -> int:
    model = get_embedding_model()
    return int(getattr(model, "get_sentence_embedding_dimension", lambda: 384)())


def load_faiss_index(create_if_missing: bool = True):
    """Load FAISS index; create empty IndexFlatL2 if missing and allowed."""
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("faiss is not installed. Please install faiss to use the vector index.") from e

    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    if not create_if_missing:
        raise FileNotFoundError(f"FAISS index not found at {INDEX_FILE}")
    return faiss.IndexFlatL2(get_embedding_dimension())


def save_faiss_index(index) -> None:
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("faiss is not installed. Please install faiss to use the vector index.") from e
    faiss.write_index(index, INDEX_FILE)


def load_metadata_store() -> Dict:
    if not os.path.exists(METADATA_FILE):
        return {}
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata_store(store: Dict) -> None:
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


def get_whisper_model(model_size: str = "base"):
    global _whisper_model
    try:
        import whisper  # type: ignore
    except Exception as e:
        raise RuntimeError("openai-whisper is not installed. Please install it to use ASR.") from e

    if _whisper_model is None or getattr(_whisper_model, "model_name", None) != model_size:
        _whisper_model = whisper.load_model(model_size)
        try:
            _whisper_model.model_name = model_size  # type: ignore[attr-defined]
        except Exception:
            pass
    return _whisper_model


__all__ = [
    "get_embedding_model",
    "get_embedding_dimension",
    "load_faiss_index",
    "save_faiss_index",
    "load_metadata_store",
    "save_metadata_store",
    "get_whisper_model",
    "OLLAMA_VL_MODEL",
    "INDEX_FILE",
    "METADATA_FILE",
]
