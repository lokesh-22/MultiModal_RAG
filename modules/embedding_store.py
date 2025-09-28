import os
import json
import uuid
import faiss
from sentence_transformers import SentenceTransformer
from config import VECTOR_DIR

os.makedirs(VECTOR_DIR, exist_ok=True)

EMBEDDING_MODEL = SentenceTransformer("local_models/all-MiniLM-L6-v2")
METADATA_FILE = os.path.join(VECTOR_DIR, "metadata.json")
INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")

# Load existing FAISS index if exists
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata_store = json.load(f)
else:
    index = faiss.IndexFlatL2(384)  # dimension for MiniLM-L6-v2
    metadata_store = {}

def add_to_index(text, modality, source_file, page_num=None):
    chunk_id = str(uuid.uuid4())
    embedding = EMBEDDING_MODEL.encode([text])
    index.add(embedding)
    
    metadata_store[chunk_id] = {
        "id": chunk_id,
        "modality": modality,
        "source_file": source_file,
        "page_num": page_num,
        "text_excerpt": text
    }

def save_index():
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, indent=2, ensure_ascii=False)
