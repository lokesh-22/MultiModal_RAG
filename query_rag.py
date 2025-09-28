import json
from sentence_transformers import SentenceTransformer
import faiss
from config import VECTOR_DIR

# Load embedding model
EMBEDDING_MODEL = SentenceTransformer("local_models/all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index(f"{VECTOR_DIR}/index.faiss")

# Load metadata store
with open(f"{VECTOR_DIR}/metadata.json", "r", encoding="utf-8") as f:
    metadata_store = json.load(f)

def retrieve_top_chunk(query):
    """
    Retrieve the single most similar chunk from FAISS for the given query.
    Returns a dictionary with text and citation metadata.
    """
    # Encode the query
    q_emb = EMBEDDING_MODEL.encode([query])
    
    # Search for the top-1 most similar chunk
    D, I = index.search(q_emb, 1)
    
    # Get the corresponding chunk ID and metadata
    all_keys = list(metadata_store.keys())
    top_idx = I[0][0]
    chunk_id = all_keys[top_idx]
    chunk_meta = metadata_store[chunk_id]
    
    # Prepare structured output
    result = {
        "text": chunk_meta["text_excerpt"],
        "citation": {
            "source_file": chunk_meta.get("source_file", "N/A"),
            "page_num": chunk_meta.get("page_num", "N/A"),
            "chunk_index": chunk_meta.get("chunk_index", "N/A")
        }
    }
    return result

if __name__ == "__main__":
    print("📚 Retrieval-only RAG System Ready!")
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        top_chunk = retrieve_top_chunk(query)
        print("\nMost Relevant Chunk:\n")
        print(top_chunk["text"])
        print("\nCitation Info:")
        print(f"Source File: {top_chunk['citation']['source_file']}")
        print(f"Page Number: {top_chunk['citation']['page_num']}")
        print(f"Chunk Index: {top_chunk['citation']['chunk_index']}")
