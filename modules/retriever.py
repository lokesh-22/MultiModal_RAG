import faiss
import json
from sentence_transformers import SentenceTransformer
import ollama
from config import VECTOR_DIR

# Load embeddings + FAISS index
EMBEDDING_MODEL = SentenceTransformer("local_models/all-MiniLM-L6-v2")
index = faiss.read_index(f"{VECTOR_DIR}/index.faiss")
with open(f"{VECTOR_DIR}/metadata.json", "r", encoding="utf-8") as f:
    metadata_store = json.load(f)

def retrieve_chunks(query, top_k=3):
    """
    Retrieve top_k chunks from FAISS based on query embedding.
    """
    q_emb = EMBEDDING_MODEL.encode([query])
    D, I = index.search(q_emb, top_k)
    results = []
    all_keys = list(metadata_store.keys())
    for idx in I[0]:
        chunk_id = all_keys[idx]
        meta = metadata_store[chunk_id]
        results.append((meta["text_excerpt"], meta))
    return results

def retrieve_answer(query, top_k=3):
    """
    LLM-driven RAG: Ask Qwen3 what information it needs,
    then retrieve relevant chunks, and answer with citations.
    """
    # Step 1: Ask Qwen what context it needs (optional, can guide retrieval)
    instruction = (
        "You are an assistant with access to a vector store of documents.\n"
        "Decide what information you need to answer the user's question. "
        "Return a clear query or keywords for retrieval.\n"
        f"User Question: {query}"
    )
    
    retrieval_hint = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": "You are a helpful expert assistant."},
            {"role": "user", "content": instruction}
        ]
    )["message"]["content"]

    # Step 2: Use the hint to fetch top chunks
    chunks = retrieve_chunks(retrieval_hint, top_k)

    # Step 3: Combine retrieved chunks with metadata for citations
    context_text = ""
    for chunk, meta in chunks:
        context_text += f"[Source: {meta['source_file']}, page: {meta.get('page_num', 'N/A')}]\n{chunk}\n\n"

    # Step 4: Final answer using context + user question
    final_prompt = (
        f"Answer the question using ONLY the context below.\n\n"
        f"Context:\n{context_text}\nQuestion: {query}"
    )

    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": "You are a helpful expert assistant."},
            {"role": "user", "content": final_prompt}
        ]
    )

    return response["message"]["content"]

# Example usage
if __name__ == "__main__":
    user_query = "What are the key points about caching in web applications?"
    answer = rag_answer(user_query, top_k=3)
    print("Answer with citations:\n", answer)
