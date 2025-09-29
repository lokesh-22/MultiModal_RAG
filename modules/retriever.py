import ollama
from modules.models import (
    get_embedding_model,
    load_faiss_index,
    load_metadata_store,
)
import os
from dotenv import load_dotenv 

load_dotenv()

# Centralized loaders
EMBEDDING_MODEL = get_embedding_model()
index = load_faiss_index(create_if_missing=True)
metadata_store = load_metadata_store()

def retrieve_chunks(query, top_k=3):
    """
    Retrieve top_k chunks from FAISS based on query embedding.
    """
    # Guard when index is empty or metadata missing
    try:
        ntotal = getattr(index, 'ntotal', 0)
    except Exception:
        ntotal = 0

    if ntotal == 0 or not metadata_store:
        return []

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
    # If no documents are indexed, short-circuit with a helpful message
    try:
        if getattr(index, 'ntotal', 0) == 0 or not metadata_store:
            return (
                "No documents are indexed yet. Please upload a PDF/Image/Audio via /upload "
                "before asking questions."
            )
    except Exception:
        return (
            "Vector index unavailable. Please upload a PDF/Image/Audio via /upload to initialize the index."
        )

    # Step 1: Ask Qwen what context it needs (optional, can guide retrieval)
    instruction = (
        "You are an assistant with access to a vector store of documents.\n"
        "Decide what information you need to answer the user's question. "
        "Return a clear query or keywords for retrieval.\n"
        f"User Question: {query}"
    )
    
    retrieval_hint = ollama.chat(
        model=os.getenv("OLLAMA_VL_MODEL", "qwen3:8b"),
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
        model=os.getenv("OLLAMA_VL_MODEL", "qwen3:8b"),
        messages=[
            {"role": "system", "content": "You are a helpful expert assistant."},
            {"role": "user", "content": final_prompt}
        ]
    )

    return response["message"]["content"]

# Example usage
if __name__ == "__main__":
    user_query = "What are the key points about caching in web applications?"
    answer = retrieve_answer(user_query, top_k=3)
    print("Answer with citations:\n", answer)
