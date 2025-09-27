from modules.rag_pipeline import process_inputs

if __name__ == "__main__":
    print("🚀 Starting multimodal ingestion...")
    process_inputs()
    print("✅ Ingestion complete. Embeddings stored in FAISS, text stored in processed/")
