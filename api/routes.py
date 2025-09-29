import os
import json
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form
from modules.rag_pipeline import process_inputs
from modules.embedding_store import add_to_index, save_index, METADATA_FILE
from modules.pdf_processor import process_pdf
from modules.image_processor import describe_image
from modules.audio_processor import transcribe_audio
from modules.retriever import retrieve_answer
from modules.utils import chunk_text_with_overlap

router = APIRouter()

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file (PDF/Image/Audio), process & store in FAISS.
    """
    file_ext = os.path.splitext(file.filename)[1].lower()
    save_path = f"data/uploads/{file.filename}"
    os.makedirs("data/uploads", exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    if file_ext == ".pdf":
        results = process_pdf(save_path)
        for r in results:
            chunks = chunk_text_with_overlap(r["text"])
            for chunk in chunks:
                add_to_index(chunk, "text", file.filename, page_num=r["page"])

    elif file_ext in [".png", ".jpg", ".jpeg"]:
        caption = describe_image(save_path)
        chunks = chunk_text_with_overlap(caption)
        for chunk in chunks:
            add_to_index(chunk, "image", file.filename)

    elif file_ext in [".mp3", ".wav", ".m4a"]:
        transcript = transcribe_audio(save_path)
        chunks = chunk_text_with_overlap(transcript)
        for chunk in chunks:
            add_to_index(chunk, "audio", file.filename)

    else:
        return {"error": "Unsupported file type"}

    save_index()
    return {"message": f"{file.filename} processed & indexed."}


@router.post("/ask/")
async def ask_question(query: str = Form(...), file: UploadFile | None = File(None)):
    """
    Ask a question. If a PDF/Image/Audio file is attached, first extract its
    content (like in /upload but without indexing) and append to the query.
    Then retrieve the answer from FAISS + LLM.
    """
    augmented_query = query

    # If a file is provided, extract its content and append to the query
    if file is not None:
        file_ext = os.path.splitext(file.filename)[1].lower()
        os.makedirs("data/uploads", exist_ok=True)
        # Use a unique filename to avoid collisions
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        save_path = f"data/uploads/ask_{timestamp}_{file.filename}"

        with open(save_path, "wb") as f:
            f.write(await file.read())

        extracted_text = None

        if file_ext == ".pdf":
            results = process_pdf(save_path)  # list of {"page": int, "text": str}
            extracted_text = "\n\n".join(r["text"] for r in results if r.get("text"))

        elif file_ext in [".png", ".jpg", ".jpeg"]:
            extracted_text = describe_image(save_path)

        elif file_ext in [".mp3", ".wav", ".m4a"]:
            extracted_text = transcribe_audio(save_path)

        else:
            return {"error": "Unsupported file type"}

        if extracted_text and extracted_text.strip():
            augmented_query = (
                f"{query}\n\nAdditional context from attached file ({file.filename}):\n{extracted_text}"
            )

    answer = retrieve_answer(augmented_query)
    return {"answer": answer}


@router.get("/documents/")
async def get_all_documents():
    """
    Get a list of all uploaded documents with their metadata.
    """
    try:
        # Check if metadata file exists
        if not os.path.exists(METADATA_FILE):
            return {"documents": [], "total_count": 0, "message": "No documents uploaded yet"}
        
        # Load metadata
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata_store = json.load(f)
        
        # Group by source file to get unique documents
        documents_map = {}
        
        for chunk_id, metadata in metadata_store.items():
            source_file = metadata["source_file"]
            modality = metadata["modality"]
            
            if source_file not in documents_map:
                # Get file stats if file exists
                file_path = f"data/uploads/{source_file}"
                file_size = None
                upload_date = None
                
                if os.path.exists(file_path):
                    file_stats = os.stat(file_path)
                    file_size = file_stats.st_size
                    upload_date = datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                
                documents_map[source_file] = {
                    "filename": source_file,
                    "modality": modality,
                    "file_size": file_size,
                    "upload_date": upload_date,
                    "chunk_count": 0,
                    "pages": set() if modality == "text" else None
                }
            
            # Increment chunk count
            documents_map[source_file]["chunk_count"] += 1
            
            # Add page numbers for PDF files
            if modality == "text" and metadata.get("page_num"):
                documents_map[source_file]["pages"].add(metadata["page_num"])
        
        # Convert sets to sorted lists and prepare final response
        documents = []
        for doc_info in documents_map.values():
            if doc_info["pages"] is not None:
                doc_info["pages"] = sorted(list(doc_info["pages"]))
                doc_info["page_count"] = len(doc_info["pages"])
            else:
                doc_info.pop("pages", None)
                doc_info["page_count"] = None
            
            documents.append(doc_info)
        
        # Sort by upload date (most recent first)
        documents.sort(key=lambda x: x["upload_date"] or "", reverse=True)
        
        return {
            "documents": documents,
            "total_count": len(documents),
            "total_chunks": len(metadata_store)
        }
        
    except Exception as e:
        return {"error": f"Failed to retrieve documents: {str(e)}"}


@router.get("/documents/{filename}")
async def get_document_details(filename: str):
    """
    Get detailed information about a specific document including all its chunks.
    """
    try:
        # Check if metadata file exists
        if not os.path.exists(METADATA_FILE):
            return {"error": "No documents found"}
        
        # Load metadata
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata_store = json.load(f)
        
        # Find all chunks for this document
        document_chunks = []
        document_info = None
        
        for chunk_id, metadata in metadata_store.items():
            if metadata["source_file"] == filename:
                document_chunks.append({
                    "chunk_id": chunk_id,
                    "page_num": metadata.get("page_num"),
                    "modality": metadata["modality"],
                    "text_excerpt": metadata["text_excerpt"][:200] + "..." if len(metadata["text_excerpt"]) > 200 else metadata["text_excerpt"]
                })
                
                # Set document info from first chunk
                if document_info is None:
                    file_path = f"data/uploads/{filename}"
                    file_size = None
                    upload_date = None
                    
                    if os.path.exists(file_path):
                        file_stats = os.stat(file_path)
                        file_size = file_stats.st_size
                        upload_date = datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                    
                    document_info = {
                        "filename": filename,
                        "modality": metadata["modality"],
                        "file_size": file_size,
                        "upload_date": upload_date,
                        "chunk_count": 0
                    }
        
        if not document_chunks:
            return {"error": f"Document '{filename}' not found"}
        
        # Sort chunks by page number for PDFs
        if document_chunks[0]["modality"] == "text":
            document_chunks.sort(key=lambda x: x["page_num"] or 0)
        
        document_info["chunk_count"] = len(document_chunks)
        
        return {
            "document_info": document_info,
            "chunks": document_chunks
        }
        
    except Exception as e:
        return {"error": f"Failed to retrieve document details: {str(e)}"}
