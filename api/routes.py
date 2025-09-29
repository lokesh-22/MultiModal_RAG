import os
import json
import shutil
import mimetypes
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from modules.rag_pipeline import process_inputs
from modules.embedding_store import add_to_index, save_index, METADATA_FILE
from modules.pdf_processor import process_pdf
from modules.image_processor import describe_image
from modules.audio_processor import transcribe_audio
from modules.retriever import retrieve_answer
from modules.utils import chunk_text_with_overlap
from config import BASE_DIR, VECTOR_DIR, PROCESSED_DIR
from modules.models import get_embedding_dimension

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

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
    # answer is a dict {"answer": str, "citations": [...]}
    return answer


@router.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    """
    Return a file from the uploads folder by filename.
    Safely resolves the path to prevent directory traversal.
    """
    uploads_dir = os.path.join(BASE_DIR, "data", "uploads")
    safe_name = os.path.basename(filename)
    file_path = os.path.join(uploads_dir, safe_name)

    real_uploads = os.path.realpath(uploads_dir)
    real_file = os.path.realpath(file_path)

    # Prevent path traversal and ensure file exists
    if not real_file.startswith(real_uploads) or not os.path.isfile(real_file):
        # Keep consistency with other routes that return JSON errors
        # You can switch to: raise HTTPException(status_code=404, detail=...) if preferred
        return {"error": f"File '{filename}' not found"}

    return FileResponse(real_file, filename=safe_name)


@router.delete("/reset/")
async def reset_all():
    """
    Danger: Clear all data.
    - Deletes uploads (data/uploads)
    - Deletes vectorstore files (index.faiss, metadata.json)
    - Deletes processed outputs (processed/ and data/processed)
    - Resets in-memory index/metadata and writes fresh empty files
    """
    uploads_dir = os.path.join(BASE_DIR, "data", "uploads")
    data_processed_dir = os.path.join(BASE_DIR, "data", "processed")
    temp_dir = os.path.join(BASE_DIR, "temp")

    deleted = {"files": 0, "dirs": 0}

    def rm_tree(path: str):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            deleted["dirs"] += 1

    def rm_file(path: str):
        if os.path.isfile(path):
            try:
                os.remove(path)
                deleted["files"] += 1
            except Exception:
                pass

    # 1) Clear uploads
    if os.path.isdir(uploads_dir):
        shutil.rmtree(uploads_dir, ignore_errors=True)
        os.makedirs(uploads_dir, exist_ok=True)

    # 2) Clear vectorstore (index + metadata, including nested dirs)
    if os.path.isdir(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR, ignore_errors=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)

    # 3) Clear processed outputs
    if os.path.isdir(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if os.path.isdir(data_processed_dir):
        shutil.rmtree(data_processed_dir, ignore_errors=True)
    os.makedirs(data_processed_dir, exist_ok=True)

    # 4) Clear temp folder used by PDF processing (if present)
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 5) Reset in-memory embedding store and write fresh empty index/metadata
    try:
        from modules import embedding_store as es
        dim = get_embedding_dimension()
        if faiss is not None:
            es.index = faiss.IndexFlatL2(dim)
        else:
            es.index = None  # type: ignore
        es.metadata_store = {}
        # Write fresh empty index/metadata to prevent future read errors
        if faiss is not None and es.index is not None:
            faiss.write_index(es.index, os.path.join(VECTOR_DIR, "index.faiss"))
        with open(os.path.join(VECTOR_DIR, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({}, f)
    except Exception as e:
        # Non-fatal; continue
        pass

    return {
        "message": "All data cleared: uploads, vectorstore, processed outputs.",
        "uploads_dir": uploads_dir,
        "vectorstore_dir": VECTOR_DIR,
        "processed_dirs": [PROCESSED_DIR, data_processed_dir],
        "deleted_counts": deleted,
    }


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
async def get_document_file(filename: str):
    uploads_dir = os.path.join(BASE_DIR, "data", "uploads")
    safe_name = os.path.basename(filename)
    file_path = os.path.join(uploads_dir, safe_name)

    real_uploads = os.path.realpath(uploads_dir)
    real_file = os.path.realpath(file_path)

    if not real_file.startswith(real_uploads) or not os.path.isfile(real_file):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    # Guess MIME type to help the browser render inline (pdf/image/audio)
    media_type = mimetypes.guess_type(real_file)[0] or "application/octet-stream"

    # Force inline display instead of download
    headers = {
        "X-Source-Path": os.path.relpath(real_file, BASE_DIR),
        "Content-Disposition": f'inline; filename="{safe_name}"'
    }
    # Do not pass filename= to avoid automatic 'attachment' disposition
    return FileResponse(real_file, media_type=media_type, headers=headers)
