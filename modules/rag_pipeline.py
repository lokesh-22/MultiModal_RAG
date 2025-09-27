import os
import json
from modules.pdf_processor import process_pdf
from modules.image_processor import describe_image
from modules.audio_processor import transcribe_audio
from modules.embedding_store import add_to_index, save_index
from config import PDF_DIR, IMAGE_DIR, AUDIO_DIR, PROCESSED_DIR

def save_processed(data, filename):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def chunk_text_with_overlap(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def process_inputs():
    # PDFs
    for pdf in os.listdir(PDF_DIR):
        pdf_path = os.path.join(PDF_DIR, pdf)
        results = process_pdf(pdf_path)
        save_processed(results, f"{os.path.splitext(pdf)[0]}_processed.json")
        for r in results:
            chunks = chunk_text_with_overlap(r["text"], chunk_size=300, overlap=50)
            for chunk in chunks:
                add_to_index(chunk, "text", pdf, page_num=r["page"])

    # Images
    for img in os.listdir(IMAGE_DIR):
        img_path = os.path.join(IMAGE_DIR, img)
        caption = describe_image(img_path)
        save_processed({"file": img, "text": caption}, f"{os.path.splitext(img)[0]}_processed.json")
        chunks = chunk_text_with_overlap(caption, chunk_size=300, overlap=50)
        for chunk in chunks:
            add_to_index(chunk, "image", img)

    # Audio
    for audio in os.listdir(AUDIO_DIR):
        audio_path = os.path.join(AUDIO_DIR, audio)
        transcript = transcribe_audio(audio_path)
        save_processed({"file": audio, "text": transcript}, f"{os.path.splitext(audio)[0]}_processed.json")
        chunks = chunk_text_with_overlap(transcript, chunk_size=300, overlap=50)
        for chunk in chunks:
            add_to_index(chunk, "audio", audio)

    save_index()
