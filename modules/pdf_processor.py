import fitz  # this comes from PyMuPDF

from PIL import Image
import os
import ollama

def extract_page_with_vllm(page, page_number):
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    temp_path = f"temp/page_{page_number}.png"
    os.makedirs("temp", exist_ok=True)
    img.save(temp_path, "PNG")
    
    prompt = (
        "Extract all text from this page. "
        "If a region contains an image with text, read it. "
        "If a region contains a non-text image, describe it in detail."
    )
    response = ollama.chat(
        model="qwen2.5vl:7b",
        messages=[
            {"role": "system", "content": "You are a document OCR and image description assistant."},
            {"role": "user", "content": prompt, "images": [temp_path]}
        ]
    )
    return response["message"]["content"]

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    results = []
    for i, page in enumerate(doc):
        vllm_text = extract_page_with_vllm(page, i+1)
        results.append({
            "page": i+1,
            "text": vllm_text
        })
    return results
