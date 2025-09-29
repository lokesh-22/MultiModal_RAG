import ollama
import os
from dotenv import load_dotenv
load_dotenv()
def describe_image(image_path):
    prompt = "Describe this image in detail. If text is present, extract it."
    response = ollama.chat(
        model=os.getenv("OLLAMA_VL_MODEL", "qwen2.5vl:7b"),
        messages=[
            {"role": "system", "content": "You are an image caption and OCR assistant."},
            {"role": "user", "content": prompt, "images": [image_path]}
        ]
    )
    return response["message"]["content"]
