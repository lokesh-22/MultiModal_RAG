from sentence_transformers import SentenceTransformer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

LOCAL_DIR = "local_models"

def download_sentence_transformer():
    path = os.path.join(LOCAL_DIR, "all-MiniLM-L6-v2")
    print(f"Downloading SentenceTransformer -> {path}")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.save(path)
    print("✅ SentenceTransformer saved locally.")

def download_whisper():
    path = os.path.join(LOCAL_DIR, "whisper-small")
    print(f"Downloading Whisper -> {path}")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    processor.save_pretrained(path)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.save_pretrained(path)
    print("✅ Whisper model saved locally.")

if __name__ == "__main__":
    os.makedirs(LOCAL_DIR, exist_ok=True)
    download_sentence_transformer()
    download_whisper()
    print("\n🎉 All models downloaded and stored in local_models/ folder.")
