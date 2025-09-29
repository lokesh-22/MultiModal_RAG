import os
import torch
from faster_whisper import WhisperModel

# Pick device based on CUDA availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Use model size string and local cache directory
MODEL_SIZE = "small"  # Change to "medium", "large-v2", etc. if needed
LOCAL_DIR = "local_models"

print(f"📥 Loading Faster-Whisper model '{MODEL_SIZE}' with cache in {LOCAL_DIR} on {DEVICE} ...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, download_root=LOCAL_DIR)
print("✅ Model loaded successfully.")

def transcribe_audio(audio_path):
    """
    Transcribe audio using locally cached faster-whisper model (no timestamps).
    """
    segments, info = model.transcribe(audio_path, beam_size=2, vad_filter=True)
    transcript = " ".join([segment.text for segment in segments])
    return transcript
