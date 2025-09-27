import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_DIR = os.path.join(BASE_DIR, "data/pdfs")
IMAGE_DIR = os.path.join(BASE_DIR, "data/images")
AUDIO_DIR = os.path.join(BASE_DIR, "data/audio")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")
