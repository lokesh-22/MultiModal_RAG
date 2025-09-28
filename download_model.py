from sentence_transformers import SentenceTransformer

# Step 1: Load the model (requires internet)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Save it locally
model.save("local_models/all-MiniLM-L6-v2")

print("Model downloaded and saved locally!")
