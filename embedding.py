import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load BGE embedding model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Load chunks from JSONL
chunks_path = "data/chunks_aiayn.jsonl"
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

# Extract text and prepare for embedding
texts = [chunk["text"] for chunk in chunks]

# Generate embeddings using BGE
embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
embeddings = np.array(embeddings, dtype=np.float32)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "data/paper_embeddings.index")

# Save full metadata
metadata = [chunk["metadata"] for chunk in chunks]
with open("data/chunks_aiayn_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("✅ FAISS index and full metadata saved to 'data/'")