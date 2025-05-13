import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load BGE embedding model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Load chunks from JSONL
chunks_path = "data/chunks.jsonl"
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

# Extract text and prepare for embedding
texts = [chunk["text"] for chunk in chunks]

# Generate embeddings using BGE
# BGE expects queries and documents to be embedded slightly differently.
# For documents, no prefix is required.
embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
embeddings = np.array(embeddings, dtype=np.float32)  # FAISS requires float32

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # Euclidean distance
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "paper_embeddings.index")

# Save metadata to reference chunks later
metadata = [
    {
        "paper_title": chunk["paper_title"],
        "heading": chunk["heading"],
        "chunk_index": chunk["chunk_index"]
    }
    for chunk in chunks
]
with open("chunks_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("✅ FAISS index built using BGE embeddings and saved as 'paper_embeddings.index'.")

