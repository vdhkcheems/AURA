import faiss
import json
from sentence_transformers import SentenceTransformer
import torch

# Load model (BGE-base)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

# Load FAISS index
index = faiss.read_index("data/paper_embeddings.index")

# Load metadata
with open("data/chunks_aiayn_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load chunk texts
chunks = []
with open("data/chunks_aiayn.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

# Utility: Build full heading from section hierarchy
def build_heading(meta):
    parts = [meta.get("section"), meta.get("subsection"), meta.get("subsubsection")]
    return " > ".join([p for p in parts if p])

# Embed query using BGE (with recommended prefix)
def embed_query(query):
    prompt = "Represent this sentence for retrieval: " + query
    return model.encode([prompt], convert_to_numpy=True)

# Retrieve top-k chunks based on the query
def retrieve_top_k(query, k=5):
    query_embedding = embed_query(query)
    D, I = index.search(query_embedding, k)

    results = []
    for idx in I[0]:
        meta = metadata[idx]
        chunk_text = chunks[idx]["text"]

        result = meta.copy()
        result["text"] = chunk_text
        result["heading"] = build_heading(meta)
        results.append(result)

    return results

# Example usage
if __name__ == "__main__":
    query = input("Ask a question: ")
    top_chunks = retrieve_top_k(query, k=5)

    print("\n🔍 Top Retrieved Chunks:\n")
    for i, chunk in enumerate(top_chunks):
        print(f"--- Chunk {i+1} ---")
        print(f"Title        : {chunk.get('paper_title')}")
        print(f"Heading      : {chunk.get('heading') or 'N/A'}")
        print(f"Authors      : {', '.join(chunk.get('authors', []))}")
        print(f"Organization : {chunk.get('organization', 'N/A')}")
        print(f"year : {chunk.get('year', 'N/A')}")
        print(f"Text         : {chunk['text'][:500]}{'...' if len(chunk['text']) > 500 else ''}")
        print()
