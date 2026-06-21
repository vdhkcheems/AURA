# Gemini and Qdrant Indexing v1

## Goal

Turn `data/processed/ml-core-v1/chunks.jsonl` into a hosted text-retrieval index. This is an offline ingestion step; the future web app only embeds a user question, queries Qdrant, and sends retrieved evidence to the answer model.

## Chosen v1 stack

- **Embedding model:** `gemini-embedding-001` at 768 dimensions.
- **Document task type:** `RETRIEVAL_DOCUMENT`, with the paper title supplied to Gemini.
- **Question task type:** `QUESTION_ANSWERING`.
- **Vector database:** Qdrant Cloud, cosine distance.
- **Collection:** `aura_text_ml_core_v1_gemini_embedding_001_v1` by default.

The collection name is deliberately versioned. Changing the corpus, embedding model, vector size, or embedding input format requires a new collection rather than mixing incompatible vectors.

## Payload and filters

Each Qdrant point uses a deterministic UUID derived from immutable corpus and chunk IDs. The payload retains the original `chunk_id`, complete chunk text, paper metadata, section path, block range, source files, and reserved figure links. Qdrant payload indexes cover `paper_id`, `topics`, `year`, `section_path`, and `is_appendix`.

## Local setup

Create a free Qdrant Cloud cluster, then copy `.env.example` to `.env` and set the Qdrant URL/API key alongside the existing Gemini key. Never commit `.env`.

```bash
python3 -m venv .venv-rag
source .venv-rag/bin/activate
pip install -r services/rag/requirements.txt
python3 -m services.rag.aura_rag.index_qdrant --dry-run
python3 -m services.rag.aura_rag.index_qdrant
python3 -m services.rag.aura_rag.retrieve_qdrant "How does self-attention work?"
python3 -m services.rag.aura_rag.evaluate_retrieval
```

The default indexer pacing (`6` chunks every `10` seconds) is deliberately
conservative for Gemini's free tier. It also waits 65 seconds and retries a
rate-limited batch (up to 10 times), which makes the first index robust to the
rolling token quota for equation-heavy chunks. Use `--batch-size`,
`--batch-interval-seconds`, and the rate-limit options only after checking the
quota for the Gemini project that owns the key.

Indexing writes the ignored generated file `data/processed/ml-core-v1/index-manifest.json`, recording the chunk-file hash, collection, model, vector dimension, and timestamp.

`data/evaluations/ml-core-v1-retrieval.jsonl` contains 21 paper-grounded cases, including two cases for each indexed paper and one cross-paper case. The evaluator reports paper-level recall at the selected `--limit` (default: 5); inspect individual misses before changing model, chunking, or ranking behavior.

## Deployment boundary

The future Vercel API route must use `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION` as server-side environment variables. The browser must never receive them. Add server-side rate limiting before opening the public chat because Gemini's free tier is quota-limited.
