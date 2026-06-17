# AURA RAG Service

This directory will hold the Python-side research processing code for AURA.

The planned responsibilities are:

- Download or load curated paper sources.
- Normalize arXiv LaTeX and fallback PDF text.
- Generate section-aware chunks.
- Create embeddings.
- Upsert vectors and metadata into Qdrant.
- Provide retrieval utilities for local validation.

For the first migration stage, this is intentionally separate from the web app. User uploads and live processing jobs are deferred until the curated corpus flow is reliable.

## Manifest validation

Validate the curated corpus manifest before using it for ingestion:

```bash
python3 -m services.rag.aura_rag.manifest data/manifests/ml-core-v1.json
```
