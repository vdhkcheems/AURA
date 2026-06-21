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

## Gemini and Qdrant indexing

The text index uses Gemini embeddings and Qdrant Cloud. The current deployed
index is `aura_text_ml_core_v1_gemini_embedding_001_v1`: 451 chunks, Gemini
`gemini-embedding-001` at 768 dimensions, cosine similarity, and full chunk
provenance in the Qdrant payload. The shipped 21-case paper-level smoke test
retrieved every expected paper in the top five results.

Install the service-only dependencies, validate the existing generated chunks,
then run an idempotent index upsert after setting `GEMINI_API_KEY`, `QDRANT_URL`,
and `QDRANT_API_KEY` in the root `.env` file:

```bash
pip install -r services/rag/requirements.txt
python3 -m services.rag.aura_rag.index_qdrant --dry-run
python3 -m services.rag.aura_rag.index_qdrant
python3 -m services.rag.aura_rag.retrieve_qdrant "How does self-attention work?"
python3 -m services.rag.aura_rag.evaluate_retrieval
```

The indexer batches six chunks and backs off automatically after Gemini `429`
responses, which keeps initial indexing compatible with the free-tier rolling
token quota. Re-running it upserts the same stable point IDs safely.

See [the indexing guide](../../docs/embedding_and_indexing.md) for Qdrant setup,
collection conventions, filters, and deployment boundaries.

## Manifest validation

Validate the curated corpus manifest before using it for ingestion:

```bash
python3 -m services.rag.aura_rag.manifest data/manifests/ml-core-v1.json
```

## Source acquisition

Inspect the arXiv source downloads without writing files:

```bash
python3 -m services.rag.aura_rag.acquire_sources data/manifests/ml-core-v1.json --dry-run
```

Download the planned papers after confirming the list:

```bash
python3 -m services.rag.aura_rag.acquire_sources data/manifests/ml-core-v1.json
```

Archives and acquisition records are stored under `data/raw/<corpus-id>/<paper-id>/` and are intentionally ignored by Git. The command validates that each downloaded archive is readable and includes at least one `.tex` file. Use `--include-legacy` to fetch the original prototype paper as well.

## Source inspection

Inspect each acquired archive, resolve its LaTeX include tree, and write JSON reports:

```bash
python3 -m services.rag.aura_rag.inspect_sources data/manifests/ml-core-v1.json
```

Each report is written to `data/processed/<corpus-id>/<paper-id>/inspection.json`. It identifies the root document, resolved source files, missing or cyclic includes, section counts, and PDF-wrapper sources that require a later fallback. These generated reports are intentionally ignored by Git.

## Source normalization

Normalize the extractable LaTeX papers into section-aware JSON documents:

```bash
python3 -m services.rag.aura_rag.normalize_sources data/manifests/ml-core-v1.json
```

Each output is written to `data/processed/<corpus-id>/<paper-id>/normalized.json`. The normalizer preserves section hierarchy, prose, display equations, figure/table captions, and source-file provenance. It skips PDF-wrapper papers such as Adam and reports them as pending a future PDF fallback.

## Chunk generation

Create section-aware retrieval chunks from every normalized document available in the manifest:

```bash
python3 -m services.rag.aura_rag.chunk_sources data/manifests/ml-core-v1.json
```

The command writes `data/processed/<corpus-id>/chunks.jsonl`. Chunks retain section paths, source files, block ranges, equations, captions, and a reserved `related_figure_ids` field for the future figure evidence layer.
