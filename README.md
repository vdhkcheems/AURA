# AURA

Artificial Understanding of Research Articles is a research-paper question-answering project. It is evolving from a single-paper Streamlit prototype into a deployable, multi-paper RAG application.

## Project Status

The repository currently contains two tracks:

- **Legacy prototype:** a Streamlit application backed by Gemini, Sentence Transformers, and a local FAISS index for one paper, "Attention Is All You Need."
- **Migration in progress:** a curated machine-learning corpus pipeline that will become the backend foundation for a Next.js web application and Qdrant vector search.

The Streamlit app remains available for reference while the new architecture is built in small, verified stages. It is not the intended long-term deployment path.

## Migration Progress

The curated-corpus foundation is complete and ready for vector indexing:

- A validated manifest defines an initial 11-paper machine-learning corpus.
- All 11 arXiv source archives have been acquired and inspected.
- Ten papers, including *Attention Is All You Need*, have extractable LaTeX source trees with no unresolved or cyclic includes.
- The ten extractable papers have been normalized into section-aware documents that retain prose, equations, captions, and source-file provenance.
- Those documents have been split into 440 section-aware retrieval chunks with stable IDs, section paths, block ranges, and reserved figure-linkage fields.
- The chunking and future figure-linkage contract is documented in [docs/chunking_and_figure_contract.md](docs/chunking_and_figure_contract.md).
- "Adam: A Method for Stochastic Optimization" is a PDF-wrapper source and remains pending a separate PDF fallback path.

The next stage is embedding the chunks, indexing them in local Qdrant, and validating retrieval quality before building the web/API layer. Figure rendering and visual-evidence retrieval are planned after the text retrieval foundation is reliable.

For the detailed roadmap, see [docs/improvement_plan.md](docs/improvement_plan.md).

## Repository Layout

```text
app.py, generation.py, query_rag.py  Legacy Streamlit prototype
data/manifests/                      Versioned corpus definitions
data/raw/                            Downloaded source archives (generated, ignored)
data/processed/                      Inspection, normalized, and chunk outputs (generated, ignored)
services/rag/                        Python corpus processing and retrieval code
apps/web/                            Reserved for the Next.js application
packages/shared/                     Reserved for shared TypeScript contracts
infra/                               Deployment and infrastructure configuration
```

## Working With The Corpus

The manifest, acquisition, inspection, normalization, and chunking tooling uses only the Python standard library at this stage.

Validate the manifest:

```bash
python3 -m services.rag.aura_rag.manifest data/manifests/ml-core-v1.json
```

Preview the planned arXiv source downloads:

```bash
python3 -m services.rag.aura_rag.acquire_sources data/manifests/ml-core-v1.json --dry-run
```

Acquire the complete source corpus, including the legacy prototype paper:

```bash
python3 -m services.rag.aura_rag.acquire_sources data/manifests/ml-core-v1.json --include-legacy
```

Inspect the acquired LaTeX document trees:

```bash
python3 -m services.rag.aura_rag.inspect_sources data/manifests/ml-core-v1.json --include-legacy
```

Normalize the extractable LaTeX papers:

```bash
python3 -m services.rag.aura_rag.normalize_sources data/manifests/ml-core-v1.json --include-legacy
```

Generate section-aware retrieval chunks:

```bash
python3 -m services.rag.aura_rag.chunk_sources data/manifests/ml-core-v1.json
```

Inspection reports are written to `data/processed/ml-core-v1/<paper-id>/inspection.json`, normalized documents to `data/processed/ml-core-v1/<paper-id>/normalized.json`, and the corpus chunk file to `data/processed/ml-core-v1/chunks.jsonl`. Source archives and generated outputs are intentionally not committed to Git.

## Legacy Streamlit Prototype

The existing prototype requires a Gemini API key and the Python dependencies listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `GEMINI_API_KEY` in `.env`, then run:

```bash
streamlit run app.py
```

This path is legacy and may require a Python version supported by its native ML dependencies. It is separate from the new corpus pipeline.

## Target Architecture

The intended production system separates responsibilities cleanly:

- A Next.js frontend deployed on Vercel.
- An API layer for chat and retrieval requests.
- Python services for offline source processing, normalization, chunking, and indexing.
- Qdrant for persistent vector storage and metadata filtering.

User paper uploads and asynchronous processing jobs are planned after the curated corpus ingestion flow is reliable.

## Contributing

Keep generated corpus data out of version control, validate the manifest before ingestion, and add focused tests for new processing behavior. The migration plan is deliberately incremental so each corpus stage can be inspected before the next one begins.
