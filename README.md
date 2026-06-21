# AURA

Artificial Understanding of Research Articles is a research-paper question-answering project. It is evolving from a single-paper Streamlit prototype into a deployable, multi-paper RAG application.

## Project Status

The repository currently contains two tracks:

- **Legacy prototype:** a Streamlit application backed by Gemini, Sentence Transformers, and a local FAISS index for one paper, "Attention Is All You Need."
- **Migration in progress:** a curated machine-learning corpus pipeline that will become the backend foundation for a Next.js web application and Qdrant vector search.

The Streamlit app remains available for reference while the new architecture is built in small, verified stages. It is not the intended long-term deployment path.

## Migration Progress

The curated-corpus and text-retrieval foundation is complete:

- A validated manifest defines an initial 11-paper machine-learning corpus.
- All 11 arXiv source archives have been acquired and inspected.
- Ten papers, including *Attention Is All You Need*, have extractable LaTeX source trees with no unresolved or cyclic includes.
- The ten extractable papers have been normalized into section-aware documents that retain prose, equations, captions, and source-file provenance.
- Those documents are split into section-aware retrieval chunks with stable IDs, section paths, block ranges, and reserved figure-linkage fields.
- The chunking and future figure-linkage contract is documented in [docs/chunking_and_figure_contract.md](docs/chunking_and_figure_contract.md).
- All 451 chunks from the ten extractable papers are embedded with Gemini (`gemini-embedding-001`, 768 dimensions) and indexed in the versioned Qdrant collection `aura_text_ml_core_v1_gemini_embedding_001_v1`.
- The initial 21-case paper-level retrieval evaluation achieved 21/21 hits at top 5. It is a smoke-test baseline, not a substitute for a larger relevance evaluation.
- "Adam: A Method for Stochastic Optimization" is a PDF-wrapper source and remains pending a separate PDF fallback path.

The server-side chat/retrieval API is now implemented in the Next.js web app.
The next stage is wiring it to an interactive chat interface, adding durable rate
limiting, and deploying it with server-side Vercel environment variables. Figure
rendering and visual-evidence retrieval remain planned after the text retrieval
foundation is reliable.

The indexing utility is now available for a hosted Gemini/Qdrant text index. See
[docs/embedding_and_indexing.md](docs/embedding_and_indexing.md) for setup and
the required Qdrant Cloud environment variables.

For the detailed roadmap, see [docs/improvement_plan.md](docs/improvement_plan.md).

## Repository Layout

```text
app.py, generation.py, query_rag.py  Legacy Streamlit prototype
data/manifests/                      Versioned corpus definitions
data/raw/                            Downloaded source archives (generated, ignored)
data/processed/                      Inspection, normalized, and chunk outputs (generated, ignored)
services/rag/                        Python corpus processing and retrieval code
apps/web/                            Next.js app and server-side chat/health API routes
packages/shared/                     Reserved for shared TypeScript contracts
infra/                               Deployment and infrastructure configuration
```

## Build the Curated Corpus From Scratch

The corpus pipeline downloads arXiv source archives, inspects their LaTeX trees,
normalizes extractable documents, and generates retrieval chunks. It uses only
the Python standard library; a virtual environment and `pip install` are not
required for these commands.

### 1. Clone the repository

```bash
git clone git@github.com:vdhkcheems/AURA.git
cd AURA
```

Use an HTTPS clone URL instead if that is how your GitHub credentials are
configured.

### 2. Choose the corpus mode

The manifest contains 11 papers:

- **Default mode** processes the 10 papers marked `planned`.
- **Include legacy mode** also processes *Attention Is All You Need*, the paper
  retained from the original Streamlit prototype.
- **Adam** is selected in either mode, but is intentionally skipped during
  normalization because its arXiv source is a PDF wrapper. Its PDF fallback is
  not implemented yet.

For a complete 10-paper LaTeX corpus, use `--include-legacy` consistently for
acquisition, inspection, and normalization. Chunking automatically reads every
normalized document that exists.

### 3. Validate and preview downloads

Run these from the repository root:

```bash
python3 -m services.rag.aura_rag.manifest data/manifests/ml-core-v1.json
python3 -m services.rag.aura_rag.acquire_sources data/manifests/ml-core-v1.json --dry-run --include-legacy
```

The preview makes no network requests or file changes.

### 4. Acquire source archives

```bash
python3 -m services.rag.aura_rag.acquire_sources data/manifests/ml-core-v1.json --include-legacy
```

This downloads arXiv source archives to `data/raw/ml-core-v1/<paper-id>/` and
records acquisition metadata beside each archive.

To omit the legacy paper, remove `--include-legacy` from this and the next two
commands.

### 5. Inspect the LaTeX source trees

```bash
python3 -m services.rag.aura_rag.inspect_sources data/manifests/ml-core-v1.json --include-legacy
```

This resolves `\input` and `\include` trees and writes
`data/processed/ml-core-v1/<paper-id>/inspection.json`. Review these reports if
a paper is marked for PDF fallback or has unresolved includes.

### 6. Normalize the extractable papers

```bash
python3 -m services.rag.aura_rag.normalize_sources data/manifests/ml-core-v1.json --include-legacy
```

This writes `normalized.json` for each extractable paper. The normalizer retains
section paths, prose, inline and display LaTeX math, captions, and source-file
provenance. Adam is reported as skipped until PDF fallback support exists.

### 7. Generate retrieval chunks

```bash
python3 -m services.rag.aura_rag.chunk_sources data/manifests/ml-core-v1.json
```

This writes the combined corpus file at:

```text
data/processed/ml-core-v1/chunks.jsonl
```

Each chunk has a stable ID, paper and section metadata, block range, source-file
provenance, text, and character count. See
[docs/chunking_and_figure_contract.md](docs/chunking_and_figure_contract.md) for
the processed-data contract.

### 8. Verify the pipeline

```bash
python3 -m unittest discover -s services/rag/tests -v
python3 -m services.rag.aura_rag.chunk_sources data/manifests/ml-core-v1.json --dry-run
```

The test suite validates manifest, acquisition, inspection, normalization, and
chunking behavior. The dry run reports per-paper chunk counts without rewriting
`chunks.jsonl`.

### Start over with a clean generated corpus

The raw archives and processed artifacts are generated and ignored by Git. To
delete only this corpus version and rebuild it, run:

```bash
rm -rf data/raw/ml-core-v1 data/processed/ml-core-v1
```

Then repeat steps 3–7. This does not delete source code, the manifest, or the
tracked `.gitkeep` placeholders in `data/raw/` and `data/processed/`.

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
