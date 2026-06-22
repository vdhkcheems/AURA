# AURA

**Artificial Understanding of Research Articles** is an open-source research
paper companion for people learning machine learning. Pick a supported paper,
ask a question in plain language, follow up naturally, and inspect the exact
paper passages AURA retrieved before it answered.

**Try AURA:** [aura-aa.vercel.app](https://aura-aa.vercel.app/)

The live product is a Next.js application backed by Gemini and Qdrant. Its
curated machine-learning library currently contains ten LaTeX-source papers and
451 section-aware text chunks, including equations and source provenance.

## What AURA does

- Explains machine-learning papers conversationally, including follow-up
  questions within a chat.
- Retrieves evidence from the curated paper library before every answer rather
  than answering from general model knowledge.
- Shows the retrieved paper sections behind each answer.
- Renders Markdown and LaTeX mathematics in the chat.
- Stores guest chat history locally in the visitor's browser.
- Streams answer text as Gemma generates it, after retrieval has completed.

## Run locally

### Prerequisites

- Node.js 20 or newer
- A Gemini API key with access to `gemini-embedding-001` and
  `gemma-4-31b-it`
- A Qdrant Cloud cluster containing the AURA collection (or an equivalent
  collection built with the indexing service below)

### 1. Configure the web app

```bash
cd apps/web
cp .env.example .env.local
```

Fill in `apps/web/.env.local`:

```env
GEMINI_API_KEY=your_gemini_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION=aura_text_ml_core_v1_gemini_embedding_001_v1
```

These values are server-only. Never commit `.env.local` or expose them with a
`NEXT_PUBLIC_` prefix.

### 2. Start AURA

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The introduction is at
`/`; the paper chat workspace is at `/chat`.

### 3. Verify the connection (optional)

```bash
curl http://localhost:3000/api/health
```

The response includes the Qdrant collection name and indexed point count, but
never exposes credentials.

## Architecture

```text
Next.js guest workspace (/ and /chat)
        │
        ▼
POST /api/chat — embed question → retrieve Qdrant evidence → stream answer
        │                           Gemini embedding        Gemma 4 31B
        ▼
Qdrant Cloud — section-aware paper chunks and provenance

Python RAG service — offline source acquisition, LaTeX normalization,
                     chunking, embedding, Qdrant indexing, and evaluation
```

The web app uses `gemini-embedding-001` for query embeddings and
`gemma-4-31b-it` for grounded answers. Qdrant stores 768-dimensional vectors,
paper metadata, section paths, source-file provenance, and text chunks.

## Rebuild or extend the paper index

The Python service in [`services/rag`](services/rag/) is an offline ingestion
tool, not a process required to run the existing hosted index. Use it when
rebuilding the collection or adding papers.

```bash
pip install -r services/rag/requirements.txt
python3 -m services.rag.aura_rag.manifest data/manifests/ml-core-v1.json
python3 -m unittest discover -s services/rag/tests -v
python3 -m services.rag.aura_rag.index_qdrant --dry-run
```

Set `GEMINI_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY` in the root `.env`
before running a real index upsert. Full source-processing and indexing details
are in [`services/rag/README.md`](services/rag/README.md) and
[`docs/embedding_and_indexing.md`](docs/embedding_and_indexing.md).

## Work in progress

1. **Figure and image support:** extract figures from source papers, store them
   in object storage, retrieve their visual evidence, and display them beside
   answers.
2. **Account-based authentication:** let users sign in, sync chat history
   across devices, and optionally import guest conversations.

## Repository layout

```text
apps/web/          Next.js interface and server-side chat API
services/rag/      Offline corpus processing, indexing, and evaluation tools
data/manifests/    Versioned curated-paper manifest
docs/              Chunking and hosted-index contracts
```

## Contributing

Issues, paper suggestions, and pull requests are welcome. Please keep API keys
and generated corpus artifacts out of Git, and run the relevant web or Python
checks before opening a change.
