# AURA Improvement Plan

## Target

Turn AURA from a Streamlit-based proof of concept into a full-fledged, deployable research-paper understanding web app with a larger curated corpus of machine learning papers.

The near-term focus is not user uploads or a fully automated processing pipeline. The first target is:

- Replace the current Streamlit app with a production-shaped web architecture.
- Separate frontend, backend, and Python ML/RAG responsibilities.
- Expand the supported research corpus beyond the current single paper.
- Use a deployable vector database setup, likely Qdrant.
- Build the foundation for a future ingestion pipeline and user-upload workflow.

## Guiding Principles

1. Keep the product usable after each phase.
2. Avoid mixing app UI code, API code, and ML processing logic.
3. Treat paper ingestion as a first-class system, even if the first version is manually curated.
4. Prefer structured arXiv LaTeX sources where possible because they are easier to chunk than PDFs.
5. Make deployment decisions early enough that local development resembles production.
6. Defer user-upload complexity until the curated corpus flow is reliable.

## Proposed Architecture

### Frontend

- Next.js app.
- TypeScript.
- Chat interface for asking questions about the research corpus.
- Paper/library browsing interface.
- Source display for retrieved chunks.
- Deployed on Vercel.

### Backend API

- API layer responsible for chat requests, retrieval requests, paper metadata, and routing.
- Can start as Next.js API routes or a separate backend service.
- Should hide model provider keys from the frontend.
- Should expose clean contracts such as:
  - `POST /api/chat`
  - `GET /api/papers`
  - `GET /api/papers/:id`
  - `GET /api/health`

### Python ML/RAG Layer

- Python remains responsible for:
  - Paper parsing and normalization.
  - Chunk generation.
  - Embedding generation.
  - Index creation or vector database upserts.
  - Retrieval utilities, if not moved behind an API service.
- This should become a separate package or service rather than being tightly coupled to the web app.

### Vector Database

- Use Qdrant for vector storage and similarity search.
- Local development can use Docker.
- Production can use Qdrant Cloud or a managed Qdrant instance.
- Store vectors with useful payload metadata:
  - Paper title.
  - Authors.
  - arXiv ID.
  - Year.
  - Section hierarchy.
  - Chunk text.
  - Chunk index.
  - Source type.
  - LaTeX source location when available.

### LLM Provider

- Keep Gemini support initially if it is already working.
- Design the backend so the model provider can be swapped later.
- Avoid putting provider-specific calls deep inside UI code.

## Phase 1: Stabilize and Document the Current System

Goal: Make the current project easier to understand before migrating.

Steps:

1. Document the existing app behavior.
   - Current Streamlit UI.
   - Current RAG flow.
   - Current one-paper corpus.
   - Current embedding model.
   - Current Gemini dependency.

2. Identify current responsibilities by file.
   - `app.py`: UI and user session state.
   - `generation.py`: Gemini setup, routing, answer generation.
   - `query_rag.py`: FAISS retrieval.
   - `embedding.py`: one-off embedding/index generation.
   - `utils.py`: prompt helpers and response formatting.
   - `data/`: local chunks, metadata, and FAISS index.

3. Define the migration boundary.
   - UI will move to Next.js.
   - API and orchestration logic will move out of Streamlit.
   - Python will remain responsible for ingestion, chunking, embeddings, and possibly retrieval.

4. Create a baseline test query set.
   - Add a small list of representative questions for the current "Attention Is All You Need" corpus.
   - Include factual, conceptual, equation-related, and out-of-scope questions.
   - Use this later to compare old and new behavior.

Deliverable:

- A documented understanding of the current system.
- A small evaluation set for regression checks.

## Phase 2: Design the New Repository Structure

Goal: Create a clean structure that separates web, API, and ML processing.

Recommended structure:

```text
AURA/
  apps/
    web/
      Next.js frontend and API routes
  packages/
    shared/
      Shared TypeScript types and schemas
  services/
    rag/
      Python ingestion, chunking, embedding, and indexing code
  data/
    raw/
      Downloaded paper sources
    processed/
      Normalized intermediate files
    manifests/
      Paper metadata and corpus definitions
  docs/
    improvement_plan.md
  infra/
    docker-compose.yml
    deployment notes
```

Steps:

1. Decide whether the first backend lives inside Next.js API routes or as a separate service.
   - Simple option: Next.js API routes call Qdrant and the LLM directly.
   - More scalable option: separate backend service owns retrieval and generation.

2. Define environment variables.
   - `GEMINI_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `QDRANT_COLLECTION`
   - `EMBEDDING_MODEL`

3. Define shared response types.
   - Chat request.
   - Chat response.
   - Retrieved source.
   - Paper metadata.
   - Error response.

4. Keep legacy Streamlit code temporarily.
   - Do not delete it until the Next.js app reaches functional parity.
   - Use it as a behavior reference.

Deliverable:

- A new architecture structure ready for implementation.
- Clear boundaries between frontend, API, and Python RAG code.

## Phase 3: Build the Curated ML Paper Corpus

Goal: Expand from one paper to a useful curated ML/AI corpus.

Initial paper categories:

1. Transformers and attention.
   - "Attention Is All You Need"
   - BERT
   - GPT-style language model papers
   - Transformer-XL

2. Representation learning.
   - Word2Vec
   - GloVe
   - SimCLR
   - CLIP

3. Diffusion and generative models.
   - DDPM
   - Latent Diffusion Models
   - Score-based generative modeling

4. Reinforcement learning.
   - DQN
   - PPO
   - AlphaGo or AlphaZero papers

5. Optimization and training.
   - Adam
   - Batch Normalization
   - Layer Normalization
   - Dropout

6. Retrieval and RAG.
   - Dense Passage Retrieval
   - REALM
   - RAG
   - ColBERT

Steps:

1. Create a corpus manifest.
   - Each paper should have a stable ID.
   - Include title, authors, year, arXiv ID, source URL, topic tags, and license notes where relevant.

2. Prefer arXiv source downloads when available.
   - Use LaTeX source for section-aware parsing.
   - Fall back to PDF extraction only when source is unavailable or too messy.

3. Start with a small but meaningful batch.
   - Target 10 to 20 papers first.
   - Validate quality before scaling to 50+ papers.

4. Keep raw and processed artifacts separate.
   - Raw source stays unchanged.
   - Processed text/chunks are generated outputs.

5. Add corpus versioning.
   - Example: `ml-core-v1`.
   - This helps track which papers and chunks were used for a deployed index.

Deliverable:

- A curated ML corpus manifest.
- Raw paper sources or source references.
- A clear first corpus version.

## Phase 4: Build a Better Chunking Strategy

Goal: Replace simple chunking with section-aware, citation-aware, equation-aware chunking.

Chunking priorities:

1. Preserve paper structure.
   - Title.
   - Abstract.
   - Section.
   - Subsection.
   - Subsubsection.
   - Appendix.

2. Preserve useful technical context.
   - Equations should stay near their explanations.
   - Figure and table captions should be retained when meaningful.
   - Definitions and notation should not be split aggressively.

3. Keep chunks retrieval-friendly.
   - Avoid very tiny fragments.
   - Avoid huge sections that dilute embedding quality.
   - Use overlap where needed.

4. Attach strong metadata to every chunk.
   - Paper ID.
   - Chunk ID.
   - Section heading.
   - Section path.
   - Token or character count.
   - Source line or approximate location, when available.
   - Related figure IDs, when a chunk contains, cites, or immediately explains a figure.

5. Preserve figure relationships during normalization and chunking.
   - Capture figure labels, captions, source files, and `\includegraphics` asset references.
   - Keep original assets immutable under raw source storage.
   - Reserve stable figure IDs that can later connect chunks, rendered assets, and figure retrieval results.
   - Do not discard a figure relationship merely because the first text-only RAG release does not render images.

Recommended first strategy:

1. Parse LaTeX into a normalized document tree.
2. Split by section hierarchy.
3. Within large sections, split by paragraphs while preserving equations.
4. Add overlap between neighboring chunks.
5. Store chunks as JSONL before embedding.

Example chunk schema:

```json
{
  "paper_id": "attention-is-all-you-need",
  "chunk_id": "attention-is-all-you-need::3.2::0004",
  "title": "Attention Is All You Need",
  "authors": ["Ashish Vaswani", "Noam Shazeer"],
  "year": 2017,
  "section_path": ["Model Architecture", "Attention"],
  "text": "Chunk text here...",
  "source": {
    "type": "arxiv-latex",
    "arxiv_id": "1706.03762"
  }
}
```

Deliverable:

- A reusable chunk format.
- A first version of the section-aware chunking approach.
- Chunks generated for the expanded corpus.

## Phase 4.5: Add a Figure Evidence Layer

Goal: Make diagrams, plots, qualitative results, and other useful paper figures available as source-grounded visual evidence without mixing image bytes into the text RAG index.

Approach:

1. Extract figure metadata from LaTeX.
   - Figure label, caption, paper ID, section path, source file, and nearby text references.
   - Resolve `\includegraphics` paths against the arXiv source archive.
   - Record figures generated from LaTeX/TikZ separately for a later rendering adapter.

2. Produce browser-ready assets.
   - Retain the source asset unchanged.
   - Copy or optimize PNG/JPEG assets.
   - Render PDF/EPS source figures to PNG or WebP.
   - Use a PDF-page extraction fallback only when source assets are unavailable.

3. Generate and index visual descriptions.
   - Build a description from the caption, nearby paper text, and a vision-capable model when available.
   - Index figure records separately from text chunks using title, section path, caption, generated description, and text references.
   - Store asset keys and metadata in the vector database; store image bytes in object storage rather than Qdrant.

4. Retrieve figures as connected evidence.
   - First retrieve text chunks normally.
   - Prefer figures directly linked to those chunks or explicitly referenced by them.
   - Optionally query the figure-description index and rerank candidates.
   - Let the response layer show a figure only when it clears a relevance threshold or the user explicitly asks for one.

5. Display figures in the web app.
   - Show a "Helpful figure" card with the rendered image, caption, paper/section attribution, and a concise explanation.
   - Keep the image optional and expandable so it supports the answer rather than interrupting it.

Deliverable:

- A separate, source-grounded figure record format and asset pipeline.
- Figure metadata linked to text chunks.
- Retrieval and UI support for displaying relevant visual evidence.

## Phase 5: Move from Local FAISS to Qdrant

Goal: Make retrieval deployable and independent of local files.

Steps:

1. Create a Qdrant collection.
   - Pick vector size based on the embedding model.
   - Use cosine similarity unless there is a strong reason not to.

2. Write an indexing script.
   - Read processed chunks.
   - Generate embeddings.
   - Upsert vectors and payloads into Qdrant.

3. Write retrieval code.
   - Embed the user query.
   - Query Qdrant.
   - Return top-k chunks and metadata.
   - Keep text-chunk and figure-description retrieval as separate collections or payload namespaces, joined by stable paper/chunk/figure IDs.

4. Add local development support.
   - Use Docker Compose for local Qdrant.
   - Provide a simple command to index the corpus locally.

5. Add index validation.
   - Confirm collection exists.
   - Confirm vector count matches chunk count.
   - Run sample queries and inspect top results.

Deliverable:

- Qdrant-based retrieval.
- Local vector database setup.
- Indexing script for the curated corpus.

## Phase 6: Build the Next.js Web App

Goal: Replace the Streamlit UI with a real web frontend.

Core screens:

1. Chat screen.
   - Ask questions.
   - Display generated answers.
   - Display cited sources.
   - Display an optional, relevance-ranked "Helpful figure" when visual evidence materially improves the answer.
   - Show whether retrieval was used.

2. Paper library screen.
   - List supported papers.
   - Filter by topic.
   - Open paper details.

3. Paper detail screen.
   - Show title, authors, abstract, year, tags.
   - Show indexed sections.
   - Allow starting a chat scoped to that paper.

4. Basic system status screen or API health route.
   - Useful during deployment and debugging.

Frontend expectations:

- Clean research-tool interface, not a marketing landing page.
- Dense enough for real use.
- Good source readability.
- Clear distinction between answer text and retrieved evidence.
- Responsive layout for desktop and mobile.

Deliverable:

- Next.js app with the core AURA experience.
- Functional chat UI backed by the new API.
- Paper browsing backed by corpus metadata.

## Phase 7: Build the Chat and Retrieval API

Goal: Create a stable API layer between the frontend, vector database, and LLM provider.

Chat flow:

1. Receive user query.
2. Decide whether retrieval is needed.
3. If retrieval is needed:
   - Embed query.
   - Query Qdrant.
   - Build context from retrieved chunks.
   - Retrieve linked or semantically relevant figure candidates when the answer could benefit from visual evidence.
   - Generate grounded response.
4. If retrieval is not needed:
   - Generate normal assistant response or politely steer back to research help.
5. Return answer plus source metadata.

Response should include:

- `answer`
- `mode`
- `sources`
- `paper_ids`
- `retrieval_scores`
- `model`
- `warnings`, if any

Important safeguards:

- Do not expose provider API keys.
- Keep prompts server-side.
- Handle provider failures gracefully.
- Avoid returning unsupported answers as if they came from papers.
- Keep the option to force paper-grounded mode.

Deliverable:

- Stable chat API.
- Source-aware response contract.
- Optional figure evidence contract.
- Frontend connected to real retrieval and generation.

## Phase 8: Deployment Setup

Goal: Make the system easy to deploy and operate.

Recommended deployment shape:

- Vercel for the Next.js app.
- Qdrant Cloud or managed Qdrant for vector search.
- Python ingestion/indexing run as an offline job from a local machine, CI job, or separate worker.
- Secrets managed through Vercel and Qdrant provider settings.

Steps:

1. Add environment variable documentation.
2. Add local development commands.
3. Add production deployment notes.
4. Add a health endpoint.
5. Add basic logging around:
   - Chat requests.
   - Retrieval count.
   - LLM errors.
   - Empty retrieval results.
6. Add basic rate limiting if the app is public.

Deliverable:

- Deployed Next.js app.
- Connected Qdrant index.
- Documented local and production setup.

## Phase 9: Evaluation and Quality Checks

Goal: Make sure increasing the corpus actually improves usefulness instead of creating noisy retrieval.

Steps:

1. Create evaluation questions for each paper.
2. Create cross-paper comparison questions.
3. Create out-of-scope questions.
4. Track whether retrieved chunks are relevant.
5. Track whether the generated answer is grounded in retrieved context.
6. Compare retrieval quality across chunking versions.

Useful metrics:

- Top-k retrieval relevance.
- Citation accuracy.
- Hallucination rate.
- Answer completeness.
- Latency.
- Failed requests.

Deliverable:

- Evaluation set.
- Repeatable manual or automated quality checks.
- Confidence that the expanded corpus is usable.

## Phase 10: Later User Upload Workflow

Goal: Add user-provided papers only after the curated corpus flow is strong.

Future upload flow:

1. User uploads PDF or provides arXiv URL.
2. System creates a processing job.
3. Processing job extracts or downloads source.
4. System chunks paper.
5. System embeds chunks.
6. System stores vectors in a user-specific or document-specific namespace.
7. User can chat with the uploaded paper.

Important future decisions:

- Whether uploaded papers are private by default.
- How long uploaded files and vectors are retained.
- Whether each user gets isolated Qdrant collections or payload filters.
- How to handle failed PDF parsing.
- How to show processing progress.
- How to control cost and abuse.

Deliverable:

- Future design for user uploads, not part of the first migration.

## Suggested Milestones

### Milestone 1: Architecture Foundation

- Finalize repo structure.
- Add environment documentation.
- Define API contracts.
- Preserve legacy Streamlit behavior for reference.

### Milestone 2: Curated Corpus v1

- Choose 10 to 20 ML papers.
- Create corpus manifest.
- Download or reference arXiv LaTeX sources.
- Generate normalized chunks.

### Milestone 3: Qdrant Retrieval

- Run Qdrant locally.
- Index the curated chunks.
- Query Qdrant from a script.
- Validate retrieval quality.

### Milestone 3.5: Figure Evidence Foundation

- Preserve figure metadata and chunk-to-figure links.
- Render directly referenced source assets into web-ready derivatives.
- Build a separate figure-description index.
- Validate that figures shown in answers are relevant and correctly attributed.

### Milestone 4: Next.js App v1

- Build chat UI.
- Build paper library UI.
- Connect chat API to retrieval and generation.
- Show answer sources.

### Milestone 5: Deployment v1

- Deploy web app to Vercel.
- Connect production Qdrant.
- Configure secrets.
- Add health checks and deployment documentation.

### Milestone 6: Quality Pass

- Run evaluation queries.
- Tune chunking and prompts.
- Improve citation/source display.
- Decide whether the system is ready for more papers.

## Immediate Next Steps

1. Decide the first 10 to 20 papers for the curated ML corpus.
2. Decide whether the backend should start inside Next.js or as a separate service.
3. Choose the embedding model for the new Qdrant index.
4. Define the first corpus manifest schema.
5. Create a small evaluation question set from the existing paper.
6. Start the migration only after these decisions are written down.
7. Define a chunking and figure-linkage contract before implementing the chunker.

## Open Decisions

1. Should the first API backend be Next.js-only, or should retrieval/generation live in a separate Python service?
2. Should embeddings continue with `BAAI/bge-base-en-v1.5`, or should the project move to a newer embedding model?
3. Should the curated corpus include only arXiv papers with LaTeX source, or allow important PDF-only papers?
4. Should the first deployed app support global chat across all papers, paper-scoped chat, or both?
5. Should Qdrant payloads store full chunk text, or should chunk text live in separate storage with Qdrant storing references?
6. What is the expected public usage level, and does the first deployment need rate limiting?
7. Which object storage provider should hold rendered figure assets in production?
8. Should vision-generated figure descriptions be produced during offline ingestion or on demand?

## Definition of Success

AURA reaches the target state when:

- The user-facing app runs as a Next.js web app.
- The system is deployable through Vercel and an external vector database.
- The frontend, backend/API, and Python processing logic are clearly separated.
- The corpus includes multiple ML research papers, not just one.
- Retrieval uses Qdrant rather than a local FAISS file loaded inside the app.
- Answers include clear source references.
- The project has documented setup, deployment, and corpus indexing steps.
- The architecture can later support user-uploaded papers without another full rewrite.
