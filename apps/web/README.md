# AURA Web App

The AURA web app is a Next.js App Router project. Its first implementation is a
server-side, paper-grounded chat API over the live Gemini/Qdrant text index.

## Local setup

Copy `.env.example` to `.env.local` in this directory, then provide the same
server-only values used for corpus indexing. Do not use `NEXT_PUBLIC_` for any
of these variables and do not commit `.env.local`.

```bash
cd apps/web
npm install
npm run dev
```

## API

### `POST /api/chat`

Request:

```json
{
  "question": "How does self-attention work?",
  "paperId": "attention-is-all-you-need",
  "topic": "transformers"
}
```

`paperId` and `topic` are optional Qdrant payload filters. The route embeds the
question with Gemini, retrieves the top five chunks from Qdrant, and uses the
Gemini API's Gemma 4 31B instruction model to answer only from that evidence. Its response includes the answer,
retrieval scores, complete source records, cited-paper IDs, model name, and
warnings.

### `GET /api/health`

Checks server configuration and Qdrant collection availability. It returns the
collection name and indexed point count but never exposes secrets.

## Deployment

Configure these server-side environment variables in Vercel:

- `GEMINI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION` (defaults to `aura_text_ml_core_v1_gemini_embedding_001_v1`)

Before making the chat public, add durable server-side rate limiting. Gemini's
free tier is shared infrastructure and the API must not be left open to anonymous
quota exhaustion.
