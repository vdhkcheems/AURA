# AURA Web App

The AURA web app is a Next.js App Router project with a guest-first,
paper-grounded chat workspace over the live Gemini/Qdrant text index.
`/` is the product introduction; `/chat` opens the workspace.

## Local setup

Copy `.env.example` to `.env.local` in this directory, then provide the same
server-only values used for corpus indexing. Do not use `NEXT_PUBLIC_` for any
of these variables and do not commit `.env.local`.

```bash
cd apps/web
npm install
npm run dev
```

## Guest workspace

The `/chat` workspace is usable without an account. It lets a visitor create, rename,
and delete chats; browse the supported-paper library; scope a chat to one paper;
and ask follow-up questions. Chat titles, messages, selected paper scopes, and
retrieved sources are stored only in that browser under `aura.guest-chats.v1`.

The sign-in button is intentionally a visual placeholder for a future
account-backed history and sync feature. It does not authenticate a user or send
guest conversation content anywhere except the chat API when a question is
submitted.

## API

### `POST /api/chat`

Request:

```json
{
  "question": "How does self-attention work?",
  "paperId": "attention-is-all-you-need",
  "topic": "transformers",
  "history": [
    { "role": "user", "content": "What is a residual connection?" },
    { "role": "assistant", "content": "It adds an earlier representation..." }
  ]
}
```

`paperId` and `topic` are optional Qdrant payload filters. The route embeds the
question with Gemini, retrieves the top five chunks from Qdrant, then streams
Gemma 4 31B's grounded answer as newline-delimited JSON. It first emits `meta`
with source records, model, and warnings; then `delta` events with answer text;
and finally `done`. `history` is optional and accepts at most 12 short prior
turns; it is used only to make a single answer conversational and is not
persisted by the server.

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
