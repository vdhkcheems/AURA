"""Retrieve source-ready AURA chunks from the Gemini/Qdrant text index."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .embeddings import DEFAULT_VECTOR_SIZE, Embedder, EmbeddingError, GeminiEmbedder
from .index_qdrant import DEFAULT_COLLECTION
from .qdrant_store import QdrantVectorStore, SearchResult, VectorStoreError


class RetrievalError(RuntimeError):
    """Raised when an AURA retrieval request is invalid or incomplete."""


class RetrievalStore(Protocol):
    def search(
        self, collection_name: str, vector: list[float], *, limit: int,
        paper_id: str | None = None, topic: str | None = None,
    ) -> list[SearchResult]: ...


@dataclass(frozen=True)
class RetrievedChunk:
    score: float
    chunk_id: str
    paper_id: str
    title: str
    section_path: list[str]
    text: str
    source_files: list[str]
    related_figure_ids: list[str]


def retrieve(
    question: str,
    *,
    collection_name: str,
    embedder: Embedder,
    store: RetrievalStore,
    limit: int = 5,
    paper_id: str | None = None,
    topic: str | None = None,
) -> list[RetrievedChunk]:
    if not question.strip():
        raise RetrievalError("question must not be empty")
    if not 1 <= limit <= 20:
        raise RetrievalError("limit must be between 1 and 20")
    results = store.search(
        collection_name, embedder.embed_question(question), limit=limit, paper_id=paper_id, topic=topic
    )
    return [_to_retrieved_chunk(result) for result in results]


def _to_retrieved_chunk(result: SearchResult) -> RetrievedChunk:
    payload = result.payload
    required = {"chunk_id", "paper_id", "title", "section_path", "text", "source_files", "related_figure_ids"}
    missing = required - payload.keys()
    if missing:
        raise RetrievalError(f"Qdrant result is missing payload fields: {sorted(missing)}")
    return RetrievedChunk(
        score=result.score,
        chunk_id=payload["chunk_id"],
        paper_id=payload["paper_id"],
        title=payload["title"],
        section_path=payload["section_path"],
        text=payload["text"],
        source_files=payload["source_files"],
        related_figure_ids=payload["related_figure_ids"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve relevant AURA chunks from Qdrant.")
    parser.add_argument("question")
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--paper-id")
    parser.add_argument("--topic")
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
        embedder = GeminiEmbedder(os.getenv("GEMINI_API_KEY", ""), vector_size=DEFAULT_VECTOR_SIZE)
        store = QdrantVectorStore(
            os.getenv("QDRANT_URL", ""), os.getenv("QDRANT_API_KEY"), vector_size=DEFAULT_VECTOR_SIZE
        )
        results = retrieve(
            args.question, collection_name=args.collection, embedder=embedder, store=store,
            limit=args.limit, paper_id=args.paper_id, topic=args.topic,
        )
    except (RetrievalError, EmbeddingError, VectorStoreError) as exc:
        print(exc)
        return 1
    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
