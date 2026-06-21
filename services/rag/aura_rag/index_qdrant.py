"""Index stable AURA chunks into a versioned Gemini/Qdrant collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Sequence

from .embeddings import DEFAULT_VECTOR_SIZE, EmbeddingDocument, Embedder, EmbeddingError, GeminiEmbedder
from .qdrant_store import QdrantVectorStore, VectorPoint, VectorStoreError


INDEX_SCHEMA_VERSION = "1"
DEFAULT_COLLECTION = "aura_text_ml_core_v1_gemini_embedding_001_v1"
POINT_NAMESPACE = uuid.UUID("ebc3d08a-85c0-4c47-a4df-47ad4d8e4c15")


class IndexingError(RuntimeError):
    """Raised when chunks cannot safely be embedded or indexed."""


class IndexStore(Protocol):
    vector_size: int

    def ensure_collection(self, collection_name: str) -> None: ...

    def ensure_payload_indexes(self, collection_name: str) -> None: ...

    def upsert(self, collection_name: str, points: Sequence[VectorPoint]) -> None: ...

    def count(self, collection_name: str) -> int: ...


@dataclass(frozen=True)
class IndexingSummary:
    collection_name: str
    corpus_id: str
    chunk_count: int
    vector_size: int
    model_name: str
    chunk_sha256: str


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    chunk_path = Path(path)
    try:
        chunks = [json.loads(line) for line in chunk_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexingError(f"Could not read chunks from {chunk_path}: {exc}") from exc
    _validate_chunks(chunks)
    return chunks


def index_chunks(
    chunks: Sequence[dict[str, Any]],
    *,
    collection_name: str,
    embedder: Embedder,
    store: IndexStore,
    chunk_bytes: bytes,
) -> IndexingSummary:
    """Embed and idempotently upsert one immutable chunk corpus."""
    _validate_chunks(chunks)
    if embedder.vector_size != store.vector_size:
        raise IndexingError("embedder and vector store dimensions must match")
    documents = [_embedding_document(chunk) for chunk in chunks]
    vectors = embedder.embed_documents(documents)
    if len(vectors) != len(chunks):
        raise IndexingError(f"Embedding provider returned {len(vectors)} vectors for {len(chunks)} chunks")
    if any(len(vector) != embedder.vector_size for vector in vectors):
        raise IndexingError("Embedding provider returned a vector with the wrong dimension")

    corpus_ids = {chunk["corpus_id"] for chunk in chunks}
    if len(corpus_ids) != 1:
        raise IndexingError("An index run must contain exactly one corpus_id")
    corpus_id = next(iter(corpus_ids))
    store.ensure_collection(collection_name)
    store.ensure_payload_indexes(collection_name)
    points = [
        VectorPoint(id=_point_id(chunk), vector=vector, payload=_payload(chunk))
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    store.upsert(collection_name, points)
    stored_count = store.count(collection_name)
    if stored_count != len(chunks):
        raise IndexingError(
            f"Collection count is {stored_count}, expected exactly {len(chunks)}. "
            "Use a fresh versioned collection if it contains another corpus."
        )
    return IndexingSummary(
        collection_name=collection_name,
        corpus_id=corpus_id,
        chunk_count=len(chunks),
        vector_size=embedder.vector_size,
        model_name=embedder.model_name,
        chunk_sha256=hashlib.sha256(chunk_bytes).hexdigest(),
    )


def write_index_manifest(path: str | Path, summary: IndexingSummary) -> None:
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "indexed_at": datetime.now(UTC).isoformat(),
        **asdict(summary),
    }
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _embedding_document(chunk: dict[str, Any]) -> EmbeddingDocument:
    section = " > ".join(chunk["section_path"])
    text = f"Paper: {chunk['title']}\nSection: {section}\n\n{chunk['text']}"
    return EmbeddingDocument(text=text, title=chunk["title"])


def _point_id(chunk: dict[str, Any]) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{chunk['corpus_id']}:{chunk['chunk_id']}"))


def _payload(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": chunk["schema_version"],
        "corpus_id": chunk["corpus_id"],
        "paper_id": chunk["paper_id"],
        "chunk_id": chunk["chunk_id"],
        "title": chunk["title"],
        "authors": chunk["authors"],
        "year": chunk["year"],
        "topics": chunk["topics"],
        "section_id": chunk["section_id"],
        "section_path": chunk["section_path"],
        "block_range": chunk["block_range"],
        "text": chunk["text"],
        "character_count": chunk["character_count"],
        "source_files": chunk["source_files"],
        "related_figure_ids": chunk["related_figure_ids"],
        "is_appendix": "Appendix" in chunk["section_path"],
    }


def _validate_chunks(chunks: Sequence[dict[str, Any]]) -> None:
    if not chunks:
        raise IndexingError("No chunks supplied for indexing")
    required = {
        "schema_version", "corpus_id", "paper_id", "chunk_id", "title", "authors", "year", "topics",
        "section_id", "section_path", "block_range", "text", "character_count", "source_files",
        "related_figure_ids",
    }
    chunk_ids: set[str] = set()
    for chunk in chunks:
        missing = required - chunk.keys()
        if missing:
            raise IndexingError(f"Chunk is missing fields: {sorted(missing)}")
        if not isinstance(chunk["chunk_id"], str) or not chunk["chunk_id"] or chunk["chunk_id"] in chunk_ids:
            raise IndexingError(f"Chunk IDs must be unique and non-empty: {chunk.get('chunk_id')!r}")
        chunk_ids.add(chunk["chunk_id"])
        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            raise IndexingError(f"Chunk text is empty: {chunk['chunk_id']}")
        if chunk["character_count"] != len(chunk["text"]):
            raise IndexingError(f"Incorrect character count: {chunk['chunk_id']}")
        if not chunk["section_path"] or not chunk["source_files"]:
            raise IndexingError(f"Chunk provenance is incomplete: {chunk['chunk_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed AURA chunks with Gemini and upsert them into Qdrant.")
    parser.add_argument("chunks_path", type=Path, nargs="?", default=Path("data/processed/ml-core-v1/chunks.jsonl"))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--index-manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--batch-interval-seconds", type=float, default=10.0)
    parser.add_argument("--rate-limit-retries", type=int, default=10)
    parser.add_argument("--rate-limit-backoff-seconds", type=float, default=65.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        chunk_bytes = args.chunks_path.read_bytes()
        chunks = load_chunks(args.chunks_path)
        if args.dry_run:
            print(f"VALID {len(chunks)} chunks for collection {args.collection}.")
            return 0
        from dotenv import load_dotenv
        load_dotenv()
        embedder = GeminiEmbedder(
            os.getenv("GEMINI_API_KEY", ""),
            vector_size=DEFAULT_VECTOR_SIZE,
            batch_size=args.batch_size,
            minimum_batch_interval_seconds=args.batch_interval_seconds,
            rate_limit_retries=args.rate_limit_retries,
            rate_limit_backoff_seconds=args.rate_limit_backoff_seconds,
        )
        store = QdrantVectorStore(
            os.getenv("QDRANT_URL", ""), os.getenv("QDRANT_API_KEY"), vector_size=DEFAULT_VECTOR_SIZE
        )
        summary = index_chunks(
            chunks, collection_name=args.collection, embedder=embedder, store=store, chunk_bytes=chunk_bytes
        )
        manifest_path = args.index_manifest or args.chunks_path.parent / "index-manifest.json"
        write_index_manifest(manifest_path, summary)
    except (IndexingError, EmbeddingError, VectorStoreError, OSError) as exc:
        print(exc)
        return 1
    print(f"INDEXED {summary.chunk_count} chunks into {summary.collection_name}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
