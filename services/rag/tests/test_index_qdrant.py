"""Tests for provider-neutral Gemini/Qdrant indexing orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.rag.aura_rag.index_qdrant import (
    IndexingError,
    _embedding_document,
    _point_id,
    index_chunks,
    load_chunks,
    write_index_manifest,
)
from services.rag.aura_rag.qdrant_store import VectorPoint


def _chunk(chunk_id: str = "example-paper::001::0001") -> dict:
    text = "A retrieval-ready paragraph."
    return {
        "schema_version": "1", "corpus_id": "example-v1", "paper_id": "example-paper",
        "chunk_id": chunk_id, "title": "Example Paper", "authors": ["A. Author"], "year": 2026,
        "topics": ["testing"], "section_id": "001", "section_path": ["Method"],
        "block_range": {"start": 0, "end": 1}, "text": text, "character_count": len(text),
        "source_files": ["main.tex"], "related_figure_ids": [],
    }


class _FakeEmbedder:
    model_name = "gemini-embedding-001"
    vector_size = 3

    def embed_documents(self, documents):
        self.documents = list(documents)
        return [[0.1, 0.2, 0.3] for _ in documents]

    def embed_question(self, _question):
        return [0.1, 0.2, 0.3]


class _FakeStore:
    vector_size = 3

    def __init__(self) -> None:
        self.points: list[VectorPoint] = []
        self.collection_name: str | None = None
        self.payload_indexes_called = False

    def ensure_collection(self, collection_name: str) -> None:
        self.collection_name = collection_name

    def ensure_payload_indexes(self, _collection_name: str) -> None:
        self.payload_indexes_called = True

    def upsert(self, _collection_name: str, points) -> None:
        self.points = list(points)

    def count(self, _collection_name: str) -> int:
        return len(self.points)


class IndexQdrantTests(unittest.TestCase):
    def test_indexes_stable_points_with_complete_payloads(self) -> None:
        chunk = _chunk()
        embedder, store = _FakeEmbedder(), _FakeStore()
        summary = index_chunks([chunk], collection_name="aura-test", embedder=embedder, store=store, chunk_bytes=b"chunk-data")

        self.assertEqual(summary.chunk_count, 1)
        self.assertEqual(summary.vector_size, 3)
        self.assertEqual(store.collection_name, "aura-test")
        self.assertTrue(store.payload_indexes_called)
        self.assertEqual(store.points[0].id, _point_id(chunk))
        self.assertEqual(store.points[0].payload["chunk_id"], chunk["chunk_id"])
        self.assertFalse(store.points[0].payload["is_appendix"])
        self.assertEqual(embedder.documents[0].title, "Example Paper")
        self.assertIn("Section: Method", embedder.documents[0].text)

    def test_rejects_duplicate_chunk_ids(self) -> None:
        with self.assertRaisesRegex(IndexingError, "unique"):
            index_chunks([_chunk(), _chunk()], collection_name="aura-test", embedder=_FakeEmbedder(), store=_FakeStore(), chunk_bytes=b"chunk-data")

    def test_manifest_records_reproducible_index_fingerprint(self) -> None:
        summary = index_chunks([_chunk()], collection_name="aura-test", embedder=_FakeEmbedder(), store=_FakeStore(), chunk_bytes=b"chunk-data")
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "index-manifest.json"
            write_index_manifest(manifest_path, summary)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["collection_name"], "aura-test")
        self.assertEqual(manifest["chunk_sha256"], summary.chunk_sha256)
        self.assertIn("indexed_at", manifest)

    def test_load_chunks_validates_character_count(self) -> None:
        invalid = _chunk()
        invalid["character_count"] = 0
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "chunks.jsonl"
            path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(IndexingError, "character count"):
                load_chunks(path)

    def test_embedding_document_has_paper_and_section_context(self) -> None:
        document = _embedding_document(_chunk())
        self.assertEqual(document.title, "Example Paper")
        self.assertTrue(document.text.startswith("Paper: Example Paper"))


if __name__ == "__main__":
    unittest.main()
