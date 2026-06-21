"""Tests for Qdrant retrieval result shaping and filters."""

from __future__ import annotations

import unittest

from services.rag.aura_rag.qdrant_store import SearchResult
from services.rag.aura_rag.retrieve_qdrant import RetrievalError, retrieve


class _FakeEmbedder:
    model_name = "gemini-embedding-001"
    vector_size = 3

    def embed_documents(self, _documents):
        return []

    def embed_question(self, question):
        self.question = question
        return [0.1, 0.2, 0.3]


class _FakeStore:
    def search(self, collection_name, vector, *, limit, paper_id=None, topic=None):
        self.call = (collection_name, vector, limit, paper_id, topic)
        return [SearchResult(score=0.83, payload={
            "chunk_id": "example-paper::001::0001", "paper_id": "example-paper", "title": "Example Paper",
            "section_path": ["Method"], "text": "Useful evidence.", "source_files": ["main.tex"],
            "related_figure_ids": [],
        })]


class RetrieveQdrantTests(unittest.TestCase):
    def test_embeds_question_and_preserves_source_metadata(self) -> None:
        embedder, store = _FakeEmbedder(), _FakeStore()
        results = retrieve("How does the method work?", collection_name="aura-test", embedder=embedder, store=store, limit=3, paper_id="example-paper", topic="testing")
        self.assertEqual(embedder.question, "How does the method work?")
        self.assertEqual(store.call, ("aura-test", [0.1, 0.2, 0.3], 3, "example-paper", "testing"))
        self.assertEqual(results[0].section_path, ["Method"])
        self.assertEqual(results[0].score, 0.83)

    def test_rejects_invalid_query_limits(self) -> None:
        with self.assertRaisesRegex(RetrievalError, "between 1 and 20"):
            retrieve("question", collection_name="aura-test", embedder=_FakeEmbedder(), store=_FakeStore(), limit=21)


if __name__ == "__main__":
    unittest.main()
