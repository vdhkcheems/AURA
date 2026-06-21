"""Tests for embedding adapter response validation without Gemini network calls."""

from __future__ import annotations

import unittest

from services.rag.aura_rag.embeddings import EmbeddingDocument, EmbeddingError, GeminiEmbedder


class _FailingModels:
    def embed_content(self, **_kwargs):
        raise RuntimeError("quota exhausted")


class _FailingClient:
    models = _FailingModels()


class _Types:
    class EmbedContentConfig:
        def __init__(self, **_kwargs) -> None:
            pass


class GeminiEmbedderTests(unittest.TestCase):
    def test_provider_errors_are_exposed_as_embedding_errors(self) -> None:
        embedder = object.__new__(GeminiEmbedder)
        embedder._client = _FailingClient()
        embedder._types = _Types()
        embedder.vector_size = 768
        embedder.batch_size = 32
        embedder.minimum_batch_interval_seconds = 0
        embedder.rate_limit_retries = 0
        embedder.rate_limit_backoff_seconds = 1

        with self.assertRaisesRegex(EmbeddingError, "quota exhausted"):
            embedder.embed_documents([EmbeddingDocument("text", "title")])


if __name__ == "__main__":
    unittest.main()
