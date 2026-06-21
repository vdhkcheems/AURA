"""Gemini embedding adapters used by AURA's offline and query-time retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import Protocol, Sequence


GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_VECTOR_SIZE = 768


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider returns an unusable response."""


@dataclass(frozen=True)
class EmbeddingDocument:
    """A retrieval document with the title Gemini uses as document context."""

    text: str
    title: str


class Embedder(Protocol):
    """Provider-neutral contract for document and question embeddings."""

    model_name: str
    vector_size: int

    def embed_documents(self, documents: Sequence[EmbeddingDocument]) -> list[list[float]]: ...

    def embed_question(self, question: str) -> list[float]: ...


class GeminiEmbedder:
    """Embed AURA text with Gemini's asymmetric retrieval task types.

    Imports are intentionally lazy so corpus parsing tests do not require the
    optional hosted-service dependencies.
    """

    model_name = GEMINI_EMBEDDING_MODEL

    def __init__(
        self,
        api_key: str,
        *,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        batch_size: int = 6,
        minimum_batch_interval_seconds: float = 10.0,
        rate_limit_retries: int = 10,
        rate_limit_backoff_seconds: float = 65.0,
    ) -> None:
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY is required for Gemini embeddings")
        if vector_size <= 0:
            raise EmbeddingError("vector_size must be positive")
        if batch_size <= 0:
            raise EmbeddingError("batch_size must be positive")
        if minimum_batch_interval_seconds < 0:
            raise EmbeddingError("minimum_batch_interval_seconds cannot be negative")
        if rate_limit_retries < 0 or rate_limit_backoff_seconds <= 0:
            raise EmbeddingError("rate-limit retry settings must be non-negative and positive")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - exercised in deployment setup
            raise EmbeddingError(
                "Gemini dependencies are missing. Install services/rag/requirements.txt."
            ) from exc
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.vector_size = vector_size
        self.batch_size = batch_size
        self.minimum_batch_interval_seconds = minimum_batch_interval_seconds
        self.rate_limit_retries = rate_limit_retries
        self.rate_limit_backoff_seconds = rate_limit_backoff_seconds

    def embed_documents(self, documents: Sequence[EmbeddingDocument]) -> list[list[float]]:
        vectors: list[list[float]] = []
        request_count = 0
        for title, group in _group_by_title(documents):
            for start in range(0, len(group), self.batch_size):
                batch = group[start : start + self.batch_size]
                if request_count:
                    sleep(self.minimum_batch_interval_seconds)
                response = self._embed_with_backoff(
                    contents=[document.text for document in batch],
                    config=self._types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        title=title,
                        output_dimensionality=self.vector_size,
                    ),
                    kind="document",
                )
                vectors.extend(_extract_vectors(response.embeddings, self.vector_size))
                request_count += 1
        return vectors

    def embed_question(self, question: str) -> list[float]:
        if not question.strip():
            raise EmbeddingError("question must not be empty")
        response = self._embed_with_backoff(
            contents=question,
            config=self._types.EmbedContentConfig(
                task_type="QUESTION_ANSWERING",
                output_dimensionality=self.vector_size,
            ),
            kind="question",
        )
        vectors = _extract_vectors(response.embeddings, self.vector_size)
        if len(vectors) != 1:
            raise EmbeddingError("Gemini returned an unexpected number of query vectors")
        return vectors[0]

    def _embed_with_backoff(self, *, contents: object, config: object, kind: str):
        for attempt in range(self.rate_limit_retries + 1):
            try:
                return self._client.models.embed_content(model=self.model_name, contents=contents, config=config)
            except Exception as exc:  # Provider errors are intentionally dependency-agnostic.
                if not _is_rate_limited(exc) or attempt == self.rate_limit_retries:
                    raise EmbeddingError(f"Gemini {kind} embedding request failed: {exc}") from exc
                sleep(self.rate_limit_backoff_seconds)
        raise AssertionError("unreachable")


def _group_by_title(documents: Sequence[EmbeddingDocument]) -> list[tuple[str, list[EmbeddingDocument]]]:
    """Group without changing document order inside each title group."""
    grouped: dict[str, list[EmbeddingDocument]] = {}
    for document in documents:
        if not document.text.strip() or not document.title.strip():
            raise EmbeddingError("embedding documents require non-empty text and title")
        grouped.setdefault(document.title, []).append(document)
    return list(grouped.items())


def _extract_vectors(embeddings: object, vector_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    for embedding in embeddings:  # type: ignore[union-attr]
        values = list(embedding.values)
        if len(values) != vector_size:
            raise EmbeddingError(
                f"Gemini returned a {len(values)}-dimension vector; expected {vector_size}"
            )
        vectors.append([float(value) for value in values])
    return vectors


def _is_rate_limited(error: Exception) -> bool:
    message = str(error)
    return "429" in message or "RESOURCE_EXHAUSTED" in message
