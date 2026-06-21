"""Thin Qdrant adapter kept separate from AURA retrieval domain logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


class VectorStoreError(RuntimeError):
    """Raised when Qdrant cannot satisfy AURA's index contract."""


@dataclass(frozen=True)
class VectorPoint:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    score: float
    payload: dict[str, Any]


class QdrantVectorStore:
    """Qdrant implementation of collection setup, upserts, and filtered search."""

    def __init__(self, url: str, api_key: str | None, *, vector_size: int) -> None:
        if not url:
            raise VectorStoreError("QDRANT_URL is required")
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - exercised in deployment setup
            raise VectorStoreError(
                "Qdrant dependencies are missing. Install services/rag/requirements.txt."
            ) from exc
        self._client = QdrantClient(url=url, api_key=api_key)
        self.vector_size = vector_size

    def ensure_collection(self, collection_name: str) -> None:
        from qdrant_client.http.models import Distance, VectorParams

        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            return
        collection = self._client.get_collection(collection_name)
        configured_size = collection.config.params.vectors.size
        if configured_size != self.vector_size:
            raise VectorStoreError(
                f"Collection {collection_name!r} has vectors of size {configured_size}; "
                f"expected {self.vector_size}. Use a new versioned collection."
            )

    def ensure_payload_indexes(self, collection_name: str) -> None:
        from qdrant_client.http.models import PayloadSchemaType

        for field_name, schema in (
            ("paper_id", PayloadSchemaType.KEYWORD),
            ("topics", PayloadSchemaType.KEYWORD),
            ("year", PayloadSchemaType.INTEGER),
            ("section_path", PayloadSchemaType.KEYWORD),
            ("is_appendix", PayloadSchemaType.BOOL),
        ):
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )

    def upsert(self, collection_name: str, points: Sequence[VectorPoint]) -> None:
        from qdrant_client.http.models import PointStruct

        self._client.upsert(
            collection_name=collection_name,
            points=[PointStruct(id=point.id, vector=point.vector, payload=point.payload) for point in points],
            wait=True,
        )

    def count(self, collection_name: str) -> int:
        return self._client.count(collection_name=collection_name, exact=True).count

    def search(
        self,
        collection_name: str,
        vector: list[float],
        *,
        limit: int,
        paper_id: str | None = None,
        topic: str | None = None,
    ) -> list[SearchResult]:
        from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue

        conditions = []
        if paper_id:
            conditions.append(FieldCondition(key="paper_id", match=MatchValue(value=paper_id)))
        if topic:
            conditions.append(FieldCondition(key="topics", match=MatchAny(any=[topic])))
        response = self._client.query_points(
            collection_name=collection_name,
            query=vector,
            query_filter=Filter(must=conditions) if conditions else None,
            limit=limit,
            with_payload=True,
        )
        return [SearchResult(score=point.score, payload=dict(point.payload or {})) for point in response.points]
