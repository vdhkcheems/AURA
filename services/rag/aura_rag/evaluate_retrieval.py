"""Run paper-level retrieval recall checks against a populated AURA index."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embeddings import DEFAULT_VECTOR_SIZE, EmbeddingError, GeminiEmbedder
from .index_qdrant import DEFAULT_COLLECTION
from .qdrant_store import QdrantVectorStore, VectorStoreError
from .retrieve_qdrant import RetrievalError, retrieve


class EvaluationError(RuntimeError):
    """Raised when a retrieval evaluation file is malformed."""


@dataclass(frozen=True)
class EvaluationResult:
    id: str
    expected_paper_ids: list[str]
    retrieved_paper_ids: list[str]
    hit: bool


def evaluate(cases: list[dict[str, Any]], *, collection_name: str, embedder, store, limit: int) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for case in cases:
        _validate_case(case)
        retrieved = retrieve(case["question"], collection_name=collection_name, embedder=embedder, store=store, limit=limit)
        paper_ids = list(dict.fromkeys(chunk.paper_id for chunk in retrieved))
        expected = case["expected_paper_ids"]
        results.append(EvaluationResult(case["id"], expected, paper_ids, all(paper_id in paper_ids for paper_id in expected)))
    return results


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    try:
        cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not read evaluation cases: {exc}") from exc
    if not cases:
        raise EvaluationError("Evaluation set is empty")
    for case in cases:
        _validate_case(case)
    return cases


def _validate_case(case: dict[str, Any]) -> None:
    if not isinstance(case.get("id"), str) or not case["id"].strip():
        raise EvaluationError("Every evaluation case needs a non-empty id")
    if not isinstance(case.get("question"), str) or not case["question"].strip():
        raise EvaluationError(f"Evaluation case {case['id']!r} needs a question")
    expected = case.get("expected_paper_ids")
    if not isinstance(expected, list) or not expected or not all(isinstance(item, str) and item for item in expected):
        raise EvaluationError(f"Evaluation case {case['id']!r} needs expected_paper_ids")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate paper-level recall for an AURA Qdrant index.")
    parser.add_argument("--cases", type=Path, default=Path("data/evaluations/ml-core-v1-retrieval.jsonl"))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
        cases = load_cases(args.cases)
        embedder = GeminiEmbedder(os.getenv("GEMINI_API_KEY", ""), vector_size=DEFAULT_VECTOR_SIZE)
        store = QdrantVectorStore(os.getenv("QDRANT_URL", ""), os.getenv("QDRANT_API_KEY"), vector_size=DEFAULT_VECTOR_SIZE)
        results = evaluate(cases, collection_name=args.collection, embedder=embedder, store=store, limit=args.limit)
    except (EvaluationError, RetrievalError, EmbeddingError, VectorStoreError) as exc:
        print(exc)
        return 1
    hit_count = sum(result.hit for result in results)
    print(json.dumps({"recall_at_k": hit_count / len(results), "hit_count": hit_count, "case_count": len(results), "results": [result.__dict__ for result in results]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
