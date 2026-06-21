"""Tests for the retrieval evaluation file and paper-level recall calculation."""

from __future__ import annotations

import unittest
from pathlib import Path

from services.rag.aura_rag.evaluate_retrieval import evaluate, load_cases
from services.rag.aura_rag.qdrant_store import SearchResult


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class _Embedder:
    def embed_question(self, _question):
        return [0.1]


class _Store:
    def search(self, _collection_name, _vector, *, limit, paper_id=None, topic=None):
        return [SearchResult(0.9, {
            "chunk_id": "paper-a::001::0001", "paper_id": "paper-a", "title": "Paper A",
            "section_path": ["Method"], "text": "Evidence", "source_files": ["main.tex"],
            "related_figure_ids": [],
        })]


class EvaluateRetrievalTests(unittest.TestCase):
    def test_tracks_paper_level_hits(self) -> None:
        results = evaluate(
            [{"id": "case", "question": "question", "expected_paper_ids": ["paper-a"]}],
            collection_name="aura-test", embedder=_Embedder(), store=_Store(), limit=5,
        )
        self.assertTrue(results[0].hit)
        self.assertEqual(results[0].retrieved_paper_ids, ["paper-a"])

    def test_shipped_evaluation_set_covers_every_indexed_paper(self) -> None:
        cases = load_cases(REPOSITORY_ROOT / "data/evaluations/ml-core-v1-retrieval.jsonl")
        paper_ids = {paper_id for case in cases for paper_id in case["expected_paper_ids"]}
        self.assertEqual(len(cases), 21)
        self.assertEqual(len(paper_ids), 10)


if __name__ == "__main__":
    unittest.main()
