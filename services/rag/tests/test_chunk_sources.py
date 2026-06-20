"""Tests for section-aware chunk generation."""

from __future__ import annotations

import unittest

from services.rag.aura_rag.chunk_sources import ChunkingConfig, chunk_document


def _document(blocks: list[dict[str, str]], *, section_path: list[str] | None = None) -> dict:
    return {
        "corpus_id": "example-v1",
        "paper_id": "example-paper",
        "title": "Example Paper",
        "authors": ["A. Author"],
        "year": 2026,
        "topics": ["testing"],
        "sections": [{"id": "001", "path": section_path or ["Method"], "blocks": blocks}],
    }


class ChunkSourcesTests(unittest.TestCase):
    def test_preserves_equations_captions_and_source_provenance(self) -> None:
        document = _document(
            [
                {"type": "paragraph", "text": "A" * 30, "source_file": "main.tex"},
                {"type": "equation", "latex": "E = mc^2", "source_file": "main.tex"},
                {"type": "caption", "text": "A useful figure.", "source_file": "figure.tex"},
            ]
        )
        chunks = chunk_document(document, ChunkingConfig(100, 20, 120, 10, 10))

        self.assertEqual(len(chunks), 1)
        self.assertIn("$$\nE = mc^2\n$$", chunks[0]["text"])
        self.assertIn("Figure: A useful figure.", chunks[0]["text"])
        self.assertEqual(chunks[0]["source_files"], ["main.tex", "figure.tex"])
        self.assertEqual(chunks[0]["block_range"], {"start": 0, "end": 3})

    def test_adds_paragraph_only_overlap_and_stable_ids(self) -> None:
        document = _document(
            [
                {"type": "paragraph", "text": "one " * 6, "source_file": "main.tex"},
                {"type": "paragraph", "text": "two " * 6, "source_file": "main.tex"},
                {"type": "equation", "latex": "x = y", "source_file": "main.tex"},
                {"type": "paragraph", "text": "three " * 6, "source_file": "main.tex"},
            ]
        )
        chunks = chunk_document(document, ChunkingConfig(35, 20, 80, 10, 30))

        self.assertEqual([chunk["chunk_id"] for chunk in chunks], [
            "example-paper::001::0001",
            "example-paper::001::0002",
            "example-paper::001::0003",
            "example-paper::001::0004",
        ])
        self.assertIn("one", chunks[0]["text"])
        self.assertIn("two", chunks[1]["text"])
        self.assertIn("$$", chunks[2]["text"])
        self.assertNotIn("$$", chunks[3]["text"])

    def test_merges_short_trailing_chunk_and_keeps_appendix_path(self) -> None:
        document = _document(
            [
                {"type": "paragraph", "text": "A" * 30, "source_file": "main.tex"},
                {"type": "paragraph", "text": "B" * 30, "source_file": "main.tex"},
                {"type": "paragraph", "text": "tail", "source_file": "main.tex"},
            ],
            section_path=["Appendix", "Proofs"],
        )
        chunks = chunk_document(document, ChunkingConfig(60, 50, 100, 40, 0))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["section_path"], ["Appendix", "Proofs"])
        self.assertTrue(chunks[0]["text"].endswith("tail"))

    def test_splits_oversized_paragraphs_but_not_equations(self) -> None:
        document = _document(
            [
                {"type": "paragraph", "text": "Sentence one. " * 12, "source_file": "main.tex"},
                {"type": "equation", "latex": "x = y = z", "source_file": "main.tex"},
            ]
        )
        chunks = chunk_document(document, ChunkingConfig(40, 20, 60, 10, 0))

        self.assertTrue(all(chunk["character_count"] <= 60 for chunk in chunks))
        equation_chunks = [chunk for chunk in chunks if "$$" in chunk["text"]]
        self.assertEqual(len(equation_chunks), 1)
        self.assertIn("x = y = z", equation_chunks[0]["text"])


if __name__ == "__main__":
    unittest.main()
