"""Tests for archive structure inspection."""

from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from services.rag.aura_rag.inspect_sources import LatexArchiveInspector
from services.rag.aura_rag.manifest import PaperManifest


PAPER = PaperManifest(
    id="example-paper",
    title="Example",
    authors=["A. Author"],
    year=2026,
    arxiv_id="2601.00001",
    topics=["testing"],
    priority=1,
    source_preference="arxiv_latex",
    status="planned",
)


class LatexArchiveInspectorTests(unittest.TestCase):
    def test_resolves_nested_includes_and_counts_sections(self) -> None:
        archive_path = _make_archive(
            {
                "main.tex": "\\documentclass{article}\n\\input{sections/intro}\n",
                "sections/intro.tex": "\\section{Intro}\n\\input{methods.tex}\n",
                "sections/methods.tex": "\\subsection{Method}\n",
            }
        )
        self.addCleanup(archive_path.unlink)

        report = LatexArchiveInspector(archive_path).inspect(PAPER)

        self.assertEqual(report.root_file, "main.tex")
        self.assertEqual(report.resolved_files, ["main.tex", "sections/intro.tex", "sections/methods.tex"])
        self.assertEqual(report.section_counts, {"section": 1, "subsection": 1})
        self.assertTrue(report.extractable)

    def test_marks_unresolved_and_cyclic_includes_for_review(self) -> None:
        archive_path = _make_archive(
            {
                "main.tex": "\\documentclass{article}\n\\input{part}\n\\input{missing}\n",
                "part.tex": "\\include{main}\n",
            }
        )
        self.addCleanup(archive_path.unlink)

        report = LatexArchiveInspector(archive_path).inspect(PAPER)

        self.assertEqual(len(report.unresolved_includes), 1)
        self.assertEqual(len(report.cyclic_includes), 1)
        self.assertFalse(report.extractable)

    def test_marks_pdf_wrapper_for_fallback(self) -> None:
        archive_path = _make_archive(
            {
                "main.tex": "\\documentclass{article}\n\\begin{document}\n"
                "\\includepdf[pages=1-last]{paper.pdf}\n\\end{document}\n",
            }
        )
        self.addCleanup(archive_path.unlink)

        report = LatexArchiveInspector(archive_path).inspect(PAPER)

        self.assertEqual(report.embedded_pdf_references, ["paper.pdf"])
        self.assertTrue(report.requires_pdf_fallback)
        self.assertFalse(report.extractable)


def _make_archive(files: dict[str, str]) -> Path:
    temporary_file = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    archive_path = Path(temporary_file.name)
    temporary_file.close()
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, contents in files.items():
            encoded_contents = contents.encode("utf-8")
            member = tarfile.TarInfo(name)
            member.size = len(encoded_contents)
            archive.addfile(member, io.BytesIO(encoded_contents))
    return archive_path
