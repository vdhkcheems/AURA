"""Tests for curated arXiv source acquisition."""

from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from services.rag.aura_rag.acquire_sources import (
    SourceAcquisitionError,
    acquire_sources,
    validate_latex_archive,
)
from services.rag.aura_rag.manifest import load_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "data/manifests/ml-core-v1.json"


class SourceAcquisitionTests(unittest.TestCase):
    def test_dry_run_selects_planned_latex_papers_only(self) -> None:
        manifest = load_manifest(MANIFEST_PATH)

        with tempfile.TemporaryDirectory() as temporary_directory:
            results = acquire_sources(manifest, temporary_directory, dry_run=True)

        self.assertEqual(len(results), 10)
        self.assertTrue(all(result.outcome == "planned" for result in results))
        self.assertNotIn("attention-is-all-you-need", [result.paper_id for result in results])

    def test_downloaded_archive_is_validated_and_recorded(self) -> None:
        manifest = load_manifest(MANIFEST_PATH)
        archive_bytes = _latex_archive_bytes()

        def opener(*_args: object, **_kwargs: object) -> _Response:
            return _Response(archive_bytes)

        with tempfile.TemporaryDirectory() as temporary_directory:
            results = acquire_sources(manifest, temporary_directory, opener=opener)
            paper_root = Path(temporary_directory) / manifest.corpus_id / "bert"
            self.assertEqual(len(results), 10)
            self.assertTrue(all(result.outcome == "downloaded" for result in results))
            self.assertTrue((paper_root / "source.tar.gz").is_file())
            self.assertTrue((paper_root / "acquisition.json").is_file())

    def test_archive_without_latex_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "no-latex.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                contents = b"plain text"
                member = tarfile.TarInfo("readme.txt")
                member.size = len(contents)
                archive.addfile(member, io.BytesIO(contents))

            with self.assertRaisesRegex(SourceAcquisitionError, "no .tex files"):
                validate_latex_archive(archive_path)


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _latex_archive_bytes() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        contents = b"\\documentclass{article}\\begin{document}AURA\\end{document}"
        member = tarfile.TarInfo("paper/main.tex")
        member.size = len(contents)
        archive.addfile(member, io.BytesIO(contents))
    return buffer.getvalue()
