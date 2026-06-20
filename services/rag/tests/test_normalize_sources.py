"""Tests for section-aware LaTeX normalization."""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from services.rag.aura_rag.manifest import (
    CorpusManifest,
    EmbeddingPolicy,
    PaperManifest,
    SourcePolicy,
)
from services.rag.aura_rag.normalize_sources import LatexNormalizer, normalize_sources


PAPER = PaperManifest(
    id="example-paper",
    title="Example Paper",
    authors=["A. Author"],
    year=2026,
    arxiv_id="2601.00001",
    topics=["testing"],
    priority=1,
    source_preference="arxiv_latex",
    status="planned",
)
MANIFEST = CorpusManifest(
    corpus_id="example-v1",
    name="Example",
    description="Example corpus",
    status="planned",
    source_policy=SourcePolicy("arxiv_latex", "pdf"),
    embedding_policy=EmbeddingPolicy("example-model", "qdrant", "cosine"),
    papers=[PAPER],
)


class LatexNormalizerTests(unittest.TestCase):
    def test_preserves_hierarchy_equations_captions_and_included_prose(self) -> None:
        archive_path = _make_archive(
            {
                "main.tex": r"""\documentclass{article}
\newcommand\modelname{AURA}
\begin{document}
\begin{abstract}A short \textbf{abstract} from \modelname.\end{abstract}
\section{Method} \input{parts/method}
\subsection{Details} More detail.
\end{document}""",
                "parts/method.tex": r"""Method prose \cite{source}.
\begin{equation} E = mc^2 \end{equation}
$$ F = ma $$
\begin{figure}\caption{A useful figure.}\end{figure}""",
            }
        )
        self.addCleanup(archive_path.unlink)

        document = LatexNormalizer(archive_path, "main.tex").normalize(MANIFEST, PAPER)

        self.assertEqual([section["path"] for section in document["sections"]], [["Abstract"], ["Method"], ["Method", "Details"]])
        self.assertIn("AURA", document["sections"][0]["blocks"][0]["text"])
        method = document["sections"][1]
        self.assertIn("Method prose .", method["blocks"][0]["text"])
        self.assertEqual(method["blocks"][1], {"type": "equation", "latex": "E = mc^2", "source_file": "parts/method.tex"})
        self.assertEqual(method["blocks"][2], {"type": "equation", "latex": "F = ma", "source_file": "parts/method.tex"})
        self.assertEqual(method["blocks"][3], {"type": "caption", "text": "A useful figure.", "source_file": "parts/method.tex"})

    def test_skips_pdf_fallback_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            processed = root / "processed" / MANIFEST.corpus_id / PAPER.id
            processed.mkdir(parents=True)
            (processed / "inspection.json").write_text(
                json.dumps({"extractable": False, "root_file": "main.tex"}), encoding="utf-8"
            )

            result = normalize_sources(MANIFEST, root / "raw", root / "processed")

        self.assertEqual(result[0].outcome, "skipped")
        self.assertEqual(result[0].message, "PDF fallback pending.")


def _make_archive(files: dict[str, str]) -> Path:
    temporary_file = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    archive_path = Path(temporary_file.name)
    temporary_file.close()
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, contents in files.items():
            encoded = contents.encode("utf-8")
            member = tarfile.TarInfo(name)
            member.size = len(encoded)
            archive.addfile(member, io.BytesIO(encoded))
    return archive_path
