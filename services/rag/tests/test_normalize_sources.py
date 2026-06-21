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

    def test_preserves_inline_dollar_math_verbatim(self) -> None:
        archive_path = _make_archive(
            {
                "main.tex": r"""\documentclass{article}
\newcommand\modelname{AURA}
\begin{document}
\section{Method}
The score is $\frac{x_i-\mu}{\sigma}$ for $\modelname$ and $x^2$. A bold form is \boldmath$\Theta$.
\end{document}""",
            }
        )
        self.addCleanup(archive_path.unlink)

        document = LatexNormalizer(archive_path, "main.tex").normalize(MANIFEST, PAPER)

        self.assertEqual(
            document["sections"][0]["blocks"][0]["text"],
            r"The score is $\frac{x_i-\mu}{\sigma}$ for $\modelname$ and $x^2$. A bold form is $\Theta$.",
        )
        self.assertNotIn("unhandled command: \\frac", document["warnings"])
        self.assertNotIn("unhandled command: \\modelname", document["warnings"])

    def test_preserves_science_style_abstract_and_sectionless_introduction(self) -> None:
        archive_path = _make_archive(
            {
                "main.tex": r"""\documentclass{article}
\begin{document}
\begin{sciabstract}A short abstract with $x_0$.\end{sciabstract}
Sectionless body prose preserves $2^N$ possible models.
\begin{scilastnote}Thanks to the reviewers.\end{scilastnote}
\appendix
\section{Proofs} Appendix prose.
\end{document}""",
            }
        )
        self.addCleanup(archive_path.unlink)

        document = LatexNormalizer(archive_path, "main.tex").normalize(MANIFEST, PAPER)

        self.assertEqual(
            [section["path"] for section in document["sections"]],
            [["Abstract"], ["Introduction"], ["Acknowledgments"], ["Appendix", "Proofs"]],
        )
        self.assertEqual(document["sections"][0]["blocks"][0]["text"], "A short abstract with $x_0$.")
        self.assertEqual(
            document["sections"][1]["blocks"][0]["text"],
            "Sectionless body prose preserves $2^N$ possible models.",
        )
        self.assertEqual(document["sections"][2]["blocks"][0]["text"], "Thanks to the reviewers.")

    def test_preserves_caption_text_wrapped_in_font_size_command(self) -> None:
        archive_path = _make_archive(
            {
                "main.tex": r"""\documentclass{article}
\begin{document}
\section{Results}
\begin{figure}\caption{\small{A caption with $x_i$ preserved.}}\end{figure}
\end{document}""",
            }
        )
        self.addCleanup(archive_path.unlink)

        document = LatexNormalizer(archive_path, "main.tex").normalize(MANIFEST, PAPER)

        self.assertEqual(
            document["sections"][0]["blocks"],
            [{"type": "caption", "text": "A caption with $x_i$ preserved.", "source_file": "main.tex"}],
        )

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
