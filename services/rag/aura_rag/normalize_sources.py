"""Normalize inspected arXiv LaTeX sources into section-aware JSON documents.

This is deliberately a conservative first pass.  It preserves the document's
reading order, hierarchy, prose, display equations, and captions while
discarding layout-oriented LaTeX.  It is not a TeX renderer.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import tarfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .acquire_sources import select_papers
from .manifest import CorpusManifest, PaperManifest, load_manifest


SCHEMA_VERSION = "1"
SOURCE_MARKER = "@@AURA_SOURCE:{source}@@"
INCLUDE_PATTERN = re.compile(r"\\(?:input|include)\s*\{\s*([^}]+?)\s*\}")
SECTION_PATTERN = re.compile(r"\\(section|subsection|subsubsection)\*?\s*\{")
DISPLAY_MATH_PATTERN = re.compile(
    r"\\begin\{(?P<environment>equation\*?|align\*?|gather\*?|multline\*?|displaymath)\}"
    r"(?P<environment_text>.*?)\\end\{(?P=environment)\}"
    r"|\$\$(?P<dollar_text>.*?)\$\$"
    r"|\\\[(?P<bracket_text>.*?)\\\]",
    re.DOTALL,
)
CAPTION_PATTERN = re.compile(r"\\caption(?:\[[^]]*\])?\s*\{")
COMMAND_PATTERN = re.compile(r"\\([A-Za-z@]+)\*?")
MACRO_DEFINITION_PATTERN = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand)\s*(?:\{\\([A-Za-z@]+)\}|\\([A-Za-z@]+))"
    r"(?:\s*\[[^]]*\])?\s*\{([^{}]*)\}"
)


class NormalizationError(RuntimeError):
    """Raised when a source archive cannot be normalized."""


@dataclass(frozen=True)
class NormalizationResult:
    paper_id: str
    outcome: str
    output_path: Path | None
    section_count: int = 0
    block_count: int = 0
    character_count: int = 0
    equation_count: int = 0
    warnings: tuple[str, ...] = ()
    message: str = ""


@dataclass
class _Section:
    id: str
    path: list[str]
    level: int
    source_files: list[str] = field(default_factory=list)
    blocks: list[dict[str, str]] = field(default_factory=list)


class LatexNormalizer:
    """Expand a LaTeX document tree and create a retrieval-oriented document."""

    def __init__(self, archive_path: str | Path, root_file: str) -> None:
        self.archive_path = Path(archive_path)
        self.root_file = root_file
        self.sources = self._load_sources()
        self.macros = _extract_simple_macros(self.sources.values())
        self.warnings: Counter[str] = Counter()

    def normalize(self, manifest: CorpusManifest, paper: PaperManifest) -> dict[str, Any]:
        expanded = self._expand(self.root_file, set(), is_root=True)
        sections = self._parse_sections(expanded)
        return {
            "schema_version": SCHEMA_VERSION,
            "corpus_id": manifest.corpus_id,
            "paper_id": paper.id,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "topics": paper.topics,
            "source": {
                "type": "arxiv_latex",
                "arxiv_id": paper.arxiv_id,
                "archive_path": str(self.archive_path),
                "root_file": self.root_file,
            },
            "sections": [asdict(section) for section in sections],
            "warnings": sorted(self.warnings),
        }

    def _load_sources(self) -> dict[str, str]:
        try:
            with tarfile.open(self.archive_path, mode="r:*") as archive:
                sources: dict[str, str] = {}
                for member in archive.getmembers():
                    if not member.isfile() or not member.name.lower().endswith(".tex"):
                        continue
                    file_object = archive.extractfile(member)
                    if file_object is not None:
                        sources[_normalize_path(member.name)] = file_object.read().decode(
                            "utf-8", errors="replace"
                        )
        except (OSError, tarfile.TarError) as exc:
            raise NormalizationError(f"Could not read {self.archive_path}: {exc}") from exc
        return sources

    def _expand(self, file_name: str, ancestry: set[str], *, is_root: bool = False) -> str:
        if file_name in ancestry:
            self.warnings[f"cyclic include skipped: {file_name}"] += 1
            return ""
        contents = _without_comments(self.sources[file_name])
        if is_root:
            contents = _document_body(contents)

        def replace_include(match: re.Match[str]) -> str:
            target = self._resolve_include(file_name, match.group(1))
            if target is None:
                self.warnings[f"unresolved include: {file_name} -> {match.group(1).strip()}"] += 1
                return ""
            # Restore the parent marker after an inlined file so provenance for
            # the text following an include remains correct.
            return self._expand(target, ancestry | {file_name}) + f"\n{SOURCE_MARKER.format(source=file_name)}\n"

        expanded = INCLUDE_PATTERN.sub(replace_include, contents)
        return f"\n{SOURCE_MARKER.format(source=file_name)}\n{expanded}\n"

    def _resolve_include(self, source_file: str, requested_path: str) -> str | None:
        requested_path = requested_path.strip()
        if not requested_path or "\\" in requested_path:
            return None
        candidates = [requested_path]
        if not requested_path.lower().endswith(".tex"):
            candidates.append(f"{requested_path}.tex")
        source_directory = str(PurePosixPath(source_file).parent)
        for candidate in candidates:
            for resolved in (
                _normalize_path(posixpath.join(source_directory, candidate)),
                _normalize_path(candidate),
            ):
                if resolved in self.sources:
                    return resolved
        return None

    def _parse_sections(self, expanded: str) -> list[_Section]:
        sections: list[_Section] = []
        stack: list[str] = []
        current: _Section | None = None
        appendix_mode = False

        for source_file, text in _source_segments(expanded):
            cursor = 0
            event_pattern = re.compile(r"\\appendix\b|\\begin\{abstract\}|\\end\{abstract\}|" + SECTION_PATTERN.pattern)
            for match in event_pattern.finditer(text):
                if current is not None:
                    self._append_blocks(current, text[cursor : match.start()], source_file)
                token = match.group(0)
                if token == "\\appendix":
                    appendix_mode = True
                    stack = ["Appendix"]
                    current = None
                elif token == "\\begin{abstract}":
                    stack = ["Abstract"]
                    current = self._new_section(sections, stack, 1, source_file)
                elif token == "\\end{abstract}":
                    current = None
                    stack = []
                else:
                    command = match.group(1)
                    level = {"section": 1, "subsection": 2, "subsubsection": 3}[command]
                    heading, end = _read_braced(text, match.end() - 1)
                    heading = self._clean_inline(heading)
                    prefix = ["Appendix"] if appendix_mode else []
                    stack = prefix + stack[len(prefix) : level - 1] + [heading]
                    current = self._new_section(sections, stack, level, source_file)
                    cursor = end
                    continue
                cursor = match.end()
            if current is not None:
                self._append_blocks(current, text[cursor:], source_file)

        return [section for section in sections if section.blocks]

    def _new_section(
        self, sections: list[_Section], path: list[str], level: int, source_file: str
    ) -> _Section:
        section = _Section(f"{len(sections) + 1:03d}", list(path), level, [source_file])
        sections.append(section)
        return section

    def _append_blocks(self, section: _Section, text: str, source_file: str) -> None:
        block_count_before = len(section.blocks)
        cursor = 0
        while cursor < len(text):
            equation_match = DISPLAY_MATH_PATTERN.search(text, cursor)
            caption_match = CAPTION_PATTERN.search(text, cursor)
            matches = [match for match in (equation_match, caption_match) if match is not None]
            if not matches:
                self._append_paragraphs(section, text[cursor:], source_file)
                break
            match = min(matches, key=lambda candidate: candidate.start())
            self._append_paragraphs(section, text[cursor : match.start()], source_file)
            if match.re is DISPLAY_MATH_PATTERN:
                equation = _clean_equation(
                    match.group("environment_text")
                    or match.group("dollar_text")
                    or match.group("bracket_text")
                    or ""
                )
                if equation:
                    section.blocks.append(
                        {"type": "equation", "latex": equation, "source_file": source_file}
                    )
                cursor = match.end()
            else:
                caption, cursor = _read_braced(text, match.end() - 1)
                cleaned = self._clean_inline(caption)
                if cleaned:
                    section.blocks.append(
                        {"type": "caption", "text": cleaned, "source_file": source_file}
                    )
        if len(section.blocks) > block_count_before and source_file not in section.source_files:
            section.source_files.append(source_file)

    def _append_paragraphs(self, section: _Section, text: str, source_file: str) -> None:
        for paragraph in re.split(r"\n\s*\n", text):
            cleaned = self._clean_inline(paragraph)
            if cleaned:
                section.blocks.append({"type": "paragraph", "text": cleaned, "source_file": source_file})

    def _clean_inline(self, text: str) -> str:
        for name, replacement in self.macros.items():
            text = re.sub(rf"\\{re.escape(name)}\b", lambda _match: replacement, text)
        text = re.sub(r"\\(?:bibliography|bibliographystyle)\s*\{[^}]*\}", "", text)
        text = re.sub(r"\\(?:label|ref|eqref|autoref|cref|cite|citep|citet|citeauthor|citeyear)\w*\s*\{[^}]*\}", "", text)
        text = re.sub(r"\\(?:begin|end)\s*\{(?:figure\*?|table\*?|center|itemize|enumerate|tabular\*?|minipage)\}", "", text)
        text = re.sub(r"\\item\b", "- ", text)
        text = re.sub(r"\\includegraphics(?:\[[^]]*\])?\s*\{[^}]*\}", "", text)
        text = re.sub(r"\\(?:vspace|hspace|hfill|newline|newpage|clearpage|centering|small|footnotesize|scriptsize|large|Large|LARGE)\*?(?:\[[^]]*\])?(?:\{[^}]*\})?", "", text)
        text = re.sub(r"\\url\s*\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\href\s*\{[^}]*\}\s*\{([^}]*)\}", r"\1", text)
        for _ in range(3):
            text = re.sub(r"\\(?:textbf|textit|textrm|texttt|emph|mathbf|mathrm|mathit|operatorname)\s*\{([^{}]*)\}", r"\1", text)
        text = text.replace("~", " ").replace("\\%", "%").replace("\\_", "_")
        text = text.replace("\\&", "&").replace("\\#", "#")
        unknown = {match.group(1) for match in COMMAND_PATTERN.finditer(text)}
        ignored = {"begin", "end", "left", "right", "quad", "qquad", "displaystyle", "mathrm", "text"}
        for command in sorted(unknown - ignored):
            self.warnings[f"unhandled command: \\{command}"] += 1
        text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", "", text)
        text = text.replace("{", "").replace("}", "")
        return re.sub(r"\s+", " ", text).strip(" -")


def normalize_sources(
    manifest: CorpusManifest,
    input_root: str | Path,
    processed_root: str | Path,
    *,
    include_legacy: bool = False,
    dry_run: bool = False,
) -> list[NormalizationResult]:
    """Normalize every inspected LaTeX-source paper in a corpus manifest."""
    results: list[NormalizationResult] = []
    for paper in select_papers(manifest, include_legacy):
        report_path = Path(processed_root) / manifest.corpus_id / paper.id / "inspection.json"
        if not report_path.is_file():
            raise NormalizationError(f"Missing inspection report for {paper.id}: {report_path}")
        inspection = json.loads(report_path.read_text(encoding="utf-8"))
        if not inspection["extractable"]:
            results.append(
                NormalizationResult(paper.id, "skipped", None, message="PDF fallback pending.")
            )
            continue
        normalizer = LatexNormalizer(
            Path(input_root) / manifest.corpus_id / paper.id / "source.tar.gz",
            inspection["root_file"],
        )
        document = normalizer.normalize(manifest, paper)
        sections = document["sections"]
        blocks = [block for section in sections for block in section["blocks"]]
        warnings = tuple(document["warnings"])
        output_path = Path(processed_root) / manifest.corpus_id / paper.id / "normalized.json"
        if not dry_run:
            output_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        results.append(
            NormalizationResult(
                paper.id,
                "normalized",
                output_path,
                len(sections),
                len(blocks),
                sum(len(block.get("text", block.get("latex", ""))) for block in blocks),
                sum(block["type"] == "equation" for block in blocks),
                warnings,
                "Normalized LaTeX source.",
            )
        )
    return results


def _source_segments(expanded: str) -> list[tuple[str, str]]:
    marker = re.compile(r"@@AURA_SOURCE:([^@]+)@@")
    matches = list(marker.finditer(expanded))
    return [
        (match.group(1), expanded[match.end() : matches[index + 1].start() if index + 1 < len(matches) else None])
        for index, match in enumerate(matches)
    ]


def _without_comments(contents: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", contents)


def _document_body(contents: str) -> str:
    start = re.search(r"\\begin\{document\}", contents)
    end = re.search(r"\\end\{document\}", contents)
    if start is None:
        return contents
    return contents[start.end() : end.start() if end else None]


def _read_braced(text: str, opening_brace_index: int) -> tuple[str, int]:
    depth = 0
    for index in range(opening_brace_index, len(text)):
        if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[opening_brace_index + 1 : index], index + 1
    return text[opening_brace_index + 1 :], len(text)


def _clean_equation(text: str) -> str:
    text = re.sub(r"\\label\s*\{[^}]*\}", "", _without_comments(text))
    return re.sub(r"\s+", " ", text).strip()


def _normalize_path(path: str) -> str:
    return posixpath.normpath(path).lstrip("/").removeprefix("./")


def _extract_simple_macros(sources: Any) -> dict[str, str]:
    macros: dict[str, str] = {}
    for source in sources:
        for match in MACRO_DEFINITION_PATTERN.finditer(_without_comments(source)):
            name = match.group(1) or match.group(2)
            replacement = match.group(3)
            if name and not "#" in replacement:
                macros[name] = replacement
    return macros


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize inspected AURA LaTeX sources.")
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("--input-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest_path)
    try:
        results = normalize_sources(
            manifest,
            args.input_root,
            args.processed_root,
            include_legacy=args.include_legacy,
            dry_run=args.dry_run,
        )
    except NormalizationError as exc:
        print(exc)
        return 1
    for result in results:
        print(
            f"{result.outcome.upper():10} {result.paper_id}: sections={result.section_count}, "
            f"blocks={result.block_count}, chars={result.character_count}, equations={result.equation_count}, "
            f"warnings={len(result.warnings)}. {result.message}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
