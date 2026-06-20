"""Inspect the structure of acquired LaTeX source archives."""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import tarfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from .acquire_sources import select_papers
from .manifest import CorpusManifest, PaperManifest, load_manifest


DOCUMENT_CLASS_PATTERN = re.compile(r"\\documentclass(?:\[[^]]*\])?\s*\{[^}]+\}")
INCLUDE_PATTERN = re.compile(r"\\(?:input|include)\s*\{\s*([^}]+?)\s*\}")
SECTION_PATTERN = re.compile(r"\\(section|subsection|subsubsection|paragraph)\*?\s*\{")
PDF_INCLUDE_PATTERN = re.compile(r"\\includepdf(?:\[[^]]*\])?\s*\{\s*([^}]+?)\s*\}")


@dataclass(frozen=True)
class IncludeReference:
    source: str
    target: str
    status: str


@dataclass(frozen=True)
class InspectionReport:
    paper_id: str
    arxiv_id: str
    archive_path: str
    root_file: str | None
    tex_file_count: int
    resolved_files: list[str]
    includes: list[IncludeReference]
    unresolved_includes: list[IncludeReference]
    cyclic_includes: list[IncludeReference]
    section_counts: dict[str, int]
    embedded_pdf_references: list[str]
    requires_pdf_fallback: bool
    extractable: bool


class SourceInspectionError(RuntimeError):
    """Raised when a source archive cannot be inspected."""


class LatexArchiveInspector:
    """Read a tar archive in place and map its LaTeX document tree."""

    def __init__(self, archive_path: str | Path) -> None:
        self.archive_path = Path(archive_path)
        self._sources = self._load_tex_sources()

    def inspect(self, paper: PaperManifest) -> InspectionReport:
        root_file = self._find_root_file()
        if root_file is None:
            return InspectionReport(
                paper_id=paper.id,
                arxiv_id=paper.arxiv_id or "",
                archive_path=str(self.archive_path),
                root_file=None,
                tex_file_count=len(self._sources),
                resolved_files=[],
                includes=[],
                unresolved_includes=[],
                cyclic_includes=[],
                section_counts={},
                embedded_pdf_references=[],
                requires_pdf_fallback=False,
                extractable=False,
            )

        resolved_files: list[str] = []
        includes: list[IncludeReference] = []
        unresolved_includes: list[IncludeReference] = []
        cyclic_includes: list[IncludeReference] = []
        section_counts: Counter[str] = Counter()
        embedded_pdf_references: list[str] = []

        def visit(file_name: str, ancestry: set[str]) -> None:
            if file_name in ancestry:
                return
            if file_name not in resolved_files:
                resolved_files.append(file_name)
                contents = _without_comments(self._sources[file_name])
                section_counts.update(SECTION_PATTERN.findall(contents))
                embedded_pdf_references.extend(PDF_INCLUDE_PATTERN.findall(contents))

            for requested_path in _find_includes(self._sources[file_name]):
                target = self._resolve_include(file_name, requested_path)
                if target is None:
                    reference = IncludeReference(file_name, requested_path, "unresolved")
                    includes.append(reference)
                    unresolved_includes.append(reference)
                elif target in ancestry or target == file_name:
                    reference = IncludeReference(file_name, target, "cyclic")
                    includes.append(reference)
                    cyclic_includes.append(reference)
                else:
                    includes.append(IncludeReference(file_name, target, "resolved"))
                    visit(target, ancestry | {file_name})

        visit(root_file, set())
        requires_pdf_fallback = bool(embedded_pdf_references) and not section_counts
        return InspectionReport(
            paper_id=paper.id,
            arxiv_id=paper.arxiv_id or "",
            archive_path=str(self.archive_path),
            root_file=root_file,
            tex_file_count=len(self._sources),
            resolved_files=resolved_files,
            includes=includes,
            unresolved_includes=unresolved_includes,
            cyclic_includes=cyclic_includes,
            section_counts=dict(sorted(section_counts.items())),
            embedded_pdf_references=embedded_pdf_references,
            requires_pdf_fallback=requires_pdf_fallback,
            extractable=not unresolved_includes and not cyclic_includes and not requires_pdf_fallback,
        )

    def _load_tex_sources(self) -> dict[str, str]:
        try:
            with tarfile.open(self.archive_path, mode="r:*") as archive:
                sources: dict[str, str] = {}
                for member in archive.getmembers():
                    normalized_name = _normalize_archive_path(member.name)
                    if not normalized_name or not member.isfile() or not normalized_name.lower().endswith(".tex"):
                        continue
                    file_object = archive.extractfile(member)
                    if file_object is None:
                        continue
                    sources[normalized_name] = file_object.read().decode("utf-8", errors="replace")
        except (OSError, tarfile.TarError) as exc:
            raise SourceInspectionError(f"Could not read {self.archive_path}: {exc}") from exc

        return sources

    def _find_root_file(self) -> str | None:
        roots = [
            file_name
            for file_name, contents in self._sources.items()
            if DOCUMENT_CLASS_PATTERN.search(_without_comments(contents))
        ]
        return sorted(roots, key=lambda file_name: (file_name.count("/"), file_name))[0] if roots else None

    def _resolve_include(self, source_file: str, requested_path: str) -> str | None:
        requested_path = requested_path.strip()
        if not requested_path or "\\" in requested_path:
            return None

        candidates = [requested_path]
        if not requested_path.lower().endswith(".tex"):
            candidates.append(f"{requested_path}.tex")

        source_directory = str(PurePosixPath(source_file).parent)
        for candidate in candidates:
            relative_candidate = _normalize_archive_path(posixpath.join(source_directory, candidate))
            if relative_candidate in self._sources:
                return relative_candidate
            root_candidate = _normalize_archive_path(candidate)
            if root_candidate in self._sources:
                return root_candidate
        return None


def inspect_sources(
    manifest: CorpusManifest,
    input_root: str | Path,
    output_root: str | Path,
    *,
    include_legacy: bool = False,
    dry_run: bool = False,
) -> list[InspectionReport]:
    """Inspect acquired archives and persist one report for each selected paper."""
    reports: list[InspectionReport] = []
    for paper in select_papers(manifest, include_legacy):
        archive_path = Path(input_root) / manifest.corpus_id / paper.id / "source.tar.gz"
        if not archive_path.is_file():
            raise SourceInspectionError(f"Missing source archive for {paper.id}: {archive_path}")

        report = LatexArchiveInspector(archive_path).inspect(paper)
        reports.append(report)
        if not dry_run:
            report_path = Path(output_root) / manifest.corpus_id / paper.id / "inspection.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8"
            )

    return reports


def _without_comments(contents: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", contents)


def _find_includes(contents: str) -> list[str]:
    return INCLUDE_PATTERN.findall(_without_comments(contents))


def _normalize_archive_path(path: str) -> str | None:
    normalized_path = posixpath.normpath(path).lstrip("/")
    if normalized_path in {"", "."} or normalized_path == ".." or normalized_path.startswith("../"):
        return None
    return normalized_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect acquired AURA LaTeX source archives.")
    parser.add_argument("manifest_path", type=Path, help="Path to a validated corpus manifest.")
    parser.add_argument("--input-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without writing reports.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest_path)
    try:
        reports = inspect_sources(
            manifest,
            args.input_root,
            args.output_root,
            include_legacy=args.include_legacy,
            dry_run=args.dry_run,
        )
    except SourceInspectionError as exc:
        print(exc)
        return 1

    for report in reports:
        state = "PDF_FALLBACK" if report.requires_pdf_fallback else "EXTRACTABLE" if report.extractable else "REVIEW"
        print(
            f"{state:12} {report.paper_id}: "
            f"root={report.root_file}, files={len(report.resolved_files)}, "
            f"unresolved={len(report.unresolved_includes)}, cycles={len(report.cyclic_includes)}"
        )
    return 0 if all(report.extractable for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
