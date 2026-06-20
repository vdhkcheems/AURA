"""Acquire and validate arXiv source archives for a curated AURA corpus."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .manifest import CorpusManifest, PaperManifest, load_manifest


ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
USER_AGENT = "AURA research corpus builder/0.1 (contact: repository maintainer)"


class SourceAcquisitionError(RuntimeError):
    """Raised when a paper's source archive cannot be acquired or validated."""


@dataclass(frozen=True)
class AcquisitionResult:
    paper_id: str
    arxiv_id: str
    outcome: str
    archive_path: Path
    message: str


def select_papers(manifest: CorpusManifest, include_legacy: bool) -> list[PaperManifest]:
    """Return LaTeX-source papers in deterministic ingestion order."""
    allowed_statuses = {"planned"}
    if include_legacy:
        allowed_statuses.add("already_indexed_legacy")

    return sorted(
        (
            paper
            for paper in manifest.papers
            if paper.status in allowed_statuses
            and paper.source_preference == "arxiv_latex"
            and paper.arxiv_id
        ),
        key=lambda paper: (paper.priority, paper.id),
    )


def acquire_sources(
    manifest: CorpusManifest,
    output_root: str | Path,
    *,
    include_legacy: bool = False,
    force: bool = False,
    dry_run: bool = False,
    opener: Callable[..., object] = urlopen,
) -> list[AcquisitionResult]:
    """Download and validate source archives for eligible manifest papers."""
    corpus_root = Path(output_root) / manifest.corpus_id
    results: list[AcquisitionResult] = []

    for paper in select_papers(manifest, include_legacy):
        assert paper.arxiv_id is not None
        paper_root = corpus_root / paper.id
        archive_path = paper_root / "source.tar.gz"

        if dry_run:
            results.append(
                AcquisitionResult(
                    paper.id,
                    paper.arxiv_id,
                    "planned",
                    archive_path,
                    f"Would download {ARXIV_EPRINT_URL.format(arxiv_id=paper.arxiv_id)}",
                )
            )
            continue

        if archive_path.exists() and not force:
            try:
                validate_latex_archive(archive_path)
            except SourceAcquisitionError as exc:
                archive_path.unlink()
            else:
                results.append(
                    _record_result(
                        paper,
                        paper_root,
                        "skipped",
                        archive_path,
                        "Existing valid source archive retained.",
                    )
                )
            continue

        try:
            download_source_archive(paper.arxiv_id, archive_path, opener=opener)
            validate_latex_archive(archive_path)
        except SourceAcquisitionError as exc:
            archive_path.unlink(missing_ok=True)
            results.append(_record_failure(paper, paper_root, archive_path, str(exc)))
        else:
            results.append(
                _record_result(
                    paper,
                    paper_root,
                    "downloaded",
                    archive_path,
                    "Downloaded and validated arXiv source archive.",
                )
            )

    return results


def download_source_archive(
    arxiv_id: str,
    archive_path: Path,
    *,
    opener: Callable[..., object] = urlopen,
) -> None:
    """Download an arXiv e-print archive atomically to ``archive_path``."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id), headers={"User-Agent": USER_AGENT}
    )

    try:
        with opener(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                dir=archive_path.parent, prefix=".source-", delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                shutil.copyfileobj(response, temporary_file)
    except HTTPError as exc:
        raise SourceAcquisitionError(f"arXiv returned HTTP {exc.code} for {arxiv_id}") from exc
    except URLError as exc:
        raise SourceAcquisitionError(f"Could not reach arXiv for {arxiv_id}: {exc.reason}") from exc
    except OSError as exc:
        raise SourceAcquisitionError(f"Could not save source archive for {arxiv_id}: {exc}") from exc

    try:
        temporary_path.replace(archive_path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise SourceAcquisitionError(f"Could not finalize source archive for {arxiv_id}: {exc}") from exc


def validate_latex_archive(archive_path: Path) -> None:
    """Ensure an archive is readable and contains at least one LaTeX source file."""
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise SourceAcquisitionError(f"Downloaded file is not a readable tar archive: {exc}") from exc

    if not members:
        raise SourceAcquisitionError("Downloaded source archive is empty.")
    if not any(member.isfile() and member.name.lower().endswith(".tex") for member in members):
        raise SourceAcquisitionError("Source archive contains no .tex files.")


def _record_failure(
    paper: PaperManifest, paper_root: Path, archive_path: Path, message: str
) -> AcquisitionResult:
    return _record_result(paper, paper_root, "failed", archive_path, message)


def _record_result(
    paper: PaperManifest,
    paper_root: Path,
    outcome: str,
    archive_path: Path,
    message: str,
) -> AcquisitionResult:
    result = AcquisitionResult(paper.id, paper.arxiv_id or "", outcome, archive_path, message)
    paper_root.mkdir(parents=True, exist_ok=True)
    (paper_root / "acquisition.json").write_text(
        json.dumps(
            {
                **asdict(result),
                "archive_path": str(archive_path),
                "acquired_at": datetime.now(UTC).isoformat(),
                "source_url": ARXIV_EPRINT_URL.format(arxiv_id=paper.arxiv_id),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire arXiv LaTeX source archives for AURA.")
    parser.add_argument("manifest_path", type=Path, help="Path to a validated corpus manifest.")
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/raw"), help="Directory for raw corpus files."
    )
    parser.add_argument(
        "--include-legacy", action="store_true", help="Also acquire papers marked already_indexed_legacy."
    )
    parser.add_argument("--force", action="store_true", help="Replace existing source archives.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned downloads without writing files.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest_path)
    results = acquire_sources(
        manifest,
        args.output_root,
        include_legacy=args.include_legacy,
        force=args.force,
        dry_run=args.dry_run,
    )
    for result in results:
        print(f"{result.outcome.upper():9} {result.paper_id}: {result.message}")

    failures = sum(result.outcome == "failed" for result in results)
    print(f"Processed {len(results)} paper(s); {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
