"""Load and validate AURA corpus manifests."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
VALID_SOURCE_PREFERENCES = {"arxiv_latex", "pdf"}
VALID_VECTOR_DATABASES = {"qdrant"}
VALID_DISTANCES = {"cosine", "dot", "euclidean"}


class ManifestValidationError(ValueError):
    """Raised when a corpus manifest is not valid."""


@dataclass(frozen=True)
class SourcePolicy:
    preferred_source: str
    fallback_source: str
    notes: str = ""


@dataclass(frozen=True)
class EmbeddingPolicy:
    initial_model: str
    vector_database: str
    distance: str


@dataclass(frozen=True)
class PaperManifest:
    id: str
    title: str
    authors: list[str]
    year: int
    arxiv_id: str | None
    topics: list[str]
    priority: int
    source_preference: str
    status: str


@dataclass(frozen=True)
class CorpusManifest:
    corpus_id: str
    name: str
    description: str
    status: str
    source_policy: SourcePolicy
    embedding_policy: EmbeddingPolicy
    papers: list[PaperManifest]

    @property
    def paper_count(self) -> int:
        return len(self.papers)

    @property
    def planned_count(self) -> int:
        return sum(1 for paper in self.papers if paper.status == "planned")

    @property
    def preindexed_count(self) -> int:
        return sum(1 for paper in self.papers if paper.status == "already_indexed")


def load_manifest(path: str | Path) -> CorpusManifest:
    """Load and validate a corpus manifest JSON file."""
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"Invalid JSON in {manifest_path}: {exc}") from exc
    except OSError as exc:
        raise ManifestValidationError(f"Could not read {manifest_path}: {exc}") from exc

    errors = validate_manifest_data(raw)
    if errors:
        joined_errors = "\n".join(f"- {error}" for error in errors)
        raise ManifestValidationError(f"Manifest validation failed:\n{joined_errors}")

    return _parse_manifest(raw)


def validate_manifest_data(raw: Any) -> list[str]:
    """Return all validation errors for a raw manifest object."""
    errors: list[str] = []

    if not isinstance(raw, dict):
        return ["manifest root must be a JSON object"]

    _require_string(raw, "corpus_id", errors)
    _require_string(raw, "name", errors)
    _require_string(raw, "description", errors)
    _require_string(raw, "status", errors)

    source_policy = raw.get("source_policy")
    if not isinstance(source_policy, dict):
        errors.append("source_policy must be an object")
    else:
        _validate_source_policy(source_policy, errors)

    embedding_policy = raw.get("embedding_policy")
    if not isinstance(embedding_policy, dict):
        errors.append("embedding_policy must be an object")
    else:
        _validate_embedding_policy(embedding_policy, errors)

    papers = raw.get("papers")
    if not isinstance(papers, list):
        errors.append("papers must be a list")
    elif not papers:
        errors.append("papers must contain at least one paper")
    else:
        _validate_papers(papers, errors)

    return errors


def _validate_source_policy(source_policy: dict[str, Any], errors: list[str]) -> None:
    preferred_source = _require_string(source_policy, "preferred_source", errors, "source_policy")
    fallback_source = _require_string(source_policy, "fallback_source", errors, "source_policy")

    if preferred_source and preferred_source not in VALID_SOURCE_PREFERENCES:
        errors.append(
            f"source_policy.preferred_source must be one of {sorted(VALID_SOURCE_PREFERENCES)}"
        )
    if fallback_source and fallback_source not in VALID_SOURCE_PREFERENCES:
        errors.append(
            f"source_policy.fallback_source must be one of {sorted(VALID_SOURCE_PREFERENCES)}"
        )


def _validate_embedding_policy(embedding_policy: dict[str, Any], errors: list[str]) -> None:
    vector_database = _require_string(
        embedding_policy, "vector_database", errors, "embedding_policy"
    )
    distance = _require_string(embedding_policy, "distance", errors, "embedding_policy")
    _require_string(embedding_policy, "initial_model", errors, "embedding_policy")

    if vector_database and vector_database not in VALID_VECTOR_DATABASES:
        errors.append(
            f"embedding_policy.vector_database must be one of {sorted(VALID_VECTOR_DATABASES)}"
        )
    if distance and distance not in VALID_DISTANCES:
        errors.append(f"embedding_policy.distance must be one of {sorted(VALID_DISTANCES)}")


def _validate_papers(papers: list[Any], errors: list[str]) -> None:
    seen_ids: set[str] = set()

    for index, paper in enumerate(papers):
        prefix = f"papers[{index}]"
        if not isinstance(paper, dict):
            errors.append(f"{prefix} must be an object")
            continue

        paper_id = _require_string(paper, "id", errors, prefix)
        _require_string(paper, "title", errors, prefix)
        _require_string(paper, "status", errors, prefix)
        source_preference = _require_string(paper, "source_preference", errors, prefix)

        if paper_id:
            if paper_id in seen_ids:
                errors.append(f"{prefix}.id duplicates paper id '{paper_id}'")
            seen_ids.add(paper_id)

        authors = paper.get("authors")
        if not _is_non_empty_string_list(authors):
            errors.append(f"{prefix}.authors must be a non-empty list of strings")

        topics = paper.get("topics")
        if not _is_non_empty_string_list(topics):
            errors.append(f"{prefix}.topics must be a non-empty list of strings")

        year = paper.get("year")
        if not isinstance(year, int) or year < 1900:
            errors.append(f"{prefix}.year must be an integer >= 1900")

        priority = paper.get("priority")
        if not isinstance(priority, int) or priority < 1:
            errors.append(f"{prefix}.priority must be a positive integer")

        if source_preference and source_preference not in VALID_SOURCE_PREFERENCES:
            errors.append(
                f"{prefix}.source_preference must be one of {sorted(VALID_SOURCE_PREFERENCES)}"
            )

        arxiv_id = paper.get("arxiv_id")
        if source_preference == "arxiv_latex":
            if not isinstance(arxiv_id, str) or not arxiv_id.strip():
                errors.append(f"{prefix}.arxiv_id is required for arxiv_latex sources")
            elif not ARXIV_ID_PATTERN.match(arxiv_id):
                errors.append(f"{prefix}.arxiv_id does not look like a valid arXiv ID")


def _parse_manifest(raw: dict[str, Any]) -> CorpusManifest:
    source_policy = SourcePolicy(**raw["source_policy"])
    embedding_policy = EmbeddingPolicy(**raw["embedding_policy"])
    papers = [PaperManifest(**paper) for paper in raw["papers"]]

    return CorpusManifest(
        corpus_id=raw["corpus_id"],
        name=raw["name"],
        description=raw["description"],
        status=raw["status"],
        source_policy=source_policy,
        embedding_policy=embedding_policy,
        papers=papers,
    )


def _require_string(
    data: dict[str, Any],
    key: str,
    errors: list[str],
    prefix: str | None = None,
) -> str | None:
    value = data.get(key)
    label = f"{prefix}.{key}" if prefix else key

    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty string")
        return None

    return value


def _is_non_empty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an AURA corpus manifest.")
    parser.add_argument("manifest_path", type=Path, help="Path to a corpus manifest JSON file.")
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest_path)
    except ManifestValidationError as exc:
        print(exc)
        return 1

    print(f"Manifest valid: {manifest.corpus_id}")
    print(f"Papers: {manifest.paper_count}")
    print(f"Planned: {manifest.planned_count}")
    print(f"Already indexed: {manifest.preindexed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
