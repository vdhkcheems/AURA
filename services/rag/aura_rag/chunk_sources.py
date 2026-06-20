"""Create stable, section-aware retrieval chunks from normalized AURA papers."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import CorpusManifest, load_manifest


SCHEMA_VERSION = "1"


class ChunkingError(RuntimeError):
    """Raised when normalized documents cannot be chunked safely."""


@dataclass(frozen=True)
class ChunkingConfig:
    target_characters: int = 2_400
    minimum_chunk_characters: int = 1_600
    maximum_chunk_characters: int = 3_200
    minimum_trailing_characters: int = 400
    overlap_characters: int = 250


@dataclass(frozen=True)
class ChunkingResult:
    paper_id: str
    chunk_count: int
    character_count: int
    equation_count: int
    caption_count: int


@dataclass(frozen=True)
class _Block:
    index: int
    block_type: str
    text: str
    source_file: str


def chunk_document(document: dict[str, Any], config: ChunkingConfig = ChunkingConfig()) -> list[dict[str, Any]]:
    """Split one normalized document into bounded chunks without splitting blocks."""
    _validate_config(config)
    _validate_document(document)
    chunks: list[dict[str, Any]] = []

    for section in document["sections"]:
        blocks = _section_blocks(section, config.maximum_chunk_characters)
        if not blocks:
            continue
        section_chunks = _chunk_section(blocks, config)
        for ordinal, chunk_blocks in enumerate(section_chunks, start=1):
            text = _join_blocks(chunk_blocks)
            chunks.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "corpus_id": document["corpus_id"],
                    "paper_id": document["paper_id"],
                    "chunk_id": f"{document['paper_id']}::{section['id']}::{ordinal:04d}",
                    "title": document["title"],
                    "authors": document["authors"],
                    "year": document["year"],
                    "topics": document["topics"],
                    "section_id": section["id"],
                    "section_path": section["path"],
                    "block_range": {
                        "start": min(block.index for block in chunk_blocks),
                        "end": max(block.index for block in chunk_blocks) + 1,
                    },
                    "text": text,
                    "character_count": len(text),
                    "source_files": _ordered_unique(block.source_file for block in chunk_blocks),
                    "related_figure_ids": [],
                }
            )
    return chunks


def chunk_corpus(
    manifest: CorpusManifest,
    processed_root: str | Path,
    *,
    config: ChunkingConfig = ChunkingConfig(),
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], list[ChunkingResult]]:
    """Chunk every normalized document available for a manifest's papers."""
    processed_root = Path(processed_root)
    all_chunks: list[dict[str, Any]] = []
    results: list[ChunkingResult] = []

    for paper in sorted(manifest.papers, key=lambda item: (item.priority, item.id)):
        document_path = processed_root / manifest.corpus_id / paper.id / "normalized.json"
        if not document_path.is_file():
            continue
        try:
            document = json.loads(document_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ChunkingError(f"Could not read normalized document {document_path}: {exc}") from exc
        chunks = chunk_document(document, config)
        _validate_chunks(chunks, paper.id)
        all_chunks.extend(chunks)
        source_blocks = [block for section in document["sections"] for block in section["blocks"]]
        results.append(
            ChunkingResult(
                paper.id,
                len(chunks),
                sum(chunk["character_count"] for chunk in chunks),
                sum(block["type"] == "equation" for block in source_blocks),
                sum(block["type"] == "caption" for block in source_blocks),
            )
        )

    if not all_chunks:
        raise ChunkingError(f"No normalized documents found under {processed_root / manifest.corpus_id}")
    _validate_chunks(all_chunks)
    if not dry_run:
        output_path = processed_root / manifest.corpus_id / "chunks.jsonl"
        output_path.write_text(
            "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in all_chunks),
            encoding="utf-8",
        )
    return all_chunks, results


def _chunk_section(blocks: list[_Block], config: ChunkingConfig) -> list[list[_Block]]:
    chunks: list[list[_Block]] = []
    current: list[_Block] = []

    for block in blocks:
        candidate = [*current, block]
        candidate_length = len(_join_blocks(candidate))
        should_flush = (
            current
            and (
                candidate_length > config.maximum_chunk_characters
                or (
                    candidate_length > config.target_characters
                    and len(_join_blocks(current)) >= config.minimum_chunk_characters
                )
            )
        )
        if should_flush:
            chunks.append(current)
            current = [*_overlap_blocks(current, config.overlap_characters), block]
        else:
            current = candidate

        if len(_join_blocks(current)) > config.maximum_chunk_characters and len(current) == 1:
            # A single paragraph/equation may be large; preserving it is safer than
            # breaking source structure. The oversized chunk remains observable via
            # its character_count for later quality tuning.
            chunks.append(current)
            current = []

    if current:
        chunks.append(current)
    return _merge_short_tail(chunks, config)


def _merge_short_tail(chunks: list[list[_Block]], config: ChunkingConfig) -> list[list[_Block]]:
    if len(chunks) < 2 or len(_join_blocks(chunks[-1])) >= config.minimum_trailing_characters:
        return chunks
    previous, tail = chunks[-2], chunks[-1]
    overlap_length = 0
    for candidate_length in range(1, min(len(previous), len(tail)) + 1):
        if previous[-candidate_length:] == tail[:candidate_length]:
            overlap_length = candidate_length
    tail_new_blocks = tail[overlap_length:]
    merged = [*previous, *tail_new_blocks]
    if len(_join_blocks(merged)) <= config.maximum_chunk_characters:
        return [*chunks[:-2], merged]
    return chunks


def _overlap_blocks(blocks: list[_Block], limit: int) -> list[_Block]:
    overlap: list[_Block] = []
    for block in reversed(blocks):
        if block.block_type != "paragraph":
            break
        candidate = [block, *overlap]
        if len(_join_blocks(candidate)) > limit:
            break
        overlap = candidate
    return overlap


def _section_blocks(section: dict[str, Any], maximum_characters: int) -> list[_Block]:
    blocks: list[_Block] = []
    for index, block in enumerate(section["blocks"]):
        rendered = _render_block(block)
        if rendered:
            if block["type"] == "paragraph":
                blocks.extend(
                    _Block(index, block["type"], part, block["source_file"])
                    for part in _split_long_paragraph(rendered, maximum_characters)
                )
            else:
                blocks.append(_Block(index, block["type"], rendered, block["source_file"]))
    return blocks


def _split_long_paragraph(text: str, maximum_characters: int) -> list[str]:
    """Split only oversized prose, preferring sentence boundaries.

    A normalized paragraph can contain an entire source table or appendix.  It
    is safe to split prose for retrieval, but equations and captions remain
    atomic blocks elsewhere in the pipeline.
    """
    if len(text) <= maximum_characters:
        return [text]
    parts: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not sentence:
            continue
        if len(sentence) > maximum_characters:
            if current:
                parts.append(current)
                current = ""
            parts.extend(_split_at_whitespace(sentence, maximum_characters))
        elif current and len(current) + 1 + len(sentence) > maximum_characters:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def _split_at_whitespace(text: str, maximum_characters: int) -> list[str]:
    parts: list[str] = []
    remaining = text
    while len(remaining) > maximum_characters:
        split_at = remaining.rfind(" ", 0, maximum_characters + 1)
        if split_at <= 0:
            split_at = maximum_characters
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _render_block(block: dict[str, Any]) -> str:
    if block["type"] == "paragraph":
        return block["text"]
    if block["type"] == "equation":
        return f"$$\n{block['latex']}\n$$"
    if block["type"] == "caption":
        return f"Figure: {block['text']}"
    raise ChunkingError(f"Unsupported normalized block type: {block['type']}")


def _join_blocks(blocks: list[_Block]) -> str:
    return "\n\n".join(block.text for block in blocks)


def _ordered_unique(items: Any) -> list[str]:
    return list(dict.fromkeys(items))


def _validate_config(config: ChunkingConfig) -> None:
    if not 0 < config.minimum_trailing_characters <= config.minimum_chunk_characters:
        raise ChunkingError("minimum trailing characters must be positive and no greater than minimum chunk characters")
    if not config.minimum_chunk_characters <= config.target_characters <= config.maximum_chunk_characters:
        raise ChunkingError("chunk character limits must satisfy minimum <= target <= maximum")
    if config.overlap_characters < 0:
        raise ChunkingError("overlap characters cannot be negative")


def _validate_document(document: dict[str, Any]) -> None:
    required = {"corpus_id", "paper_id", "title", "authors", "year", "topics", "sections"}
    missing = required - document.keys()
    if missing:
        raise ChunkingError(f"Normalized document is missing fields: {sorted(missing)}")


def _validate_chunks(chunks: list[dict[str, Any]], paper_id: str | None = None) -> None:
    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ChunkingError(f"Duplicate chunk IDs found{f' for {paper_id}' if paper_id else ''}")
    for chunk in chunks:
        if not chunk["text"]:
            raise ChunkingError(f"Empty chunk text: {chunk['chunk_id']}")
        if chunk["character_count"] != len(chunk["text"]):
            raise ChunkingError(f"Incorrect character count: {chunk['chunk_id']}")
        if not chunk["section_path"] or not chunk["source_files"]:
            raise ChunkingError(f"Missing provenance: {chunk['chunk_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create section-aware chunks from AURA normalized documents.")
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest_path)
    try:
        chunks, results = chunk_corpus(manifest, args.processed_root, dry_run=args.dry_run)
    except ChunkingError as exc:
        print(exc)
        return 1
    for result in results:
        print(
            f"CHUNKED {result.paper_id}: chunks={result.chunk_count}, chars={result.character_count}, "
            f"equations={result.equation_count}, captions={result.caption_count}"
        )
    print(f"Wrote {len(chunks)} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
