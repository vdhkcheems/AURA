# Chunking and Figure-Linkage Contract v1

## Purpose

Define the stable processed-data contract that sits between normalized paper documents and later embedding/indexing work. The initial implementation creates text chunks only, but it must retain enough figure context for a separate visual-evidence pipeline to join later without reparsing every source archive.

## Scope

This contract applies to the normalized LaTeX papers in `data/processed/<corpus-id>/<paper-id>/normalized.json`.

It defines:

- Text-chunk shape and stable identifiers.
- Chunking boundaries and overlap rules.
- Figure metadata and chunk-to-figure links that the normalizer must eventually emit.
- Validation requirements before embeddings are generated.

It does not yet define image rendering, vision-model descriptions, figure embeddings, object storage, or UI behavior.

## Text Chunk Record

Chunks are emitted as JSON Lines at:

```text
data/processed/<corpus-id>/chunks.jsonl
```

Each record has this shape:

```json
{
  "schema_version": "1",
  "corpus_id": "ml-core-v1",
  "paper_id": "attention-is-all-you-need",
  "chunk_id": "attention-is-all-you-need::001::0001",
  "title": "Attention Is All You Need",
  "authors": ["Ashish Vaswani"],
  "year": 2017,
  "topics": ["transformers", "attention", "sequence-modeling"],
  "section_id": "001",
  "section_path": ["Abstract"],
  "block_range": {"start": 0, "end": 2},
  "text": "...",
  "character_count": 1840,
  "source_files": ["ms.tex"],
  "related_figure_ids": []
}
```

`chunk_id` is immutable once written. The middle component is the normalized section ID and the final component is a zero-padded ordinal within that section.
`block_range.start` is inclusive and `block_range.end` is exclusive.

## Chunking Rules

1. Chunk within a normalized section; do not join unrelated section paths.
2. Accumulate blocks in reading order. Do not split an equation or caption; split a paragraph only when it alone exceeds the maximum, preferring sentence and then whitespace boundaries.
3. Target 1,600–2,400 characters of textual content. A chunk may reach 3,200 characters when splitting would detach a short explanation from its equation, caption, or list.
4. Merge a trailing chunk shorter than 400 characters into its predecessor when both belong to the same section and the combined chunk stays below the maximum.
5. Add up to 250 characters of overlap using complete preceding paragraph blocks. Never duplicate equations or captions solely to create overlap.
6. Preserve equations in the chunk text using a clear fenced or delimited representation that remains faithful to the normalized LaTeX.
7. Keep captions with their current section and connect them to their figure IDs once figure metadata is available.
8. Include appendix content, with `"Appendix"` retained in the section path, so retrieval can distinguish it from main-paper claims.

## Figure Record (Reserved Contract)

When figure extraction is implemented, it writes records at:

```text
data/processed/<corpus-id>/figures.jsonl
```

```json
{
  "schema_version": "1",
  "corpus_id": "ml-core-v1",
  "paper_id": "attention-is-all-you-need",
  "figure_id": "attention-is-all-you-need::fig:model",
  "section_path": ["Model Architecture"],
  "label": "fig:model",
  "caption": "The Transformer model architecture.",
  "source_file": "ms.tex",
  "source_asset_path": ".../main-diagrams.pdf",
  "rendered_asset_key": null,
  "description": null,
  "referencing_chunk_ids": []
}
```

The figure ID uses the LaTeX label when one is available; otherwise use a stable source-order ordinal, such as `paper-id::figure-0003`. `source_asset_path` identifies the immutable source asset. `rendered_asset_key` is added only after a browser-ready derivative exists.

## Linking Rules

- A chunk links to a figure when it contains the figure caption, includes its `\includegraphics` call, or refers to the figure label in its source blocks.
- A figure lists every linked chunk in `referencing_chunk_ids` after chunk generation.
- Figure links are optional at first: a text chunk remains valid when a paper has no extractable source asset.
- Tables use the same future linkage mechanism but will receive their own `table` record type because structured extraction is preferable to treating data tables as images.

## Validation Requirements

Before embedding:

- Every chunk must have a unique `chunk_id`, non-empty text, section path, and source file list.
- Every chunk's block range must point to existing normalized blocks.
- Character counts must match emitted text.
- Every linked figure ID must exist in `figures.jsonl` once figure extraction is enabled.
- No rendered asset may replace or mutate the corresponding raw source asset.
- The chunker must report per-paper chunk counts, character counts, equation counts, caption counts, and unlinked figure-reference warnings.

## Deferred Decisions

- Offline versus on-demand vision descriptions.
- Figure-description embedding model and Qdrant collection design.
- Production object-storage provider and access policy.
- Rendering strategy for TikZ/PGFPlots figures and PDF-page fallback crops.
