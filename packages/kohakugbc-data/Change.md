# kohakugbc-data changes

## 2026-11-06 update to 0.2.0

- `kohakugbc_data.dataset`: an `avif` image codec (member `<key>.avif`, index codec 6, color
  flags 0). An AVIF file decodes with Pillow's built-in libavif on every platform, so
  `GbcDataset(image="pil")` reads such shards without KohakuClip. The format guide states that a
  dataset may store resized variants of the captioned image (boxes are fractions of the image).

## 2026-10-28 update to 0.1.0

The first release: KohakuGBC's data side as its own package.

- `GbcGraph`: GBC graphs held natively, from / to JSON and typed msgpack (+zstd), rendered as
  prompt text; `PromptDag`: prompt text parsed back; `resolve`: caption records into a graph.
- `kohakugbc_data.prompt`: the Python prompt renderers (`from_gbc`, `render`, `parse`,
  `to_moss`, detail levels).
- `kohakugbc_data.dataset`: the GBC dataset format (KohakuClip image shards, KohakuVault graphs
  and metadata, self-contained shards) with `GbcDataset` and `DatasetWriter`.
- One abi3 wheel per platform (Linux x86_64 / aarch64, Windows x64, macOS x86_64 / arm64) for
  CPython 3.11+.
