# Graph store and graph prompts

Two native parts in `kohakugbc._rs`, each matched against a Python reference:

- `GbcGraph`: one answer line (or bare graph) held in Rust, read from JSON or msgpack and
  written as either.
- `PromptDag` and `GbcGraph.prompt`: the graph as prompt text and back, the Rust port of
  `kohakugbc.graph.prompt`.

## The stored record

`src/kohakugbc-rust/store/record.rs`. msgpack carries it as positional arrays (`rmp-serde`, the
codec KohakuVault's `DataPacker("msgpack")` uses) inside a zstd frame: `to_msgpack()` writes
level 3, `to_msgpack(zstd=None)` plain msgpack, and `from_msgpack` reads either (it tells them
apart by the zstd magic).

| field | holds |
|---|---|
| `format` | the layout version (1); a reader refuses another |
| `head` | the answer's other top-level keys (`key`, `ok`, `stats`, `failure`, ...) as plain values, in order |
| `graph_at` | where `graph` sits among them; none for a bare graph |
| `graph` | the typed graph |
| `raw` | the graph value itself, when the typed form cannot carry it |

The typed graph links by index:

- **Vertices** are a list. Mask lists name vertices by index.
- **Out-edges** are `(target index, text)`; the source is the owning vertex.
- **In-edges** are indices into the graph's out-edges in vertex order. An in-edge is always some
  out-edge, so it is stored once.
- **Descs** are `(form, label, text)`.
  - `short` is `{label, text}`.
  - `full` is the ml-gbc caption whose `full_label` is `<desc label>-<vertex label>` and whose
    statistics, clip and toxicity scores are null.
  - Any other desc is kept whole.
- **Tag names and categories** are one string table per graph. A score list whose scores all
  have at most four decimals is stored as `u16` steps of 1e-4. Any other list is stored as `f64`.
- **Graph tags** are `image`, `categories`, `regions` and `crops`.

**Lossless by construction.** `from_json` builds the typed graph and keeps it only if
converting it back gives the same value. Otherwise the graph is stored raw: an integer where a
float was, an unknown key, a missing vertex.

## Prompt text

The format, written by `render` and read by `parse` in both implementations:

```
<gbc bbox=yxyx space=abs_centralized canvas=1360x768>
# [image] caption
tags: a, b
- [woman] [-374,-189,378,542] caption
  tags: c
  - [hair] [-288,87,375,542] caption
- [plate] [306,-133,378,48] caption
- relation [plate] [woman]: caption
```

Placement and boxes:

- **Handles:** a handle written in several places is one vertex. Each place is one of its parent
  edges.
- **Content placement:** a vertex's caption, box and tags sit at one of those places, the
  `first` or the `last`.
- **Stored boxes:** `(left, top, right, bottom)` in [0, 1].
- **Written boxes:** four integers in the header's space. `abs_centralized` is canvas pixels
  less `W // 2` / `H // 2` (MOSS-Image's layout space). `relative` is `[0, scale]`.
- **Order:** `yxyx` (MOSS-Image's model-facing order, the default) or `xyxy`.
- **Rounding:** `enclose` (default): start edges down, end edges up, after `round(., 9)`, as
  MOSS-Image's own `skills/moss-image-prompt-enhancer/scripts/compile_prompt.py`; `nearest`:
  half to even.

`kohakugbc.graph.prompt.to_moss` writes the MOSS-Image Layout draft (`summary` with `meta`,
`canvas`, `caption`; flat `items` with `label`, `caption`, `tags`, `bbox`; `abs` / `xyxy`
pixels), the input of that script, which compiles it into the one-line model prompt. On 360
graphs (4,332 items) the compiled boxes equal the boxes the GBC text writes for the same
vertices, in Python and Rust.

## Detail levels

`kohakugbc.graph.prompt.detail` shows each vertex at a sampled level, a child never above its
parents:

| level | shows |
|---|---|
| 6 | long caption + tags + box |
| 5 | short caption + tags + box |
| 4 | tags + box, or long caption + box |
| 3 | short caption + box |
| 2 | handle + box |
| 1 | handle |

`sample_levels` walks the DAG parents first; each vertex starts at its parents' lowest level
and steps down one level per failed `keep` draw (0.6 by default; the root at 6 unless given).
`apply_levels` builds the DAG those levels show. A short caption is the vertex's `short` desc,
else the first sentence of its long caption. No vertex is dropped and relation lines stay.
On 100 graphs x 50 seeds: no child is above a parent, each vertex shows exactly its level's
parts, and every text parses back to the same tree.

The Rust side reads text by Python's rules (`str.isspace`, `strip`, `splitlines`), so the same
text gives the same DAG in both implementations.

## Measured

1,506 distinct answer lines and graphs, from every run kept locally (demo, scenery, r51 and the
danbooru 27B tier samples). Windows, one Python thread, best of 5 passes. Checked first:

- JSON → `GbcGraph` → JSON, and → msgpack (plain, zstd 3 / 9 / 19) → JSON, both give each
  line's own value.
- Three values the typed form cannot carry come back the same, kept raw.
- The Rust prompt text equals the Python reference's on 15,060 renders: 10 variants of coords
  (both roundings), placement and options per graph.
- Rust `parse` equals Python `parse` on all of them, and on 11 hand-written prompts (8 of them
  rejected by both).

| form | mean bytes per line | vs JSON |
|---|---|---|
| JSON | 25,578 | 1.0x |
| JSON + zlib 6 | 4,139 | 6.2x |
| KohakuVault msgpack of the dict | 21,509 | 1.2x |
| typed msgpack | 8,798 | 2.9x |
| typed msgpack + zstd 3 | 3,357 | 7.6x |
| typed msgpack + zstd 19 | 3,157 | 8.1x |

| operation | us per line |
|---|---|
| `json.loads` (Python dict) | 120.7 |
| KohakuVault unpack to a dict | 163.9 |
| `GbcGraph.from_msgpack` | 13.7 |
| `GbcGraph.from_msgpack`, zstd 3 | 23.2 |
| `GbcGraph.to_msgpack` | 7.6 |
| `GbcGraph.to_msgpack`, zstd 3 | 41.2 |
| `GbcGraph.from_json` (parse, convert, check round trip) | 260.6 |
| `GbcGraph.to_json` | 106.8 |

| prompt | us per graph |
|---|---|
| Python `from_gbc` + `render` | 278.4 |
| Rust `GbcGraph.prompt` | 55.6 |
| Rust, from msgpack bytes to text | 67.2 |
| Python `parse` | 118.6 |
| Rust `PromptDag.parse` | 51.5 |
