# Traces

A trace records every model call of a run, keyed by what was asked
(`kohakugbc.nodes.specs.trace`). The program track replays traces instead of running models:
under KIR's reference interpreter (`kohakugbc.nodes.specs.replay.ReplayModels`) and on the
machine's pseudo engines.

## Format

A folder: `meta.json` (source, settings, `image_root`, relative to the folder when relative) and
one JSONL file per kind of line.

| File | One line per | Fields |
|---|---|---|
| `jobs.jsonl` | image job | `key`, `image`, `inputs` (captions, tags, controls), the `records` and `graph` the source produced, `rounds` |
| `vlm.jsonl` | VLM request | `key`, `request` (system, query, image sizes, schema, max tokens, sampling), `reply`, `error`, latency, time to first token, usage |
| `detect.jsonl` | detector call | `key`, `size` (crop pixels), `params` (per text: NMS threshold, high-confidence tolerance, area cap, top k), `raw` (pixel boxes, scores, text index), `selected` (kept boxes per text) |
| `embed.jsonl` | name embedded | `text`, `vector` (float32, base64) |
| `same.jsonl` | same-object decision | `a`, `b`, `same`, `score` |

## Keys

A key is the canonical JSON of what a request means: the node, the prompt set, the image job,
and what is asked about.

| Node | Key fields |
|---|---|
| `image_query` | image |
| `seed_elements` | image, source (`text` or `image`), tag hint |
| `coverage` | image, the root's element names |
| `entity` | image, entity id, box |
| `entity_batch` | image, `[entity id, box]` per region |
| `relation` | image, asked vertex, its box, `[entity id, text]` per entity |
| `composition` | image, the composition's ids (the asked one first), `[part id, text]` and box per part |
| `detect` | image, parent box (None: whole image), texts |

Boxes enter keys as their four coordinates. Relation and composition keys hold the boxes because
boxes widen as regions join, and composition keys hold every id because a second phrase may come
to name the same parts: either way the question asked again is a different request.
`Trace.conflicts` lists keys recorded twice with different replies; a complete key gives none.

## Recording the PoC

`scripts/trace/record_poc.py` runs the PoC's `run_iterative.py` in the PoC's environment and
writes its trace; `scripts/trace/poc_arms.py` records the arms the program is checked against.
The recorder replaces the PoC's unseeded choices with the program's rules
(`kohakugbc.nodes.specs.draw`):

| Choice | Draw |
|---|---|
| relations asked for in a relation query | `rng("relation-count", image, vertex).randint(2, n)` |
| the phrase naming a set of children | `rng("relational-text", image, vertex, *phrases).choice(phrases)` |
| composition root criterion, then per tree edge the diagonal wording | `rng("composition", image, composition id)`: `choice`, then `random()` per edge in tree order |
| relation sentences | repeats dropped in first-seen order |

`scripts/trace/subset.py` cuts jobs with their images into `tests/fixtures/traces`.
