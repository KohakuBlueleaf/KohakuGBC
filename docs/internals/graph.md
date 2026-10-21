# Records and graph resolution

`kohakugbc.graph` turns the records one image job collects into its GBC graph.

## Records

A record is one query result, the plain-data form of ml-gbc's `NodeInfo`
(`kohakugbc.graph.records`):

| Field | Holds |
|---|---|
| `ids` | `[entity_id, label, text]` of every entity the result answers for; the first is the vertex, the others joined it as the same object |
| `box` | `[left, top, right, bottom, confidence]` relative to the image; `None` is the whole image |
| `descs` | `[text, desc_label, [entity_id, ...]]`; the ids name the entities a relational answer is about |
| `children` | `[entity_id, label, text]` per phrase that names an entity; `text` becomes the edge label |

A relation record (first id labelled `relation`) names the vertex the question was asked on. A
record without descs gives no vertex. Records hold lists only, so a record equals its JSON round
trip.

## Resolution

The steps of ml-gbc's `node_infos_to_gbc_graph`, in order:

| Step | Result | Cost |
|---|---|---|
| ids | every joined id maps to its vertex; a numbered id (`cup_1`, by Python's `str.isdigit`) gathers under its stem, which points to its instances when there is no composition vertex | linear |
| relation split | one vertex per distinct set of entities an answer mentions (`[a\|b]`), its box the union of theirs, linked from the asked vertex by each entity id's last part | answers × entities |
| vertices | one per described record, descriptions labelled by vertex type, an out-edge per named phrase to every vertex its id maps to | linear in edges |
| in-edges | by source in id order, each source's out-edges in order, without duplicates | edges × in-degree |
| masks | per vertex, the others mostly inside it and those it is mostly inside (overlap over the inner area above the threshold, 0.85), in id order | quadratic in vertices |
| captions | the graph's original, short and detail captions from the first image vertex | linear |

The graph is the dict `GbcGraphFull.model_dump()` gives, key for key; ml-gbc's `GbcGraph` and
`GbcGraphFull` load it as written. Stats (`kohakugbc.graph.stats`): vertices, vertices by label,
edges, captions, BFS depth from the roots, roots, leaves.

## Two implementations

- `kohakugbc.graph.reference`: the Python reference, a literal port of ml-gbc's code.
- `kohakugbc._rs.resolve` (`src/kohakugbc-rust/graph/`): written from the same definition; it
  resolves and serialises without the GIL and returns the graph as JSON text with its stats.

`kohakugbc.graph.to_text` is the one the program uses, chosen once at import (Rust when built;
`KOHAKUGBC_RESOLVE=python|rust` overrides).

## How they are checked

`scripts/reference/mlgbc_cases.py` runs in an environment with ml-gbc and writes
`tests/graph/fixtures/mlgbc_cases.jsonl.gz`: for every image of recorded PoC runs and for
seeded generated record sets (joins, numbered instances with and without a composition vertex,
relations over joined and numbered entities, absent records, unknown and blank children, boxes
around the threshold, ids that only Python's `str.isdigit` reads as numbered), the records,
the digest of ml-gbc's graph and its counts, or that ml-gbc raised. `tests/graph` compares both
implementations against it by the digest of the canonical text (`kohakugbc.graph.digest`), and
shows that broken variants of the masks, the in-edge order and the numbered-id rule each fail
it.
