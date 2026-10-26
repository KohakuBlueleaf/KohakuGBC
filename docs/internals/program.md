# The GBC program

`kohakugbc/program/gbc.kir` is one image job: image, existing captions and tags, controls in;
the GBC graph (JSON text) and its counts out. `kohakugbc.nodes.library.build()` is its node
library `gbc`. Every value is passed and returned; no node mutates its inputs.

## Structure

| `@def` | Does |
|---|---|
| `ask` | a VLM request; a failed request answers with an empty reply, as GBC reads one |
| `detect` | the regions detection finds for named entities in a parent's box, one level down |
| `root` | the image vertex: GBC's image query, or existing captions (text, image and text, tags) |
| `expand` | one region: it joins a vertex seen before (`register`), or it is described and its parts detected |
| `prefill` | several regions in one request (`entity_batch` above 1) |
| `wave`, `depth_round` | one round: the frontier as it was when the round began, in waves that take what the node budget still needs |
| `coverage` | one more root sentence naming objects not covered yet |
| `relate` | relation and composition questions over the graph so far |
| `gbc` | the job: root, rounds until `next_step` says done, the relational questions, the graph; with `tagging`, the tags placed on it (`kohakugbc.nodes.tags`) |

Model nodes are transport only: `chat` (request in, reply text out), `detector` (patch and
request in, raw boxes out), `embedder` (name in, vector out), `tagger` (picture, crops and boxes
in, tag probabilities out). Request builders
(`kohakugbc.nodes.specs.chat`, `detect`) and reply parsers (`parse`) are plain nodes, so the same
requests and parsing run on any engine.

## Reproduction of the PoC

The nodes port the PoC's round engine (`gbcpoc.engine.runner`), its JSON reply layer and seeded
root, and ml-gbc's parsers, detection post-processing, same-object merge and relational
queries. Six PoC arms recorded as traces (`scripts/trace/poc_arms.py`: text, image and text, tags
and image-query roots; entity batch 4; leaf captions from the parent; 30 images each) replay
through the program under KIR's reference interpreter (`scripts/trace/replay_program.py`).

Where the program differs from the PoC, and why:

| Difference | PoC | Program |
|---|---|---|
| prompt text | prompt files read in the platform's locale encoding (cp950 on the recording machine): non-ASCII characters of the image-query system prompt garbled | prompt files read as UTF-8 |
| unseeded choices | relation count, phrase naming a set of children, composition root and wording drawn from unseeded generators | drawn from generators seeded by what is asked (`draw.rng`); the recorded runs use the same draws |
| relation sentences | repeats dropped through a `set`, order set by the process's hash seed | repeats dropped in first-seen order; the recorded runs use the same rule |
| `relational = "relation"` / `"composition"` | each keeps the other kind of question | each keeps its own kind |
| embeddings | names embedded when a same-object comparison needs them | the same: `embed_needs` names the embeddings `register` reads (overlapping vertices' names of a different singular form, up to the first equal one), each name embedded once per job; the same decisions |
| relational questions | asked after every round, answers cached by input | asked once at the end by default (`relate = "end"`); the final graph asks the same questions; `"every_round"` gives a graph after each round |
| region registration | after its query, under a per-image lock | before its query, in program order; the same order as the PoC's one-region-at-a-time rounds |

With these, every replayed job's graph equals the PoC's by digest, and every request the
program makes is one the PoC made: `tests/program` checks this on jobs cut from each arm
(`tests/fixtures/traces`), and that a broken frontier order changes graphs.
