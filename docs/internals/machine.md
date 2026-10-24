# The GBC machine

`kohakugbc.machine.build(*layers)` builds the GBC program on Togeari and returns
`(machine, source, sink)`, Togeari's machine-module contract (`kohakugbc/machine/module.py` is
the module form, configured by `KOHAKUGBC_CONFIG`).

## Configuration

Layers are dicts or Python files whose lower-case module-level names form one
(`kohakugbc.machine.config`); later layers merge over earlier ones, dicts key by key.

| Name | Holds |
|---|---|
| `engines` | per slot (`vlm`, `detect`, `embed`, `cpu`): `entry` and the entry's settings |
| `machine` | `vlm_waiting`, `max_jobs` (admission), `window`, `start_timeout_s` (a unit's start, 600), `record`, `store` (`"memory"`), `report` (a RunReport folder), `waits` (`{"every_s", "limit"}`: why jobs wait, sampled into the report's `waits.jsonl`, `kohakugbc.machine.waits`; about 180 ms of dispatcher time per sample on the RTX 5090 machine), `knobs` (other Togeari knobs) |
| `source` | `{"kind": "folder" \| "jsonl" \| "trace", "path": ...}` |
| `sink` | `{"kind": "memory"}` or `{"kind": "graphs", "path": ...}` |
| `controls` | defaults under every job's own controls |

## Slots and entries

`kohakugbc.engines.registry`: an entry is a factory registered for a slot; the build calls the
one each slot names, once, and every problem of the configuration is one error. A slot gives its
lane's units, former and selector, and either an adapter (the units run the slot's node from
its request) or a `call` (the units run the node's Python function, which calls it).

| Slot | Node | Entries |
|---|---|---|
| `vlm` | `chat` | `api`, `vllm` ([guide](../guides/engines.md)), `replay`, `remote` |
| `detect` | `detector` | `torch` (OWLv2 or SAM 3), `replay`, `remote` |
| `embed` | `embedder` | `torch` (EmbeddingGemma), `replay`, `remote` |
| `cpu` | heavy plain nodes | `python` (Togeari's Python workers, threads or processes), `remote` |

`replay` units are Togeari's simulated engines answering from a trace with its recorded
latencies (`latency_ms` fixes one; `speed` divides them, `speeds` per unit; `units`,
`capacity`, `max_batch`, `item_error_rate`, `outages`; `unknown="hash"` gives names the trace
never embedded a vector, for runs that leave the recorded path). Their payloads pickle, so they
also run behind an engine host. `remote` units are [engines on other nodes](engine-host.md).

## Failures

| What fails | The job |
|---|---|
| a VLM request | goes on with an empty reply, as GBC reads a failed request: the entity is absent, the root names nothing, the relation has no sentence |
| a detector or embedder call | fails, naming the node and the reason |
| a unit (`UnitError`, an outage, a lost host) | goes on: the lane retries the work on the other units; with none up, it waits up to the lane's `lane_dead_s` |

## Snapshots

A job snapshots only at a boundary: an empty window and one strand. Under run-ahead a GBC job
always holds speculative work, so it never reaches one; `knobs = {"snapshot_every": N}` needs
`"run_ahead": False`, and the build refuses it otherwise. Without run-ahead a job takes a
snapshot about every 15 statements and resumes from the last one; the run costs about 9% more
on pseudo engines ([performance](../performance/pseudo-engines.md)).

## Bindings and policies

| Nodes | Binding |
|---|---|
| `chat`, `detector`, `embedder` | their slot's lane; `embedder` from the content store when `store` is set |
| `open_image`, `crop`, `crop_many`, `image_root`, `seeded_root`, `detected`, `entity_record`, `batch_replies`, `coverage_root`, `relational_queries`, `relational_request`, `relational_record`, `to_gbc` | the CPU lane |
| every other node, `kir.std` | inline |

Admission is `TargetLoad` on the VLM lane (`vlm_waiting`, `max_jobs`): images are admitted while
the lane's queue and its units' waiting counts are below the target. Issue is
`OldestJobFirst`.

## Checks

`tests/machine` runs the recorded PoC jobs of every fixture arm on replay engines: each graph
equals the PoC's by digest and each recorded job conforms to KIR's reference interpreter; a job
whose requests take 50 ms each finishes in under half its requests' sum (a wave's regions run
side by side); 30 jobs keep a VLM lane of 8 within 1.4x of its floor. `tests/machine/
test_configurations.py` runs the jobs with three VLM units, with units behind an engine host in
the same process and in another, with a unit down mid-run, with failing VLM requests and
detector calls, with a slow unit beside a fast one, with a unit joining and one leaving mid-run,
and stopped mid-run then resumed from snapshots: every job that ends ok gives the PoC's graph. `scripts/bench/pseudo_machine.py`
measures the configurations of [performance](../performance/pseudo-engines.md).
