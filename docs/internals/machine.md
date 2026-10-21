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
| `machine` | `vlm_waiting`, `max_jobs` (admission), `window`, `record`, `store` (`"memory"`), `report` (a RunReport folder), `knobs` (other Togeari knobs) |
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
| `vlm` | `chat` | `replay` |
| `detect` | `detector` | `replay` |
| `embed` | `embedder` | `replay` |
| `cpu` | heavy plain nodes | `python` (Togeari's Python workers, threads or processes) |

`replay` units are Togeari's simulated engines answering from a trace with its recorded
latencies (`latency_ms` fixes one; `speed` divides them; `units`, `capacity`, `max_batch`,
`item_error_rate`, `outages`).

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
side by side); 30 jobs keep a VLM lane of 8 within 1.4x of its floor. `scripts/bench/pseudo_machine.py`
measures the configurations of [performance](../performance/pseudo-engines.md).
