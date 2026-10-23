# qwen35

A VLM engine written for one model family, Qwen3.5 (2B, 4B, 9B), as KohakuGBC's `vlm` slot
entry `qwen35` (`kohakugbc.engines.vlm.qwen35`, over the model code here). It is a specialised
engine behind Togeari's stream contract: the machine sees the same unit as for vLLM or an API.

## Install and run

```
pip install "kohakugbc[qwen35]"
GBC_IMAGES=path/to/images gbc run configs/machine/qwen35_local.py
```

`configs/machine/qwen35_local.py` puts the whole machine on one 24 GiB GPU: this engine
(`Qwen/Qwen3.5-4B`, or the folder in `QWEN35_MODEL`), the detector on the same device, the
embedder on the CPU. Graphs go to `runs/machine/qwen35_local/graphs.jsonl` (`GBC_OUT` moves
them). On an RTX 4090 it makes a graph in about 3 s per image
([measurements](../../../../docs/performance/real-engines.md)).

The entry in a configuration:

```python
engines = {
    "vlm": {
        "entry": "qwen35",
        "model": "Qwen/Qwen3.5-4B",
        "device": "cuda:0",
        "memory": 0.65,
        "state_dtype": "bfloat16",
    }
}
```

| Setting | Meaning |
|---|---|
| `model` | a checkpoint folder or a Hugging Face name |
| `device`, `devices` | one unit per device |
| `memory` | the share of the device the unit holds, weights included (0.5) |
| `max_seqs`, `max_model_len`, `page_size` | sequences generating at once (64), tokens per sequence (16384), KV page size (16) |
| `prefill_tokens`, `prefill_min`, `prefill_wait_ms` | prompt tokens per step (4096); when prefill joins a step of decodes (512 ready tokens, or a prompt waiting 50 ms) |
| `snapshots`, `boundaries` | prefix snapshots kept (16); where they are taken: `system` (after the system prompt), `images` (after the last image) |
| `state_dtype` | the DeltaNet states: `float32` (default) or `bfloat16` (half the memory) |
| `any_whitespace` | JSON replies may put whitespace anywhere, as vLLM compiles schemas (True); False keeps them compact |
| `jump`, `jump_max` | what the grammar forces is taken without a forward: `exact` (single allowed tokens), `string` (the forced text, re-tokenized), `off`; up to 16 tokens at a time |
| `cuda_graphs` | decode steps replay captured graphs (True) |
| `vision_patches` | patches per vision-tower call (32768) |
| `prefer`, `race` | op backends in order; whether the fastest is chosen by a timed race |
| `prep_workers`, `capacity`, `seed` | preparing threads (4); requests in flight (2 x `max_seqs`); the sampling seed |
| `step_log` | a path: one JSON line per engine step and per vision call |

`unit.stats()` reports counts (steps, graph steps, prompt / cached / generated / forced tokens,
snapshot hits, vision calls), times per phase, the op backend bound per problem size, and the
traceback of a stage that failed (a failed stage answers every request in it and every later
one with a unit error).

## How it works

| Part | File | What it does |
|---|---|---|
| Shape | `model/shape.py` | every size of a checkpoint, read from its `config.json` |
| Forward | `model/forward.py` | the text model over a step of packed sequences: each adds one or more tokens after what its state slot and KV pages hold; the checkpoint's vision tower |
| Ops | `ops/` | `delta` (gated delta rule), `conv` (the DeltaNet's short convolution), `attention` (paged); each with an eager reference and backends (flash-linear-attention, Triton kernels written here, PyTorch), bound once per problem size by a timed race |
| Pools | `serve/pools.py` | state slots (recurrent and convolution state per DeltaNet layer), reference-counted KV pages, prefix snapshots (a slot with the states after a prefix, its full pages shared, its partial page copied) |
| Inputs | `serve/inputs.py` | a request as token ids, mrope positions, pixels and snapshot boundaries, on the preparing threads |
| Vision | `serve/vision.py` | a thread and CUDA stream of its own: batches the images waiting, hands each request on with its embeddings and the event that marks them done |
| Engine | `serve/engine.py` | one thread: admits requests while a slot and pages are free (resuming from the longest stored snapshot), runs each step as one forward over the decode tokens and the prompt chunks due, fills the grammar masks while the device runs, draws, accepts |
| Graphs | `serve/graphs.py` | decode steps captured per batch bucket |
| Grammar | `serve/grammar.py` | xgrammar token masks per JSON schema; the forced-token jump |
| Unit | `kohakugbc/engines/vlm/qwen35.py` | the stream contract: submit, cancel, capacity, load, stats; the placement plan's reservation |

## Tests

```
QWEN35_MODEL=path/to/Qwen3.5-4B pytest tests/vlm/qwen35
```

- `test_qwen35_ops.py`: every backend against the eager reference on packed sequences in shuffled
  slots and pages (outputs, states written back, untouched slots); each check is shown able to
  fail on a wrong state or page table.
- `test_qwen35_model.py`: the forward against transformers' on a recorded request with its image
  (argmax agreement and KL over 48 reply tokens; a forward without the image is caught);
  resuming from a snapshot at either boundary equals a full prefill (a broken convolution state
  is caught); a captured decode graph equals the eager decode.
- `test_qwen35_unit.py`: Togeari's engine contract checks on recorded GBC requests in three
  configurations (vLLM-like schemas; compact JSON with `exact`; `string` with bfloat16 states and
  both boundaries); the machine with this entry as its VLM.

## Measuring

`scripts/bench/vlm_units.py` runs any `vlm` entry through the unit contract on recorded
requests and writes per-request results, memory and the unit's stats;
`scripts/bench/real_machine.py` runs the machine on recorded jobs and compares its graphs with
the PoC's. Results: [docs/performance/real-engines.md](../../../../docs/performance/real-engines.md).
