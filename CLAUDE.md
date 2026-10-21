# KohakuGBC: repo notes

Graph-Based Captioning as a KIR program on the Togeari dataflow machine, with the engines that
run it. Three layers, split the way Togeari splits a job:

- `src/kohakugbc/program/`, `nodes/`, `graph/`: the GBC program (`gbc.kir`), the node library
  `gbc` with the specs of model nodes, the records and their resolution into GBC graphs.
- `src/kohakugbc/machine/`, `io/`: the program on Togeari (bindings, lanes, policies, the
  machine module), sources and sinks.
- `src/kohakugbc/engines/`: the engines filling Togeari's contracts (`vlm/`, `torch/`, `host/`,
  `replay/`).
- `src/kohakugbc-rust/`: `kohakugbc._rs`, graph resolution in Rust (crate `Cargo.toml` at the
  root).
- `examples/qwen35/`: the specialised Qwen3.5 engine, its own package on top of the engines.
- `configs/` (`engines/`, `machine/`), `scripts/` (`trace/`, `bench/`), `docs/` (`concepts/`,
  `guides/`, `internals/`, `performance/`).
- `plans/` and `runs/` are gitignored.

## Commands

The environment is `.venv` (uv). The install builds the extension (maturin), so a change under
`src/kohakugbc-rust/` needs `uv pip install -e ".[dev]"` again. The version lives in
`Cargo.toml` alone.

```
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --no-default-features
.venv\Scripts\python.exe -m pytest tests -q -m "not gpu and not serial and not weights" -n auto --dist worksteal
.venv\Scripts\python.exe -m pytest tests -q -m "serial and not gpu"
.venv\Scripts\python.exe -m black --line-length 100 src tests examples scripts
.venv\Scripts\python.exe -m ruff check src tests examples scripts
.venv\Scripts\python.exe scripts\check_imports.py
.venv\Scripts\python.exe scripts\check_md_black.py .
```

`gpu` tests need a CUDA device; `weights` tests load real model weights; `serial` tests assert a
wall-clock bound or a scheduling outcome and run alone.

## Rules

- Import boundary (`scripts/check_imports.py`): the program layer imports only `kohakunode`,
  image libraries and itself; the machine and io import `togeari`; engines import torch,
  transformers and vLLM (extras), never the machine; `examples/qwen35` sits on the engines.
- The GBC program and its nodes never name a device, an engine or a model; engines never name a
  GBC node.
- Code style: blank lines between logical groups; a short comment above a group says what it
  does; `match` rather than long if / elif chains; no imports inside functions; ruff + black,
  line length 100.
- Tests: no fake model. The program and machine are tested on pseudo engines replaying recorded
  traces; models are tested against their reference forward with real weights.
- Long runs go to background jobs.
