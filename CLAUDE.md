# KohakuGBC: repo notes

Graph-Based Captioning as a KIR program on the Togeari dataflow machine, with the engines that
run it. Three layers, split the way Togeari splits a job:

- `src/kohakugbc/program/`, `nodes/`, `graph/`: the GBC program (`gbc.kir`), the node library
  `gbc` with the specs of model nodes, the records and their resolution into GBC graphs.
- `src/kohakugbc/machine/`, `io/`: the program on Togeari (bindings, lanes, policies, the
  machine module), sources and sinks.
- `src/kohakugbc/engines/`: the engines filling Togeari's contracts (`vlm/`, `torch/`, `host/`,
  `replay/`).
- `src/kohakugbc_models/`: a standalone package (never imports `kohakugbc`) the torch engine
  builds: `detect/`, `embed/` and `vlm/` the models GBC runs itself (SAM 3, EmbeddingGemma, the
  Qwen3.5 engine; their entries sit in `kohakugbc/engines`), `kernels/`
  their kernels (registry, per-arch backends, tuning, CUDA graphs, fp8), `numerics/` error
  profiles against an fp64 ruler.
- `packages/kohakugbc-data/`: the data package, released to PyPI on its own (`data-v*` tags):
  `rust/` (`kohakugbc_data._rs`: graph resolution, graph store, prompts; crate `Cargo.toml` in
  the package), `src/kohakugbc_data/` (`prompt/`, `dataset/`). It imports nothing of
  `kohakugbc`; `kohakugbc` depends on it.
- vLLM changes (frontend or backend) live in KohakuvLLM, a separate drop-in plugin repo; this
  repo holds the engine entries and refers to it.
- `configs/` (`engines/`, `machine/`), `scripts/` (`trace/`, `bench/`), `docs/` (`concepts/`,
  `guides/`, `internals/`, `performance/`).
- `plans/` and `runs/` are gitignored.

## Commands

The environment is `.venv` (uv): `uv pip install -e packages/kohakugbc-data -e ".[dev]"`. The
data package's install builds the extension (maturin), so a change under
`packages/kohakugbc-data/rust/` needs `uv pip install -e packages/kohakugbc-data` again.
Versions: `kohakugbc` in `pyproject.toml`; `kohakugbc-data` in its `pyproject.toml`
(`scripts/ci/version.py` stamps its `Cargo.toml` to match). Releases of `kohakugbc-data`:
`.github/workflows/data-release.yml` (the only publisher), `data-nightly.yml`,
`data-auto-patch-release.yml`.

```
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --no-default-features
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m black --line-length 100 src tests scripts configs
.venv\Scripts\python.exe -m ruff check src tests scripts configs
.venv\Scripts\python.exe scripts\check_imports.py
.venv\Scripts\python.exe scripts\check_md_black.py .
```

GBC is a compute project: `tests/` holds data and format checks only (CPU); engines,
kernels and the machine are checked by runs on the cards (`scripts/bench/`), never by tests. CI
checks the code's form only (lint, import boundary, rustfmt, clippy).

## Rules

- Import boundary (`scripts/check_imports.py`): the program layer imports only `kohakunode`,
  image libraries and itself; the machine and io import `togeari`; engines import torch,
  transformers and vLLM (extras), never the machine; `kohakugbc_models` never imports
  `kohakugbc`.
- The GBC program and its nodes never name a device, an engine or a model; engines never name a
  GBC node.
- Code style: blank lines between logical groups; a short comment above a group says what it
  does; `match` rather than long if / elif chains; no imports inside functions; ruff + black,
  line length 100.
- Tests: no fake model. The program and machine are tested on pseudo engines replaying recorded
  traces; models are tested against their reference forward with real weights.
- Long runs go to background jobs.
