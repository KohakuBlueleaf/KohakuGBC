"""GBC's kernels, after KohakuMoss's ``kohakumoss_kernels`` conventions, for inference.

Each op has one eager reference (``backends/eager``) and per-arch implementations
(``backends/tilelang`` on sm70 / sm75, ``backends/triton`` on sm80+, ``backends/compile``:
torch.compile of the eager reference) registered with ``registry.register``; a caller resolves
an op once (``registry.resolve``) to a ``Bound`` that falls back along the arch's backend
priority when a backend cannot build or launch. Tuned kernels pick their configs with
``tuning.select`` (per-card tables, the PyTorch paths in every shortlist).
"""
