# Performance

Measured throughput of the GBC machine per engine and configuration, with the hardware and the
settings of every measurement.

| Page | Covers |
|---|---|
| [pseudo-engines.md](pseudo-engines.md) | the machine on replay engines: VLM lane occupancy, admission, window |
| [real-engines.md](real-engines.md) | the machine on real models on one GPU; the VLM engines alone on recorded requests |
| [detector.md](detector.md) | SAM 3 and EmbeddingGemma: GBC's own models against transformers / sentence-transformers and torch.compile on 5090, 4090, 3090, 2080 Ti, V100, PG503 and the CPU; precision against fp64 |
