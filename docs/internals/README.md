# Internals

How each part works.

| Page | Covers |
|---|---|
| [machine.md](machine.md) | configuration layers, slots and entries, bindings, policies, failures, snapshots |
| [placement.md](placement.md) | the placement plan: what engines hold on which device, checked before loading |
| [engine-host.md](engine-host.md) | engines on other nodes: the engine host, the remote unit, the wire |
| [program.md](program.md) | the KIR program, its node library, how it reproduces the PoC |
| [traces.md](traces.md) | the trace format, its keys, recording the PoC |
| [graph.md](graph.md) | records, graph resolution, the Python and Rust implementations |
| [kernels.md](kernels.md) | GBC's own SAM 3 and EmbeddingGemma: the kernel package, its ops per arch, tuning, CUDA graphs, fp8, the memory plan, how precision is checked |
