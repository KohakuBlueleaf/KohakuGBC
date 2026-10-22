# Placement

`kohakugbc.engines.placement` checks, before any weight loads, that every engine fits the
devices it is placed on.

Each entry says what its units will hold (`Need`: owner, device, bytes, what):

| Entry | Need per device |
|---|---|
| `torch` (detector, embedder) | the model's estimate of weights and peak activations (`memory_gb`, by default 2.0 GiB for OWLv2-base, 1.4 GiB for EmbeddingGemma) |
| `vllm` | its reservation: `gpu_memory_utilization` of every device the unit uses |

The machine adds the needs per device and compares them with the memory free at build time less
a headroom (`machine.placement_headroom`, 5% of the device); every device that does not fit is
one line of one error, and no unit starts. Togeari's torch engine measures each model's real
footprint at its start (`load().extra["footprint"]`).

| Placement | Configuration |
|---|---|
| one device, every model | every slot's `devices` names the same device |
| a copy per device | `devices` lists several: one unit per device, the lane chooses by load |
| a model across devices | a list inside `devices` (Togeari splits an `nn.Sequential` into stages); for the VLM, `vllm`'s `devices` per unit (tensor parallel) |
| several nodes | the `remote` entry over engine hosts ([engines on other nodes](engine-host.md)) |
