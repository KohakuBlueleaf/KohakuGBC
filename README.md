# KohakuGBC

Graph-Based Captioning ([GBC](https://arxiv.org/abs/2407.06723)) as one KIR program on the
[Togeari](https://github.com/KohakuBlueleaf/Togeari) dataflow machine, with the engines that run
it.

A GBC graph describes one image as a DAG: the image vertex, entity vertices for the objects its
caption names, their parts below them, and relation and composition vertices over several
entities. Every edge is grounded three ways: it is a graph edge, its label is a phrase in the
parent's caption, and the child's box lies inside the parent's box.

Producing one graph is a branching process (detect, describe each region, detect its parts,
merge regions that are the same object, ask relational questions) whose control flow cannot be
batched across images. KohakuGBC writes that process once, as a KIR program; Togeari runs many
images at once and turns their model calls into batched engine calls.

## Layers

| Layer | Package | Holds |
|---|---|---|
| GBC program | `kohakugbc.program`, `kohakugbc.nodes`, `kohakugbc.graph` | `gbc.kir`, the node library `gbc`, controls, records, graph resolution (Python reference; Rust in `kohakugbc._rs`) |
| GBC machine | `kohakugbc.machine`, `kohakugbc.io` | the program on Togeari: bindings, lanes, policies, sources, sinks |
| GBC engines | `kohakugbc.engines` | the VLM (an OpenAI-compatible API, in-process vLLM), torch engines for detection and embedding, engines on other nodes, replay engines |

## Install

```
pip install kohakugbc            # program, machine, replay engines
pip install "kohakugbc[api]"     # the VLM over an HTTP API
pip install "kohakugbc[torch]"   # detector and embedder on torch
pip install "kohakugbc[vllm]"    # the VLM in-process (Linux)
```

## Run on one GPU

```
pip install "kohakugbc[torch]"
pip install -e examples/qwen35
GBC_IMAGES=path/to/images gbc run configs/machine/qwen35_local.py
```

The VLM ([examples/qwen35](examples/qwen35/README.md), Qwen3.5-4B), the detector and the
embedder on one 24 GiB GPU; one graph per image in `runs/machine/qwen35_local/graphs.jsonl`.

## Docs

[docs/](docs/README.md): concepts, guides, internals, performance.

## License

Apache-2.0.
