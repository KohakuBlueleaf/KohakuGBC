# The detector and the embedder

GBC's own SAM 3 and EmbeddingGemma ([internals](../internals/kernels.md)) against the models
they replace, transformers' SAM 3 and sentence-transformers' EmbeddingGemma, and against
torch.compile, on six GPUs and one CPU. Every run: torch 2.14, Python 3.13, each card's kernels
tuned from an empty cache.

## Workloads

- **W1**: the teaser image's detector requests: 4 crops, 12 (crop, text) pairs.
- **W2**: the detector requests recorded in a PoC trace: 45 requests, 116 pairs, submitted 4
  or 16 per batch.
- **Names**: the 925 distinct names the PoC traces embedded or detected.

## SAM 3 through the detect engine (W2, batch 4)

`scripts/bench/detector/engine.py`: the `detect` slot as a machine builds it (Togeari's torch
engine, its prep workers and batch slots), all 45 requests submitted at once.

| GPU | dtype | GBC's SAM 3 | transformers' SAM 3 | speed-up |
| --- | --- | --- | --- | --- |
| RTX 5090 | bf16 | 1502 ms | 3441 ms | 2.29x |
| RTX 4090 | bf16 | 2409 ms | 5482 ms | 2.28x |
| RTX 3090 | bf16 | 4472 ms | 7492 ms | 1.68x |
| RTX 2080 Ti | fp16 | 7854 ms | 12802 ms | 1.63x |
| V100 16 GB | fp16 | 5032 ms | 8882 ms | 1.77x |
| PG503 32 GB | fp16 | 5441 ms | 9155 ms | 1.68x |

Peak allocated device memory (W2, 16 per batch): 2.9-3.4 GiB against 5.0-5.5 GiB.

## SAM 3 against torch.compile (W2, batch 4)

`scripts/bench/detector/native.py`: the model alone, three ways: its eager ops issued one by
one; the best torch.compile configuration measured (vision `max-autotune-no-cudagraphs`,
fusion `default`); GBC's kernels with the fusion pass and the text encoder replayed from CUDA
graphs, and fp8 where the GPU has it.

| GPU | eager | torch.compile | GBC | GBC + fp8 |
| --- | --- | --- | --- | --- |
| RTX 5090 | 2637 ms | 2073 ms | 1797 ms | 1236 ms (mx), 1432 ms (row) |
| RTX 4090 | 4614 | 3760 | 2883 | 2513 (row) |
| RTX 3090 | 6388 | 5441 | 4716 | |
| RTX 2080 Ti | 10544 | 9139 | 8116 | |
| V100 | 7533 | 6848 | 5309 | |
| PG503 | 7991 | 7451 | 5709 | |

fp8 is opt-in (`fp8: "row" | "mx"`): it doubles the bf16 logits' error against fp64 (below).

## Precision

`scripts/bench/detector/reference.py`: the query logits of every W2 pair against transformers'
model run in fp64. GBC's model in fp64 equals it to 2e-12 on every GPU that ran it.

| GPU | dtype | transformers | GBC | GBC fp8 |
| --- | --- | --- | --- | --- |
| RTX 5090 | bf16 | 0.131 | 0.099 | 0.198 (row), 0.205 (mx) |
| RTX 4090 | bf16 | 0.131 | 0.099 | 0.205 (row) |
| RTX 3090 | bf16 | 0.130 | 0.099 | |
| PG503 | fp16 | 0.0249 | 0.0223 | |
| RTX 2080 Ti | fp16 | 0.0247 | 0.0226 | |

(logits abs mean; the V100's 16 GB does not hold the fp64 ruler, the PG503 is the same arch)

## The embedder (925 names, batch 64)

`scripts/bench/embed/embedder.py`; GBC's model packs a batch into one token stream and replays
it from a CUDA graph.

| GPU | sentence-transformers fp32 | GBC fp32 | GBC bf16 / fp16 |
| --- | --- | --- | --- |
| RTX 5090 | 1726 texts/s | 7307 | 10974 |
| RTX 4090 | 1472 | 6004 | 8680 |
| V100 | 1833 | 3223 | 4488 |
| RTX 2080 Ti | 1881 | 3336 | 5497 |

fp32 (the default) gives sentence-transformers' vectors (1 - cos <= 1e-7 against fp64).

## The CPU (Xeon with AMX, 32 threads)

`scripts/bench/cpu.py`, W1: GBC's SAM 3 7.4 s (bf16) / 8.2 s (fp32) against transformers' 8.7 s
/ 10.8 s; the embedder 998 against 847 texts/s (fp32).

## Memory

`scripts/bench/detector/memory.py` measures a plan per GPU that the detect entry reads for its
placement estimate and its `budget_gb` caps. On the 5090 (bf16, 4 crops x 16 pairs): 1.62 GiB
resident, 6.3 GiB estimated at the peak with the allocator's reserve; 0.56 GiB resident with
fp8 and the text encoder on the host.
