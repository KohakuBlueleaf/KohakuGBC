# Real engines

The GBC machine and its VLM on real models, one RTX 4090 (24 GiB) under Windows 11, Python
3.13, torch 2.14.

## The machine on one GPU

`python scripts/bench/real_machine.py configs/machine/qwen35_local.py --name qwen35-4090`: the
recorded jobs of two PoC arms (30 images each) on the `qwen35` VLM (Qwen3.5-4B, `memory` 0.65,
bfloat16 states), the OWLv2 detector on the same GPU (float32, batch 8) and EmbeddingGemma on
the CPU. Every job is recorded and checked against KIR's reference interpreter. Wall time
includes loading the models.

| Arm | Jobs | Ok | Conform | Wall | Jobs/s |
|---|---|---|---|---|---|
| text (seeded by a caption) | 30 | 30 | 30/30 | 89.0 s | 0.34 |
| query | 30 | 30 | 30/30 | 102.1 s | 0.29 |

Graph statistics, means per graph, ours / the PoC's (the PoC ran Qwen3.5-4B on vLLM; replies are
sampled at temperature 0.1, so graphs differ image by image):

| Arm | Vertices | Entities | Relations | Compositions | Edges | Captions |
|---|---|---|---|---|---|---|
| text | 15.97 / 15.17 | 9.83 / 9.40 | 3.73 / 3.47 | 1.40 / 1.30 | 28.43 / 26.07 | 27.43 / 26.33 |
| query | 16.03 / 17.13 | 9.97 / 10.70 | 3.57 / 3.73 | 1.50 / 1.70 | 27.40 / 29.63 | 27.63 / 31.03 |

## The VLM alone

`scripts/bench/vlm_units.py`: the 425 VLM requests of the text arm (16 warm-up, 409 measured)
through the unit contract at 64 in flight, Qwen3.5-4B, schemas compiled with any whitespace.

| Engine | Requests/s | Reply tokens/s | Latency p50 / p95 | Valid JSON (schema requests) |
|---|---|---|---|---|
| `qwen35` (`memory` 0.8, float32 states) | 7.48 | 828 | 7.51 s / 11.85 s | 99.7% |
| vLLM 0.32 in WSL (`gpu_memory_utilization` 0.8, `max_num_seqs` 64) | 10.21 | 1101 | 6.18 s / 9.76 s | 100% |

From `qwen35`'s step log: decode steps replay captured graphs at 15-26 ms of device time (1 to
64 sequences); prompt chunks run eagerly (193 steps, 723 tokens each on average, 134 ms each);
the vision tower takes 80 ms per call (1422 patches on average) on its own stream.
