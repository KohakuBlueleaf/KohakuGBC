# The flagship machine

`configs/machine/qwen35_card.py` with `configs/controls/mid.py` + `configs/controls/sam3.py`: the
VLM is Qwen3.5-4B through vLLM 0.32.0 (an FP8 checkpoint with FP8 GDN projections on the RTX
5090, the SmoothQuant INT8 one on the RTX 3090), the detector GBC's SAM 3 (fp8 row linears on the
5090, bf16 on the 3090, batch 8), the embedder GBC's EmbeddingGemma, all on one card; one machine
per card (`GBC_CARD`). 480 images per card (`scripts/bench/machine/card.sh`, which records GPU
load, power, memory and host CPU beside the run). The KV cache is fixed per card (`GBC_VLM_KV_GB`
12 GiB on the 5090, 6 GiB on the 3090) so every cell holds the same cache; cells compared with
each other ran in one window on different cards of one box, rotated over the cards where the
cards differ. KohakuvLLM is
[github.com/KohakuBlueleaf/KohakuvLLM](https://github.com/KohakuBlueleaf/KohakuvLLM). The
sections from "Four RTX 5090s" on ran SAM 3 with fp8 row, 4 crops per vision pass, its vision
encoder eager, and admission bounded only by the VLM lane; the configuration's defaults are the
first section's bold row.

## An endless source on one RTX 5090

1200 COCO val2017 images through the machine on card 3 (KohakuvLLM's scheduler and grammar, INT8
GDN state, 384 lanes, KV 12 GiB). Job seconds are each image's time from admission to its graph;
the tail runs from the last image's admission to the end.

| SAM 3 | Admission | img/s | Job s p50 / p99 | VLM in flight | Tail |
|---|---|---|---|---|---|
| row, 4 crops, eager vision | VLM lane only (512 resident) | 1.507 | 326 / 355 | 46 | 314 s |
| mx, 8 crops, eager vision | VLM lane only | 1.619 | 302 / 330 | 55 | 294 s |
| mx, 8 crops, vision graph | VLM lane only | 2.006 | 228 / 295 | 235 | 233 s |
| mx, 8 crops, vision graph | + detect 32, 256 resident | 1.912 | 92 / 161 | 167 | 32 s |
| mx, 8 crops, vision graph | + detect 32, 192 resident | 1.909 | 72 / 121 | 158 | 80 s |
| mx, 8 crops, vision graph | + detect 32, 128 resident | 1.941 | 55 / 90 | 150 | 49 s |
| mx, 8 crops, vision graph | + detect 32, **96 resident** | **2.001** | **42 / 75** | 157 | 22 s |
| mx, 8 crops, vision graph | + detect 32, 64 resident | 1.880 | 30 / 55 | 100 | 26 s |

With only the VLM lane gating admission, every image of the run is admitted at once and waits
at the detector: 455 detect requests queued, the VLM 46 of 384 lanes busy after the first
minute. The vision CUDA graph is level for SAM 3 alone ([detector](detector.md)) but +24% here:
SAM 3's thread no longer launches the ViT's kernels beside the in-process vLLM. Bounding the
resident images bounds each image's time and the tail: at 192 resident, 600 images give job p50
74 s and a 70 s tail, 1200 give 72 s and 80 s; with the VLM lane alone, job p50 grows from 175 s
(600 images) to 228 s (1200). The
card is power-bound (96-97% busy at 580-590 W) in every row: from 96 resident images up, img/s
moves within the run-to-run noise (3-7%) while the time in the machine grows with the cap; at 64
the VLM falls to 100 lanes in flight and img/s with it. `GBC_ADMISSION=stream` (the default)
holds 96 images per card, `GBC_ADMISSION=batch` 512 (a long offline run, where the tail is
negligible).

## Four RTX 5090s, the best configuration

In-process VLM, KohakuvLLM's PhasedScheduler (prefill every 16 steps), KohakuvLLM's grammar
(`GBC_STRUCTURED=kohaku`), INT8 GDN state (`KOHAKU_GDN_STATE=int8`), 384 lanes, one machine per
card at once:

| Card | img/s | GPU busy | Power (of 575 W) | Memory peak |
|---|---|---|---|---|
| 0 | 1.592 | 94.6% | 535 W | 29.3 GiB |
| 1 | 1.617 | 93.5% | 535 W | 28.6 GiB |
| 2 | 1.599 | 95.2% | 532 W | 28.8 GiB |
| 3 | 1.650 | 94.0% | 555 W | 29.2 GiB |
| box | **6.46** | | | |

1920 of 1920 jobs; host CPU 15% busy. The parts, one per card in one window:

| Configuration | img/s |
|---|---|
| prompt + one retry, bf16 GDN state | 1.560 |
| + KohakuvLLM grammar | 1.588 |
| + INT8 GDN state | 1.591 |
| both | 1.594 |

The card is power-bound (96-98% busy at 545-560 W), so neither part moves it much; the grammar
makes every reply schema-valid by construction at no cost.

Kernel time on one card, a 40 s Nsight capture mid-run (`scripts/bench/machine/nsys.sh`), share
of GPU kernel time:

| Class | stock vLLM, prompt | PhasedScheduler + KohakuvLLM grammar |
|---|---|---|
| GPU busy | 98.8% | 96.4% |
| FP8 GEMM (VLM projections, SAM 3 linears) | 45.7% | 51.5% |
| attention (vLLM's and SAM 3's) | 21.2% | 14.4% |
| bf16 GEMM | 9.2% | 10.9% |
| GDN | 9.6% | 7.6% |
| other, copies, norms, sampling | 14.3% | 15.6% |

vLLM's split-KV decode attention takes 12.0% of stock's time and 3.1% under the PhasedScheduler,
whose decode-only steps replay full CUDA graphs. With the INT8 GDN state the engine stops at the
first CUDA-graph launch after an Nsight capture starts with node-level graph tracing (both
captures tried; the GPU idles while `cudaGraphLaunch` blocks); without the profiler, and with
graph-level tracing, it runs through, so its profile is the bf16-state one above.

## Structured replies (5090, one window)

| `GBC_STRUCTURED` | img/s | GPU busy |
|---|---|---|
| `kohaku` (KohakuvLLM's Rust grammar) | 1.589 | 96.8% |
| `prompt` (one retry; 1 reply in 400 needs it) | 1.539 | 98.6% |
| `grammar`, guidance | 1.426 | 85.4% |
| `grammar`, xgrammar | 1.226 | 74.8% |

## In-process vs `vllm serve` on the same card

Equal KV cache, the same engine arguments (`scripts/bench/machine/serve.py` serves the
configuration's `_VLLM`), one window per row group, img/s:

| Card, scheduler | in-process | behind `vllm serve` |
|---|---|---|
| 5090, stock vLLM, prompt (two rounds, cells rotated) | 1.424 / 1.406 | 1.404 / 1.427 |
| 5090, PhasedScheduler, prompt (two rounds) | 1.528 / 1.533 | 1.586 / 1.600 |
| 5090, stock vLLM, prompt | 1.418 | 1.445 |
| 5090, the best configuration | 1.653 | 1.695 |
| 3090, stock vLLM (before the 3090's fp32-accumulate unlock) | 0.408 | 0.418 |
| 3090, PhasedScheduler (before the unlock) | 0.475 | 0.425 |

On the 5090 the server is 2-4% ahead with KohakuvLLM's scheduler and level without it; on the
3090 in-process is 12% ahead. In-process, the machine's process holds the GIL longer (vLLM's
engine-core main thread 29% vs 16% of samples, SAM 3's thread 10% vs 2% in a py-spy capture at
full load): vLLM's input processing runs in the machine's process instead of the server's. Left
to `gpu_memory_utilization`, in-process vLLM profiles its memory while SAM 3 and EmbeddingGemma
load and keeps a smaller cache (5090 at 0.75: 9.4 GiB vs 14.5; 3090 at 0.72: 3.3 vs 8.4), and the
server at the same share overcommits the card (SAM 3 runs out of memory): set `GBC_VLM_KV_GB`.

KohakuvLLM's frontend in-process (`GBC_VLM_FRONTEND=kohakuvllm`: chat template and image
processor in four worker processes, the vision tower in the machine's process, the engine fed
token ids and image embeddings) gives prompt ids identical to vLLM's and a tower as close to a
float32 run as vLLM's own, but 1.21 img/s against 1.44-1.46 in the same window: the host runs at
load 94 with it, and its tower runs eagerly where vLLM's encoder is compiled and graph-captured.

## RTX 3090

SmoothQuant INT8 (`sq65all`), 128 lanes, KV 6 GiB, PhasedScheduler, in-process; after the cards'
fp32-accumulate unlock (bf16 matmul 120-130 TFLOPS, INT8 214-250 TOPS). The VLM alone (1000 in /
100 out): INT8 10.5 requests/s, bf16 6.9, FP8 weight-only 8.4, so INT8 stays. The machine, each
cell once on every card (cards 0, 1, 4, 5):

| Configuration | img/s mean | per card |
|---|---|---|
| bf16 GDN state | 0.576 | 0.568 / 0.585 / 0.604 / 0.545 |
| INT8 GDN state | **0.630** | 0.611 / 0.664 / 0.666 / 0.578 |
| KV 7 GiB | 0.580 | |
| INT8 GDN state + KV 7 GiB | 0.630 | |

The 3090 is KV-bound (cache 91-99% used); the INT8 state puts 11% more tokens in the same cache
and wins on every card (+6% to +14%); 1 GiB more cache memory (187k tokens against 160k) does
not, and adds nothing on top of the INT8 state. Card 5 is the slowest on every configuration (its power limit is 370 W
against 375-450 W).

KohakuvLLM's grammar against prompt + one retry, both with the INT8 state, each on every card in
two rounds: 0.637 vs 0.632 img/s mean (per card +2.4%, -0.3%, +0.4%, +0.3%): level, with every
reply valid. The 3090's best configuration is the 5090's: in-process, PhasedScheduler, grammar,
INT8 GDN state.
