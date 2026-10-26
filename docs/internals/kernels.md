# Kernels: GBC's own SAM 3 and EmbeddingGemma

The detect and embed slots run models GBC implements itself (`kohakugbc_models.detect.sam3`,
`kohakugbc_models.embed.gemma`), on the kernel package `kohakugbc_models.kernels`. Both load the published
checkpoints unchanged (`facebook/sam3`, `google/embeddinggemma-300m`).

## The models

**SAM 3** (`detect/sam3`): the ViT (32 layers, windowed and global attention with 2D RoPE),
the FPN neck (only the levels the box source reads), CLIP text encoder (transformers'), DETR
encoder / decoder, scoring, mask decoder. What it does differently from transformers' forward:
each crop's vision features once (transformers: once per pair), each unique text once (LRU
cache), pairs read their crop's features in place, the mask decoder only for the mask box
sources, shape-only tables (RoPE, neck positions, box frequencies) made once per placement. In
fp64 it equals transformers' model to 2.3e-12.

**EmbeddingGemma** (`embed/gemma`): Gemma 3 bidirectional encoder over one packed token stream
(per-token text spans, the sliding window on 5 of 6 layers), mean pooling, two dense layers,
L2 normalization. In fp64 it equals sentence-transformers to 7.5e-8 (fp32 pooling); in bf16 its
hidden states equal transformers' bit for bit.

## The kernel package

| Module | Role |
| --- | --- |
| `registry` | ops by backend; per-arch priority, an arch without its own entry taking its family's (sm70 / sm75: TileLang, compile, eager; sm80 / sm86 / sm89 / sm90 and the sm12x family: Triton, TileLang, compile, eager; the sm10x family: Gluon, Triton, TileLang, compile, eager; CPU: eager); once-per-device probes; a backend failing to build or launch rebinds every live op of its query to the next |
| `hardware` | a device's arch, its family (`sm10x`: B200 sm100, B300 sm103, 228 KiB shared memory, tcgen05; `sm12x`: RTX 50xx sm120, DGX Spark sm121, about 100 KiB) and its card key; `tiles_for`: a kernel's tile set plus its family's extra tiles |
| `tuning` | per (op, problem key, card) the fastest config among the kernel's tiles and the PyTorch paths (`torch`, `compile`), stored under `KOHAKUGBC_KERNEL_CACHE`; metric `loop` (wall per call, host included) or `graph` (device time inside a CUDA graph), stored apart |
| `graphs` | a callable replayed from one CUDA graph per static shape, one memory pool, thread-local capture |
| `fp8` | e4m3 recipes `row` (sm89+) and `mx` (sm100+, e8m0 1 x 32 blocks, 32 x 4 x 4 scale layout), weight quantization, the scaled GEMM |
| `bench` | every candidate of an op on captured real operands: time (L2 flushed) and its error profile against the eager reference in fp64 |

## Ops

| Op | Kernels | Where it wins (vs torch.compile, 5090 bf16 unless marked) |
| --- | --- | --- |
| `vit_attention` | Triton, TileLang | window 0.314 vs 0.41 ms; global ties; TileLang on PG503 1.31 vs 2.08 (window), 10.2 vs 14.4 (global) |
| `attention` | Triton, TileLang | text cross 0.041 vs 0.067 ms; TileLang encoder self on Volta / Turing ~2x |
| `attention_rpb` | Triton, TileLang | 0.24 vs 0.48 ms |
| `box_bias` | Triton | 0.069 vs 0.285 ms (the MLP hidden never written) |
| `mask_box` | Triton | 1.27 vs 3.56 ms, no `[P, 200, 288, 288]` logits; boxes equal fp64's |
| `varlen_attention`, `segment_mean` | Triton | EmbeddingGemma's packed stream |
| `quant`, `add_ln_quant`, `act_quant` | Triton | the fp8 GEMMs' inputs from fp32 values, no extra pass |
| `linear_act`, `qkv_rope`, `add_layer_norm` | Triton | registered; chosen per problem where faster (GELU fc1 0.86 vs 1.03 ms) |
| `linear_act`, `qkv_rope` | Gluon (sm10x) | persistent warp-specialized tcgen05 GEMM, epilogue overlapped with the next tile; against the Triton entry, cuBLAS and (`qkv_rope`) `split`. B300, 8 crops: GELU fc1 0.36 vs 0.71 ms (Triton tile); `qkv_rope` binds `split` (0.21 vs 0.65 ms) |
| `quant`, `act_quant`, `add_ln_quant` | Gluon backend (sm10x, Triton kernels) | `mx`: scales written in cuBLAS's blocked layout (no relayout per GEMM), 2D tiles; B300, 8 crops: act_quant 0.17 vs 0.32 ms (GELU's erf by Abramowitz-Stegun with the hardware reciprocal and exp2, error under 5e-7); the rest of the recipes run the Triton entry |
| `vit_attention`, `attention` | Gluon backend (sm10x) | unrotated q / k (views of the fp8 q / k / v GEMM's output) rotated once into contiguous buffers (one load per element: 62 vs 80 µs for 8 crops); global layers and the DETR encoder's unmasked self-attention through KohakuFA's tcgen05 forward when installed and measured faster (its table tuned per card under `<kernel cache>/kohakufa/`). B300: encoder self d32, 16 pairs 0.76 vs 0.94 ms (cuDNN); ViT global, 8 crops 0.85 vs 0.95 ms |

Every op has an eager reference (`backends/eager`) with transformers' casts and order, and a
torch.compile path; the tuner only binds a kernel where it measured faster.

Kernels for one family sit beside the others' and are picked by the registry: the `gluon`
backend registers only sm10x implementations (its probe refuses any other family), the Triton
kernels serve every family with the family's extra tiles (`hardware.tiles_for`). On B300 with
the `mx` ViT, 8 crops per pass: the ViT 6.6 → 4.9 ms per crop; 2000 recorded Danbooru detector
requests (`scripts/bench/detector/native.py --recorded`, prepared outside the timer, text cache
warm) 98 → 141 per second (same detections as before within run-to-run noise).

## Precision

The ruler is fp64. Each kernel's error profile (cosine, abs / rel mean / median / max, the worst
elements with their true values) is compared with the PyTorch eager path in the same dtype;
nothing is judged by a fixed threshold. Measured against transformers' bf16 on 116 real pairs:
ours bf16 logits abs mean 0.098 (transformers 0.131); fp8 0.20 (a knob, off by default).

## CUDA graphs

The DETR fusion pass (per exact pair count), the text encoder (per power-of-two count of new
texts) and EmbeddingGemma (per stream bucket) are replayed from graphs: their many small kernels
then cost device time only. Kernels tune under the `graph` metric while the graph is warmed up.

## Memory plan

`scripts/bench/detector/memory.py` measures, per card and knobs, resident bytes, vision working
bytes per crop, fusion working bytes per pair, the bytes graphs and caches keep and the
allocator's reserve; the detect entry reads the plan for its placement estimate and fits its
caps to `budget_gb`. Levers: `text_device: cpu`, `fp8`, `fusion_buckets: pow2`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
