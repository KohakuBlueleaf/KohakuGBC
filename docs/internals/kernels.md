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
| `registry` | ops by backend; per-arch priority (sm70 / sm75: TileLang, compile, eager; sm80+: Triton, TileLang, compile, eager; CPU: eager); once-per-device probes; a backend failing to build or launch rebinds every live op of its query to the next |
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

Every op has an eager reference (`backends/eager`) with transformers' casts and order, and a
torch.compile path; the tuner only binds a kernel where it measured faster.

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
