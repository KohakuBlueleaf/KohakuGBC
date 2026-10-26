# The whole GBC machine on one card: Qwen3.5 through vLLM's engine, in this process (the
# in-process ``vllm`` entry) or behind a `vllm serve` on the same card (GBC_VLM_URL, the ``api``
# entry; scripts/bench/machine/serve.py starts that server with the same engine arguments), and
# GBC's own SAM 3 and EmbeddingGemma on the same card.
# One machine per card: GBC_CARD picks it, so N processes cover an N-card box; or GBC_CARDS, one
#   machine over several cards.
# Measured (mid + sam3 controls; docs/performance/flagship.md):
#   RTX 5090, 1200 COCO images: FP8-GDN checkpoint, GBC_VLM_UTIL=0.75, GBC_VLM_LANES=384,
#     PhasedScheduler with KOHAKU_PREFILL_EVERY=16, this file's SAM 3 and admission defaults,
#     engines in a host per card: 2.06 img/s, job p50 41 s, GPU 98% busy at its power cap; four
#     cards (GBC_CARDS=0,1,2,3, 4800 images) 8.17 img/s, job p50 42 s.
#   RTX 3090, 480 images (SAM 3 with 4 crops, eager vision, admission by the VLM lane alone):
#     SmoothQuant INT8 checkpoint, GBC_VLM_UTIL=0.72, GBC_VLM_LANES=128, same scheduler: 0.41
#     img/s, bound by KV-cache capacity.
# GBC_VLM_MODEL: a W8A8 checkpoint with dynamic per-token activations. FP8 on sm89 and up
#   (RedHatAI/Qwen3.5-4B-FP8-dynamic, or with the GDN projections in FP8 too: +19% requests/s);
#   INT8 below (KohakuvLLM tools/quantize.py and tools/smooth.py; 1.9x the FP8 checkpoint's
#   requests/s on a 3090, where FP8 runs weight-only).
# GBC_VLM_STATE: the GDN recurrent state dtype, bfloat16 (+27% requests/s over float32).
# GBC_VLM_KV_GB: vLLM's KV cache in GiB, set outright. Unset, vLLM sizes it from GBC_VLM_UTIL
#   less what the card holds when it profiles: in-process, SAM 3 and EmbeddingGemma load in the
#   same seconds and count against it (5090 at 0.75: 9.4 GiB in-process vs 14.5 behind a server;
#   3090 at 0.72: 3.3 vs 8.4), so both paths compare only with this set.
# GBC_VLM_UTIL: vLLM's share of the card; SAM 3 and EmbeddingGemma take about 5 GiB beside it.
#   GBC's prompts are long (images), so the share and GBC_VLM_KV (auto or fp8) set how many
#   requests the engine holds. GBC_VLM_LANES must not exceed its Mamba cache blocks.
# GBC_VLM_PREFIX: prefix caching (1: worth about 23%). With it, vLLM's Mamba "align" mode cuts
#   prefill chunks at attention-block boundaries, and fp8 KV (544-token blocks) runs only a few
#   requests at once.
# GBC_DETECT_GB: SAM 3's placement reservation (fp8 row, batch 8 peaks at about 2.5 GB).
# GBC_DETECT_FP8: SAM 3's ViT linears in fp8: auto (mx on sm100+, row on sm89, none below), mx,
#   row, nvfp4 or none; GBC_DETECT_FP8_MLP: the MLPs' recipe (GBC_DETECT_FP8's if unset; mx with
#   nvfp4 keeps attention in MX). GBC_DETECT_FP4_SCALES: nvfp4's calibrated input amax JSON
#   (one producer pass each; scripts/bench/detector/fp4_calibrate.py writes it); unset: each
#   input's own amax per call. GBC_DETECT_VISION_GRAPH=1: SAM 3's ViT replayed from CUDA
#   graphs, one per crop count up to GBC_DETECT_CROPS (crops per vision pass, 8; at most
#   GBC_DETECT_BATCH, 8).
#   GBC_DETECT_SCHEDULE=priority_streams with GBC_DETECT_PRIORITY=N: SAM 3 on a CUDA stream of
#   priority N, ahead of the VLM's; on the RTX 5090 beside the in-process vLLM it locked the card
#   (a kernel spinning at 100% busy).
# Admission: GBC_VLM_WAITING (VLM waiting + queued items, GBC_VLM_LANES per card by default),
#   GBC_WAITING=lane=items,... (other lanes' limits on new jobs, detect=32 per card by default),
#   GBC_MAX_JOBS (resident jobs; per card 96 under GBC_ADMISSION=stream, the default, 512 under
#   batch): a job's time in the machine is about max_jobs / img/s. RTX 5090, 1200 images: 96 jobs
#   2.00 img/s, job p50 42 s, tail 22 s; 512 jobs 2.01 img/s, p50 228 s, tail 233 s.
# GBC_VLM_SCHEDULER: a vLLM scheduler class path, e.g. kohakuvllm.sched.phased.PhasedScheduler
#   (decode-only steps between prefills, KOHAKU_PREFILL_EVERY; vLLM's own by default).
# GBC_VLM_ATTN: vLLM's attention backend (FLASH_ATTN, FLASHINFER, TRITON_ATTN; vLLM's pick if
#   unset; fp8 KV on sm120 rules out FLASH_ATTN).
# GBC_VLM_STEP_TOKENS: vLLM's token budget per step (8192).
# GBC_VLM_GRAPH_MAX: the largest step (tokens) vLLM captures a CUDA graph for (vLLM's default if
#   unset).
# GBC_STRUCTURED: kohaku (default; KohakuvLLM's grammar logits processor, KOHAKU_GRAMMAR_WS),
#   grammar (vLLM's structured outputs; GBC_VLM_GRAMMAR picks the backend: auto, xgrammar,
#   guidance, outlines) or prompt (in-process, 1 reply in 400 asked again; over HTTP, no retry).
# GBC_VLM_URL: a `vllm serve` base URL (http://127.0.0.1:PORT/v1); unset: in-process.
# GBC_VLM_FRONTEND (in-process): vllm, or kohakuvllm (KohakuvLLM's frontend: GBC_VLM_PREP_WORKERS
#   preparing processes, its vision tower here, GBC_VLM_VISION_FP8=1 for FP8 linears on sm89+).
# GBC_VLM_STATS=1: vLLM's stats line every 5 s.
# GBC_VLM_DTYPE: vLLM's dtype (bfloat16; float16 below sm80). GBC_VLM_MM_CACHE_GB: vLLM's
#   multimodal processor cache (mm_processor_cache_gb; vLLM's 4 GiB if unset). GBC_VLM_MM_GRAPH=0:
#   the vision encoder without CUDA graphs (below sm80 its SDPA path cannot be captured).
# GBC_AUX_CARD: the card SAM 3 and EmbeddingGemma run on (GBC_CARD if unset), so a pair of cards
#   splits one machine: the VLM alone on GBC_CARD. GBC_DETECT_DTYPE: SAM 3's dtype (auto: bfloat16
#   on sm80+, float16 below).
# GBC_DETECT_FORMER / GBC_EMBED_FORMER: the lane's batch former, size (requests) or budget (each
#   request's costs: SAM 3's crops and pairs per pass, the embedder's texts and characters). On
#   the RTX 5090 machine budget halves SAM 3's batches (7.4 -> 3.7 requests) and costs 4% img/s.
# GBC_EMBED_DEVICE: card (default: EmbeddingGemma beside SAM 3 on the aux cards) or cpu (one
#   host process without a card, GBC_EMBED_THREADS torch threads, 8; the aux cards keep SAM 3).
# GBC_TAG=1: the PixAI tagger beside SAM 3 on the aux cards (configs/controls/tags.py places its
#   tags); GBC_TAG_FP8 (auto), GBC_TAG_BATCH (4 images per pass), GBC_TAG_MODEL.
# GBC_VLM_TIMEOUT_S: the VLM lane's deadline per attempt (none if unset); an expired request
#   answers empty and is counted in the job's policy failures.
# GBC_WAITS_S: why jobs wait, sampled every that many seconds into the report's waits.jsonl.
#   GBC_IMAGES=path/to/images gbc run configs/machine/qwen35_card.py configs/controls/mid.py

import os

_CARD = int(os.environ.get("GBC_CARD", "0"))
_AUX = int(os.environ.get("GBC_AUX_CARD", str(_CARD)))
# GBC_CARDS=0,1,2,3: one machine over those cards, each engine one unit per card (GBC_CARD and
# GBC_AUX_CARD unused); GBC_VLM_CARDS / GBC_AUX_CARDS set the VLM's and SAM 3 + EmbeddingGemma's
# cards apart (a 27B VLM fills its card: GBC_VLM_CARDS=0,1,2 GBC_AUX_CARDS=3). The admission
# defaults scale with the VLM card count
_CARDS = [int(c) for c in os.environ.get("GBC_CARDS", "").split(",") if c]
_VLM_CARDS = [int(c) for c in os.environ.get("GBC_VLM_CARDS", "").split(",") if c]
_AUX_CARDS = [int(c) for c in os.environ.get("GBC_AUX_CARDS", "").split(",") if c]
_VLM_CARDS = _VLM_CARDS or _CARDS or [_CARD]
_AUX_CARDS = _AUX_CARDS or _CARDS or [_AUX]
_N = len(_VLM_CARDS)
_ADMISSION = os.environ.get("GBC_ADMISSION", "stream")
_JOBS_PER_CARD = {"stream": 96, "batch": 512}[_ADMISSION]
_MM_CACHE = os.environ.get("GBC_VLM_MM_CACHE_GB", "")
_TIMEOUT = os.environ.get("GBC_VLM_TIMEOUT_S", "")
_VLM_LANE = {"timeout_s": float(_TIMEOUT)} if _TIMEOUT else {}
_WAITS = os.environ.get("GBC_WAITS_S", "")
_OUT = os.environ.get("GBC_OUT", f"runs/machine/qwen35_card{'-'.join(map(str, _VLM_CARDS))}")
_LANES = int(os.environ.get("GBC_VLM_LANES", "256"))
_EVENTS = os.environ.get("GBC_EVENTS", "sampled")
_DETECT_FP8 = os.environ.get("GBC_DETECT_FP8", "auto")
_WAITING = dict(
    (k, int(v))
    for k, v in (
        kv.split("=") for kv in os.environ.get("GBC_WAITING", f"detect={32 * _N}").split(",") if kv
    )
)
_EMBED_CPU = {"card": False, "cpu": True}[os.environ.get("GBC_EMBED_DEVICE", "card")]
_EMBED_THREADS = int(os.environ.get("GBC_EMBED_THREADS", "8"))
_TAG = os.environ.get("GBC_TAG", "0") == "1"
_SCHEDULER = os.environ.get("GBC_VLM_SCHEDULER", "")
_GRAPH_MAX = int(os.environ.get("GBC_VLM_GRAPH_MAX", "0"))
_ATTN = os.environ.get("GBC_VLM_ATTN", "")
_STATS = os.environ.get("GBC_VLM_STATS", "0") == "1"

_URL = os.environ.get("GBC_VLM_URL", "")
_KV_GB = float(os.environ.get("GBC_VLM_KV_GB", "0"))
_STRUCTURED = os.environ.get("GBC_STRUCTURED", "kohaku")

# vLLM's engine, in-process or behind `vllm serve` (scripts/bench/machine/serve.py reads _VLLM)
_VLLM = {
    "model": os.environ.get("GBC_VLM_MODEL", "RedHatAI/Qwen3.5-4B-FP8-dynamic"),
    "served_model_name": "gbc-vlm",
    "dtype": os.environ.get("GBC_VLM_DTYPE", "bfloat16"),
    "gpu_memory_utilization": float(os.environ.get("GBC_VLM_UTIL", "0.55")),
    **({"kv_cache_memory_bytes": int(_KV_GB * 2**30)} if _KV_GB else {}),
    "max_model_len": 8192,
    "max_num_seqs": _LANES,
    "mamba_ssm_cache_dtype": os.environ.get("GBC_VLM_STATE", "bfloat16"),
    "kv_cache_dtype": os.environ.get("GBC_VLM_KV", "auto"),
    "max_num_batched_tokens": int(os.environ.get("GBC_VLM_STEP_TOKENS", "8192")),
    "enable_prefix_caching": os.environ.get("GBC_VLM_PREFIX", "1") == "1",
    "limit_mm_per_prompt": {"image": 4, "video": 0},
    **({"mm_processor_cache_gb": float(_MM_CACHE)} if _MM_CACHE else {}),
    "compilation_config": {
        "compile_mm_encoder": True,
        "cudagraph_mm_encoder": os.environ.get("GBC_VLM_MM_GRAPH", "1") == "1",
        **({"max_cudagraph_capture_size": _GRAPH_MAX} if _GRAPH_MAX else {}),
    },
    **({"scheduler_cls": _SCHEDULER} if _SCHEDULER else {}),
    **({"attention_backend": _ATTN} if _ATTN else {}),
    "structured_outputs_config": {"backend": os.environ.get("GBC_VLM_GRAMMAR", "auto")},
    **(
        {"logits_processors": ["kohakuvllm.grammar:JsonSchemaLogits"]}
        if _STRUCTURED == "kohaku"
        else {}
    ),
    "disable_log_stats": not _STATS,
}
_TOP = ("gpu_memory_utilization", "max_model_len", "max_num_seqs")
_INPROCESS = {
    "entry": "vllm",
    "model": _VLLM["model"],
    "units": _N,
    "devices": _VLM_CARDS,
    "capacity": _LANES,
    "lane": _VLM_LANE,
    "structured": _STRUCTURED,
    "stats_s": 5.0 if _STATS else 0.0,
    "frontend": os.environ.get("GBC_VLM_FRONTEND", "vllm"),
    "frontend_args": {
        "workers": int(os.environ.get("GBC_VLM_PREP_WORKERS", "4")),
        "vision_fp8": os.environ.get("GBC_VLM_VISION_FP8", "0") == "1",
    },
    **{k: _VLLM[k] for k in _TOP},
    "engine_args": {
        k: v for k, v in _VLLM.items() if k not in (*_TOP, "model", "served_model_name")
    },
}
_HTTP = {
    "entry": "api",
    "flavour": "vllm",
    "model": _VLLM["served_model_name"],
    "structured": _STRUCTURED,
    "metrics": "vllm",
    "endpoints": [{"base_url": _URL, "max_inflight": _LANES}],
    "lane": _VLM_LANE,
}

_ENGINES = {
    "vlm": _HTTP if _URL else _INPROCESS,
    "detect": {
        "entry": "torch",
        "model": os.environ.get("GBC_DETECT_MODEL", "facebook/sam3"),
        "dtype": os.environ.get("GBC_DETECT_DTYPE", "auto"),
        "fp8": None if _DETECT_FP8 == "none" else _DETECT_FP8,
        "fp4_scales": os.environ.get("GBC_DETECT_FP4_SCALES") or None,
        "fp8_mlp": os.environ.get("GBC_DETECT_FP8_MLP") or None,
        "fusion_buckets": "pow2",
        "vision_graph": os.environ.get("GBC_DETECT_VISION_GRAPH", "1") == "1",
        "max_crops": int(os.environ.get("GBC_DETECT_CROPS", "8")),
        "devices": [f"cuda:{c}" for c in _AUX_CARDS],
        "max_batch": int(os.environ.get("GBC_DETECT_BATCH", "8")),
        "former": os.environ.get("GBC_DETECT_FORMER", "size"),
        "wait_ms": 5.0,
        "memory_gb": float(os.environ.get("GBC_DETECT_GB", "3.5")),
        "priority": int(os.environ.get("GBC_DETECT_PRIORITY", "0")),
        "engine": {"schedule": os.environ.get("GBC_DETECT_SCHEDULE", "co_resident")},
    },
    "embed": {
        "entry": "torch",
        "model": os.environ.get("GBC_EMBED_MODEL", "google/embeddinggemma-300m"),
        "devices": ["cpu"] if _EMBED_CPU else [f"cuda:{c}" for c in _AUX_CARDS],
        "max_batch": 64,
        "former": os.environ.get("GBC_EMBED_FORMER", "size"),
        "memory_gb": 1.4,
    },
    "cpu": {"entry": "python", "workers": 8},
}
if _TAG:
    _ENGINES["tag"] = {
        "entry": "torch",
        "model": os.environ.get("GBC_TAG_MODEL", "pixai-labs/pixai-tagger-v1.0"),
        "devices": [f"cuda:{c}" for c in _AUX_CARDS],
        "fp8": os.environ.get("GBC_TAG_FP8", "auto"),
        "max_batch": int(os.environ.get("GBC_TAG_BATCH", "4")),
        "memory_gb": 2.5,
    }


def _on_card(slot: str) -> dict:
    """A slot's settings in a host that sees one card (as ``cuda:0``)."""
    s = dict(_ENGINES[slot])
    if slot == "vlm":
        return {**s, "units": 1, "devices": None}
    return {**s, "devices": ["cuda:0"]}


# GBC_HOSTING: process (the default): the machine's process holds connectors only, every engine
#   in an engine host that sees one card (CUDA_VISIBLE_DEVICES); inprocess: every engine in the
#   machine's process. GBC_HOST_LAYOUT (process): card (the default): one host per card for all
#   its engines (card<c>), the in-process machine's memory per card; split: one host for the VLM
#   (vlm<c>) and one for SAM 3 and EmbeddingGemma (aux<c>, loading after vlm<c>), one more CUDA
#   context per card: on the RTX 5090 at GBC_VLM_KV_GB=12 the aux host misses by 32 MiB.
_HOSTING = os.environ.get("GBC_HOSTING", "process")
_LAYOUT = os.environ.get("GBC_HOST_LAYOUT", "card")
if _HOSTING not in ("process", "inprocess") or _LAYOUT not in ("card", "split"):
    raise ValueError(f"GBC_HOSTING={_HOSTING!r} (process, inprocess), GBC_HOST_LAYOUT={_LAYOUT!r}")
engines = _ENGINES
if _HOSTING == "process" and not _URL:
    _hosts, _refs = {}, {"vlm": [], "detect": [], "embed": [], **({"tag": []} if _TAG else {})}
    for _c in dict.fromkeys(_VLM_CARDS + _AUX_CARDS):
        _env = {"CUDA_VISIBLE_DEVICES": _c}
        _slots = ["vlm"] if _c in _VLM_CARDS else []
        _slots += ["detect"] if _c in _AUX_CARDS else []
        _slots += ["embed"] if _c in _AUX_CARDS and not _EMBED_CPU else []
        _slots += ["tag"] if _c in _AUX_CARDS and _TAG else []
        if _LAYOUT == "card":
            # vLLM refuses to start below GBC_VLM_UTIL of the card free: the VLM loads first
            _hosts[f"card{_c}"] = {
                "env": _env,
                "engines": {s: _on_card(s) for s in _slots},
                "order": [["vlm"], ["detect", "embed", "tag"]],
            }
            for _s in _slots:
                _refs[_s].append(f"card{_c}")
            continue
        if "vlm" in _slots:
            _hosts[f"vlm{_c}"] = {"env": _env, "engines": {"vlm": _on_card("vlm")}}
            _refs["vlm"].append(f"vlm{_c}")
        if "detect" in _slots:
            # vLLM refuses to start below GBC_VLM_UTIL of the card free: the card's VLM first
            _aux = {s: _on_card(s) for s in _slots if s != "vlm"}
            _after = [f"vlm{_c}"] if "vlm" in _slots else []
            _hosts[f"aux{_c}"] = {"env": _env, "engines": _aux, "after": _after}
            for _s in _aux:
                _refs[_s].append(f"aux{_c}")
    if _EMBED_CPU:
        _hosts["embed_cpu"] = {
            "env": {"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": _EMBED_THREADS},
            "engines": {"embed": dict(_ENGINES["embed"])},
        }
        _refs["embed"].append("embed_cpu")
    engines = {
        "hosts": _hosts,
        **{s: {"entry": "remote", "hosts": refs} for s, refs in _refs.items()},
        "cpu": _ENGINES["cpu"],
    }

machine = {
    "vlm_waiting": int(os.environ.get("GBC_VLM_WAITING", str(_LANES * _N))),
    "waiting": _WAITING,
    "max_jobs": int(os.environ.get("GBC_MAX_JOBS", str(_JOBS_PER_CARD * _N))),
    "report": f"{_OUT}/report",
    "waits": {"every_s": float(_WAITS)} if _WAITS else None,
    "knobs": {"events": _EVENTS},
}
source = {"kind": "folder", "path": os.environ.get("GBC_IMAGES", "images")}
sink = {"kind": "graphs", "path": f"{_OUT}/graphs.jsonl"}
