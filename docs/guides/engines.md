# Configuring engines

Each slot of the machine (`vlm`, `detect`, `embed`, `cpu`) names one entry with its settings in
the configuration's `engines`. Configuration files are Python files; later files on the
command line merge over earlier ones.

## The VLM over an API

Any OpenAI-compatible server (vLLM, SGLang, llama.cpp's server, OpenAI, OpenRouter) or
Anthropic's Messages API, one unit per endpoint:

```python
engines = {
    "vlm": {
        "entry": "api",
        "protocol": "openai",
        "flavour": "vllm",
        "model": "qwen3.5-4b",
        "endpoints": [
            {"base_url": "http://10.0.0.5:8000/v1", "max_inflight": 64},
            {"base_url": "http://10.0.0.6:8000/v1", "max_inflight": 64},
        ],
        "metrics": "vllm",
    }
}
```

| Setting | Meaning |
|---|---|
| `protocol` | `"openai"` (`/chat/completions`) or `"anthropic"` (`/messages`) |
| `flavour` | where an OpenAI-compatible server takes the JSON schema and how it turns Qwen3.5's thinking off: `vllm`, `sglang`, `llama.cpp`, `openai`, `openrouter` |
| `model`, `sampling` | the model name; extra body fields (`top_p`, `min_p`, ...) |
| `structured` | `grammar` (default): the request's JSON schema goes to the server; `kohaku`: it goes as `vllm_xargs` to KohakuvLLM's grammar logits processor on a vLLM server (served with `--logits-processors kohakuvllm.grammar:JsonSchemaLogits`); `prompt`: the prompt alone asks for the JSON |
| `endpoints` | per unit: `base_url`, `max_inflight` (32), `rate` (requests per second, or `[rate, burst]`), `api_key_env`, `headers`, `name` |
| `timeout_s`, `connect_timeout_s`, `health_path` | per request; per connection (past it the unit is down); the GET that brings a down unit back |
| `metrics` | `"vllm"`: the server's running and waiting counts in the unit's load report |

Every request carries `X-GBC-Request`, the SHA-256 of what it asks, so a cache or a log in
front of the server can tell repeated questions apart.

## The VLM in-process (vLLM)

vLLM's engine in the machine's process, one unit per `AsyncLLM` (Linux; `pip install
"kohakugbc[vllm]"`):

```python
engines = {
    "vlm": {
        "entry": "vllm",
        "model": "Qwen/Qwen3.5-4B",
        "units": 2,
        "devices": [[0], [1]],
        "gpu_memory_utilization": 0.5,
        "max_num_seqs": 64,
    }
}
```

| Setting | Meaning |
|---|---|
| `model` | a checkpoint name or folder |
| `units`, `devices` | units, and per unit the CUDA devices it uses (several: tensor parallel over them) |
| `gpu_memory_utilization`, `max_model_len`, `max_num_seqs`, `dtype` | vLLM's own (0.5, 16384, 64, `bfloat16`) |
| `capacity` | requests in flight per unit (twice `max_num_seqs`) |
| `structured`, `retries` | `prompt` (default): the prompts describe the JSON, and a reply GBC's reader cannot read as the schema's object is asked again up to `retries` (1) times; `grammar`: vLLM's structured outputs constrain decoding to the schema; `kohaku`: KohakuvLLM's grammar logits processor does (the entry adds it to the engine's `logits_processors`) |
| `engine_args`, `sampling` | any other `AsyncEngineArgs` field; extra `SamplingParams` fields |
| `stats_s` | vLLM's stats line every that many seconds (0: off) |
| `frontend`, `frontend_args`, `frontend_gb` | `vllm` (default): vLLM's own input processing and image encoder; `kohakuvllm`: KohakuvLLM's frontend, requests prepared in worker processes (`workers`) and images encoded by its vision tower in this process (`vision_fp8`, `max_patches`), handed to the engine as token ids and image embeddings; the tower's reservation in GiB (1.5) |

The unit builds the prompt with the model's chat template (thinking off) and sends the pictures
as the API path decodes them; a cancelled request is aborted in vLLM. vLLM sizes its KV cache
from `gpu_memory_utilization` less what the device holds while it profiles; other units of the
machine loading on the same device then count against it (a third less KV on an RTX 5090 with
SAM 3 and EmbeddingGemma beside it). `engine_args={"kv_cache_memory_bytes": ...}` sets the KV
cache outright. A script that builds a
machine with this entry needs an `if __name__ == "__main__":` guard: vLLM starts its engine core
by spawning a process.

[KohakuvLLM](https://github.com/KohakuBlueleaf/KohakuvLLM) is our own frontend and kernels for
this engine, a drop-in vLLM plugin package (no fork or rebuild; its TESTED.md names the vLLM
versions each part was verified on). Installed beside vLLM (`pip install
"kohakuvllm @ git+https://github.com/KohakuBlueleaf/KohakuvLLM"`), its parts are opted into
through the same settings, in-process or behind `vllm serve`: for example
`engine_args={"scheduler_cls": "kohakuvllm.sched.PhasedScheduler"}`.
`configs/machine/qwen35_card.py` is the whole machine on one card with either path (`GBC_VLM_URL`
for a server, `scripts/bench/machine/serve.py` to start one with the same engine arguments) and
the measured best settings.

## The VLM in-process (qwen35)

A Qwen3.5 engine written for this workload (`pip install "kohakugbc[qwen35]"`; Windows and
Linux), one unit per device:

```python
engines = {"vlm": {"entry": "qwen35", "model": "Qwen/Qwen3.5-4B", "memory": 0.65}}
```

Its settings, design and tests: [kohakugbc_models/vlm/qwen35](../../src/kohakugbc_models/vlm/qwen35/README.md).
`configs/machine/qwen35_local.py` runs the whole machine on one GPU with it.

## The vision models (torch)

The detector (OWLv2) and the embedder (EmbeddingGemma) on Togeari's torch engine
(`pip install "kohakugbc[torch]"`), one unit per placement:

```python
engines = {
    "detect": {"entry": "torch", "devices": "auto", "max_batch": 8, "wait_ms": 5.0},
    "embed": {"entry": "torch", "devices": ["cpu"], "max_batch": 64},
}
```

| Setting | Meaning |
|---|---|
| `model` | the detector: `google/owlv2-base-patch16-ensemble` (the PoC's) or `facebook/sam3`; the embedder: `google/embeddinggemma-300m` |
| `devices` | `"auto"` (every CUDA device, else the CPU), a device, or a list: one unit per device |
| `dtype` | float32 by default (GBC's own SAM 3: `"auto"`); `"auto"` picks bfloat16 or float16 by the device |
| `max_batch`, `wait_ms` | batch size; how long a batch waits for more requests |
| `former`, `budgets` | `"size"` (default): up to `max_batch` requests; `"budget"`: Togeari's `BudgetFormer` over each request's costs (the detector's `crops` and `pairs`, the embedder's `texts` and `chars`; `budgets` overrides the defaults), a request over a budget runs alone. On the RTX 5090 machine `budget` halves SAM 3's batches and costs 4% img/s |
| `memory_gb` | the placement plan's estimate per device |
| `engine` | Togeari torch engine options: `schedule`, `isolation`, `slots`, `memory_cap_gb`, `prep_workers` |
| `box_threshold`, `max_texts` | the detector's box score floor (0.05) and texts per forward row (16) |

SAM 3 boxes what OWLv2 misses on illustrated images. On an anime-style portrait, OWLv2's best
"girl" box (0.415) covers her legs only; SAM 3 gives one full-body box (0.89) and, inside it,
her hair (0.94) and her tail (0.91) as separate boxes, where OWLv2 boxed the tail as "hair".
`configs/machine/qwen35_sam3_local.py` layers it over the one-GPU configuration, in bfloat16.

### SAM 3: GBC's own model

For `facebook/sam3` (any `sam3` / `sam3_video` checkpoint) the entry builds GBC's own SAM 3
(`kohakugbc_models.detect.sam3`, `impl: "auto"`; `impl: "transformers"` keeps transformers' model,
which SAM3-LiteText checkpoints use): each crop's vision features once, each unique text once
(cached), (crop, text) pairs reading their crop's features in place, fused kernels per arch
(`kohakugbc_models.kernels`: Triton on sm80+, TileLang on sm70 / sm75, torch.compile and eager as
fallbacks, every choice timed per problem against the PyTorch paths), the fusion pass and the
text encoder replayed from CUDA graphs. 2.3x transformers' model through the engine on a 5090
(bf16), more with fp8 ([performance](../performance/detector.md)).

| Setting | Meaning |
|---|---|
| `dtype` | `"auto"` (bfloat16 on sm80+, float16 before) |
| `max_pairs`, `max_crops` | (crop, text) pairs per fusion pass (16), crops per vision pass (4) |
| `box_source` | `pred` (the predicted boxes), `mask` (the mask's box on its 288 grid, in one kernel, no logits written), `mask_full` (the mask resized to the crop first); `mask_threshold` 0.5 |
| `kernels`, `fusion_graph`, `fusion_buckets`, `text_graph` | the kernel registry's ops (true), the fusion pass replayed from CUDA graphs (true) per `exact` pass size (or `pow2`: fewer graphs, padded passes), the text encoder replayed (true) |
| `fp8`, `fp8_encoder` | the ViT's linears in fp8: `row` (sm89+) or `mx` (sm100+); the DETR encoder's MLPs too (measured inside the noise: off) |
| `text_device` | `"cpu"`: the text encoder in float32 on the host (0.66 GiB less on the device; the cache serves repeated texts) |
| `text_cache` | texts whose features are kept (4096, least recently used dropped) |
| `budget_gb` | fit `max_pairs` / `max_crops` within this many GiB from the card's measured memory plan (`scripts/bench/detector/memory.py`); `memory_gb` defaults to the plan's estimate |
| `capture` | `load` (default): every fusion pass size and text count captured into CUDA graphs when the model loads, other shapes run uncaptured; `lazy`: each captured on its first call. On one card beside other models, `load` with `fusion_buckets: pow2` keeps 2.7 GiB reserved where `exact` takes 5.4 |
| `retry_singular` | a text no box answers is asked again with its last word singular, its boxes kept under the text (true; SAM 3 answers nothing for some plural phrases: "devil girls" none, "devil girl" 0.75); for transformers' SAM 3 too |

Kernels tune on first use per problem and store their choice under `KOHAKUGBC_KERNEL_CACHE`
(default `~/.cache/kohakugbc_kernels`); the memory plans live there too. Setting
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the process environment lowers the
allocator's reserve by about 1 GiB.

### EmbeddingGemma: GBC's own model

For `gemma3_text` checkpoints (`google/embeddinggemma-300m`) the embed entry builds GBC's own
EmbeddingGemma (`kohakugbc_models.embed.gemma`, `impl: "auto"`; `impl: "st"` keeps
sentence-transformers): a batch's texts packed into one token stream (no padding), a
variable-length bidirectional attention kernel, the pass replayed from one CUDA graph per stream
bucket. float32 by default (equal to sentence-transformers' vectors, 1 - cos <= 1e-7);
`residual` (float32) keeps the residual stream in fp32 when `dtype` is 16-bit. 1.9-8.3x
sentence-transformers' throughput on a 5090 in float32. `capture` (`load`) captures every stream
bucket up to `graph_tokens` (4096) at load; longer streams run uncaptured.

OWLv2's default is float32 because its results then do not depend on its batch: in
float16 a batched crop's scores move by up to 0.085 from the crop alone, in float32 by at most
1.3e-4 (no detection changes). Float16 with `max_batch` 1 is how the PoC ran it, and gives the
PoC's boxes exactly. The detector needs torchvision: transformers' slower image processor
resizes differently and gives other boxes.

`gbc doctor CONFIG ...` prints the devices, the entries, and the configuration's placement plan.

## Replay

Pseudo engines that answer from a recorded trace (`runs/traces/...`), for running the program
and the machine without models:

```python
_TRACE = "runs/traces/poc30/text"
engines = {
    "vlm": {"entry": "replay", "trace": _TRACE, "capacity": 32},
    "detect": {"entry": "replay", "trace": _TRACE},
    "embed": {"entry": "replay", "trace": _TRACE},
}
source = {"kind": "trace", "path": _TRACE}
```

`python -m kohakugbc.engines.replay.server TRACE --port 8088` serves a trace as an
OpenAI-compatible and Anthropic-compatible chat server, for the `api` entry without a model.

## Engines on other nodes

Run `gbc host CONFIG --slot vlm --listen 0.0.0.0:7501` on the engine's node, with the shared key
in `KOHAKUGBC_HOST_KEY`, and name it in the machine's configuration:

```python
engines = {
    "vlm": {
        "entry": "remote",
        "hosts": ["10.0.0.5:7501/replay-vlm-0"],
        "runs": {"entry": "replay", "trace": "runs/traces/poc30/text"},
    }
}
```

`runs` is the entry the host runs, with its settings: the lane takes its adapter, batching and
lane options ([engines on other nodes](../internals/engine-host.md)).

## CPU workers

```python
engines = {"cpu": {"entry": "python", "workers": 8, "mode": "thread"}}
```

`mode = "process"` runs the CPU nodes (crops, pictures, parsing, resolution) in worker processes.
