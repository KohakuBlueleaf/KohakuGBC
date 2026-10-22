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
| `engine_args`, `sampling` | any other `AsyncEngineArgs` field; extra `SamplingParams` fields |

The unit builds the prompt with the model's chat template (thinking off), sends the pictures as
the API path decodes them, and constrains the reply to the request's JSON schema; a cancelled
request is aborted in vLLM. A script that builds a machine with this entry needs an
`if __name__ == "__main__":` guard: vLLM starts its engine core by spawning a process.

## The VLM in-process (the qwen35 example)

A Qwen3.5 engine written for this workload (`pip install -e examples/qwen35`; Windows and
Linux), one unit per device:

```python
engines = {"vlm": {"entry": "qwen35", "model": "Qwen/Qwen3.5-4B", "memory": 0.65}}
```

Its settings, design and tests: [examples/qwen35](../../examples/qwen35/README.md).
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
| `dtype` | float32 by default; `"auto"` picks bfloat16 or float16 by the device |
| `max_batch`, `wait_ms` | batch size; how long a batch waits for more requests |
| `memory_gb` | the placement plan's estimate per device |
| `engine` | Togeari torch engine options: `schedule`, `isolation`, `slots`, `memory_cap_gb`, `prep_workers` |
| `box_threshold`, `max_texts` | the detector's box score floor (0.05) and texts per forward row (16) |

SAM 3 boxes what OWLv2 misses on illustrated images. On an anime-style portrait, OWLv2's best
"girl" box (0.415) covers her legs only; SAM 3 gives one full-body box (0.89) and, inside it,
her hair (0.94) and her tail (0.91) as separate boxes, where OWLv2 boxed the tail as "hair". It
runs its vision encoder once per crop and one decoder pass per text; batched equals single.
`configs/machine/qwen35_sam3_local.py` layers it over the one-GPU configuration, in bfloat16.

The detector's default is float32 because its results then do not depend on its batch: in
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
