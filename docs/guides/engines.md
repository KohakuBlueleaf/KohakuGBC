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
