# The whole GBC machine on one RTX 4090: Qwen3.5 in FP8 (RedHatAI/Qwen3.5-4B-FP8-dynamic) behind
# a vLLM server, GBC's own SAM 3 (fp8 row) and EmbeddingGemma in this process on the same card.
# GBC_VLM_LANES sets the VLM requests in flight and waiting (64: on 30 images VLM throughput
# stops rising there, 7.5 requests/s; 96-192 only queue inside vLLM); GBC_EVENTS=all logs every
# engine call (scripts/docs/machine_view.py draws it). Start the server first (WSL or Linux),
# sized to leave the card's rest to this process:
#   vllm serve RedHatAI/Qwen3.5-4B-FP8-dynamic --served-model-name qwen3.5-4b \
#       --gpu-memory-utilization 0.56 --max-model-len 8192 --max-num-seqs 256
# GBC_IMAGES=path/to/images gbc run configs/machine/qwen35_fp8_4090.py

import os

_OUT = os.environ.get("GBC_OUT", "runs/machine/qwen35_fp8_4090")
_VLM = os.environ.get("GBC_VLM_URL", "http://127.0.0.1:8000/v1")
_LANES = int(os.environ.get("GBC_VLM_LANES", "64"))
_EVENTS = os.environ.get("GBC_EVENTS", "sampled")

engines = {
    "vlm": {
        "entry": "api",
        "protocol": "openai",
        "flavour": "vllm",
        "model": os.environ.get("GBC_VLM_MODEL", "qwen3.5-4b"),
        "endpoints": [{"base_url": _VLM, "max_inflight": _LANES}],
        "timeout_s": 600,
        "metrics": "vllm",
    },
    "detect": {
        "entry": "torch",
        "model": "facebook/sam3",
        "dtype": "bfloat16",
        "fp8": "row",
        # 5 graphs captured at load instead of 16: 2.7 GiB reserved where exact takes 5.4
        "fusion_buckets": "pow2",
        "devices": ["cuda:0"],
        "max_batch": 4,
        "wait_ms": 5.0,
    },
    "embed": {"entry": "torch", "devices": ["cuda:0"], "max_batch": 64},
    "cpu": {"entry": "python", "workers": 8},
}
machine = {"vlm_waiting": _LANES, "report": f"{_OUT}/report", "knobs": {"events": _EVENTS}}
source = {"kind": "folder", "path": os.environ.get("GBC_IMAGES", "images")}
sink = {"kind": "graphs", "path": f"{_OUT}/graphs.jsonl"}
