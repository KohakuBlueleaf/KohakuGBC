# The whole GBC machine on one GPU: the qwen35 example engine as the VLM, the detector on the
# same device, the embedder on the CPU; graphs of every image under GBC_IMAGES.
# pip install -e examples/qwen35
# GBC_IMAGES=path/to/images gbc run configs/machine/qwen35_local.py

import os

_MODEL = os.environ.get("QWEN35_MODEL", "Qwen/Qwen3.5-4B")
_OUT = os.environ.get("GBC_OUT", "runs/machine/qwen35_local")

engines = {
    # 24 GiB: the 4B weights (8.5 GiB), bfloat16 states for 64 sequences, ~70k tokens of KV
    "vlm": {
        "entry": "qwen35", "model": _MODEL, "device": "cuda:0", "memory": 0.65,
        "state_dtype": "bfloat16",
    },  # fmt: skip
    "detect": {"entry": "torch", "devices": ["cuda:0"], "max_batch": 8, "wait_ms": 5.0},
    "embed": {"entry": "torch", "devices": ["cpu"], "max_batch": 64},
    "cpu": {"entry": "python", "workers": 8},
}
machine = {"vlm_waiting": 64, "report": f"{_OUT}/report"}
source = {"kind": "folder", "path": os.environ.get("GBC_IMAGES", "images")}
sink = {"kind": "graphs", "path": f"{_OUT}/graphs.jsonl"}
