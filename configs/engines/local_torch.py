# The vision models on this process's devices: the detector on every CUDA device (one unit per
# device), the embedder on the CPU. Layer it under a VLM configuration.

engines = {
    "detect": {"entry": "torch", "devices": "auto", "max_batch": 8, "wait_ms": 5.0},
    "embed": {"entry": "torch", "devices": ["cpu"], "max_batch": 64},
}
