# configs/machine/qwen35_local.py with SAM 3 as the detector: on illustrations it boxes whole
# people and their parts where OWLv2 boxes a part. On one 24 GiB device beside the VLM the
# device fills: one portrait image took 79 s against 34 s with OWLv2 (RTX 4090, Windows).
# GBC_IMAGES=path/to/images gbc run configs/machine/qwen35_local.py \
#     configs/machine/qwen35_sam3_local.py

engines = {
    "vlm": {"memory": 0.55, "max_seqs": 32, "vision_patches": 16384},
    "detect": {"model": "facebook/sam3", "dtype": "bfloat16", "max_batch": 4},
}
