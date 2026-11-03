# The root alone (the image query's caption, one VLM request a job); the machine can leave SAM 3
# and EmbeddingGemma out (GBC_DETECT=0 GBC_EMBED=0). Layer: ... configs/controls/root.py

controls = {
    "max_top_level": 0,
    "max_rounds": 0,
    "max_breadth_rounds": 0,
    "relational": "none",
}
