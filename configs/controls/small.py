# Size preset: small graphs (about half the PoC's vertices). A configuration layer:
# gbc run configs/machine/<machine>.py configs/controls/small.py

controls = {
    "min_nodes": 5,
    "max_nodes": 8,
    "max_rounds": 3,
    "max_top_level": 4,
    "max_instances_per_text": 3,
    "max_depth": 2,
    "box_score": 0.5,
}
