# Size preset: mid graphs (the PoC's OWLv2 graphs' size). A configuration layer:
# gbc run configs/machine/<machine>.py configs/controls/mid.py

controls = {
    "min_nodes": 8,
    "max_nodes": 12,
    "max_rounds": 4,
    "max_top_level": 6,
    "max_instances_per_text": 4,
    "max_depth": 3,
    "box_score": 0.4,
}
