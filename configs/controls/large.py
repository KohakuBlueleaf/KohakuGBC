# Size preset: large graphs (more vertices, deeper rounds). A configuration layer:
# gbc run configs/machine/<machine>.py configs/controls/large.py

controls = {
    "min_nodes": 12,
    "max_nodes": 30,
    "max_rounds": 6,
    "max_top_level": 8,
    "max_instances_per_text": 6,
    "max_depth": 3,
    "box_score": 0.2,
}
