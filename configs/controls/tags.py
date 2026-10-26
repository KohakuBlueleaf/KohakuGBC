# Tags on the graph: the image's on the image vertex, each crop's own on the vertices whose only
# parent is the image. Needs a tag engine (GBC_TAG=1 on the card machine):
# gbc run configs/machine/qwen35_card.py configs/controls/mid.py ... configs/controls/tags.py

controls = {
    "tagging": "regions",
    "tag_source": "tagger",
    "tag_placement": "top",
    "tag_regions": "crop",
}
