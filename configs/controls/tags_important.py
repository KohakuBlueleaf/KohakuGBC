# Tags on the graph: the image's whole set on the image vertex; the vertices holding at least
# tag_min_tags of them (pooled scores) tagged again from their own crops, at most
# tag_max_regions; every other vertex none. Needs a tag engine (GBC_TAG=1 on the card machine):
# gbc run configs/machine/qwen35_card.py configs/programs/gbc_nest.py \
#     configs/controls/tags_important.py

controls = {
    "tagging": "regions",
    "tag_source": "tagger",
    "tag_select": "important",
    "tag_regions": "crop",
}
