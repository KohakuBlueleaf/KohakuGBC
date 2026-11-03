"""Controls: one job input holding every choice about what the graph contains.

Defaults are the PoC's (``GbcControls`` and its run arguments) plus ``relate = "end"``.
"""

DEFAULTS = {
    # How the root is made: GBC's image query, or from existing captions (and tags)
    "root": "query",
    # Node budget on image + entity vertices: rounds stop at the maximum, breadth rounds run below
    # the minimum
    "min_nodes": 8,
    "max_nodes": 20,
    "max_rounds": 4,
    # Entity queries per round (None: as many as the budget allows)
    "round_size": None,
    "priority": "area_depth",
    "max_breadth_rounds": 1,
    "coverage_max_new": 4,
    # Fan-out caps and depth: the image is depth 0; regions deeper than max_depth are not
    # expanded
    "max_top_level": 6,
    "max_features_per_entity": 4,
    "max_instances_per_text": 6,
    "max_depth": 3,
    # Detection selection, the PoC's (tuned for OWLv2, whose near-duplicate boxes need a tight
    # NMS): kept above box_score, NMS at nms_iou (nms_iou_single for a single object), a box
    # containing another dropped
    "box_score": 0.1,
    "nms_iou": 0.2,
    "nms_iou_single": 0.05,
    "drop_containers": True,
    # Regions per VLM request; leaves captioned by the parent's reply
    "entity_batch": 1,
    "leaf_caption": "query",
    # Which relational questions, and where they are asked: relation questions on every vertex
    # with several children, or on the image vertex only (``relation_vertices``)
    "relational": "both",
    "relate": "end",
    "relation_vertices": "all",
    # The prompt set of the image, entity, relation and composition queries (None: ``json``, or
    # ``json_leafcap`` with leaf_caption "parent"); a set's missing files are its base set's
    "prompts": None,
    # GBC-Nest only (kohakugbc.nodes.nest, kohakugbc.nodes.shape)
    "probes": [],
    "probe_score": 0.5,
    "probe_gap": 0.5,
    "small_area": 0.0,
    "single_tie": 0.9,
    "head_overlap": 0.5,
    "group_check": True,
    "nest_check": True,
    # Pictures the VLM is sent: the longer side scaled down to this many pixels
    "max_side": 1568,
    # Same-object check: embedding similarity at least this
    "same_threshold": 0.725,
    "overlap_threshold": 0.85,
    # Seeded roots: tags at least this confident name elements; tags in the element prompt
    "tag_threshold": 0.35,
    "tag_hint": False,
    # Tags on the finished graph (kohakugbc.nodes.tags): where, from which set, and a region's
    # threshold as a multiple of the tag's category threshold, how far (logits) below the best
    # region a region still holds the tag; tag_placement "top": the image and the vertices whose
    # only parent is the image, nothing moved; tag_regions: crops tagged alone or pooled boxes
    "tagging": "none",
    "tag_source": "tagger",
    "tag_placement": "top",
    "tag_regions": "crop",
    "tag_region_scale": 1.0,
    "tag_region_margin": 1.0,
}

CHOICES = {
    "root": ("query", "text", "image_text", "tags"),
    "priority": ("area_depth", "area", "deep"),
    "leaf_caption": ("query", "parent"),
    "relational": ("both", "relation", "composition", "none"),
    "relate": ("end", "every_round"),
    "relation_vertices": ("all", "root"),
    "prompts": (None, "json", "json_leafcap", "json_nest"),
    "tagging": ("none", "image", "regions"),
    "tag_source": ("tagger", "given", "both"),
    "tag_placement": ("top", "place"),
    "tag_regions": ("crop", "pool"),
}


# Each program's defaults over DEFAULTS (GBC-Nest: the mid + sam3 presets, then its own)
PROGRAM_DEFAULTS = {
    "gbc": {},
    "gbc_nest": {
        "min_nodes": 8,
        "max_nodes": 12,
        "max_rounds": 4,
        "max_top_level": 16,
        "max_instances_per_text": 4,
        "max_depth": 3,
        "box_score": 0.3,
        "nms_iou": 0.7,
        "nms_iou_single": 0.05,
        "drop_containers": False,
        "relation_vertices": "root",
        "prompts": "json_nest",
        "probes": ["words", "logo", "comic panel", "person"],
        "small_area": 0.02,
    },
}


def complete(controls: dict | None, program: str = "gbc") -> dict:
    """The program's defaults updated by ``controls``; an unknown name or value is an error."""
    out = {**DEFAULTS, **PROGRAM_DEFAULTS[program]}
    for name, value in (controls or {}).items():
        if name not in DEFAULTS:
            raise ValueError(f"unknown control {name!r}")
        if name in CHOICES and value not in CHOICES[name]:
            raise ValueError(f"control {name}={value!r}: one of {CHOICES[name]}")
        out[name] = value
    if out["min_nodes"] > out["max_nodes"]:
        raise ValueError("control min_nodes is above max_nodes")
    if out["entity_batch"] < 1:
        raise ValueError("control entity_batch is at least 1")
    return out
