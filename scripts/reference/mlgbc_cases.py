"""Reference cases for graph resolution, computed by ml-gbc itself.

Runs in an environment where ml-gbc's ``gbc`` package imports (the PoC's venv). For every image
of the given ``node_infos.jsonl`` files, and for seeded generated record sets, it writes the
records (``kohakugbc.graph.records`` form), the image size ml-gbc read, and the digest and
counts of ``node_infos_to_gbc_graph``'s result, or that ml-gbc raised.

    <poc venv>/python scripts/reference/mlgbc_cases.py OUT.jsonl.gz NODE_INFOS.jsonl ...
"""

import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from gbc.captioning.conversion.to_gbc_graph import node_infos_to_gbc_graph
from gbc.captioning.primitives import NodeInfo

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from kohakugbc.graph.digest import digest  # noqa: E402
from kohakugbc.graph.records import from_node_info  # noqa: E402

ENTITY_INPUT = ["gbc.captioning.primitives.action_io", "ActionInput"]


def mlgbc(dumps: list[dict], img_path: str) -> dict:
    """ml-gbc's graph of one image's NodeInfo dumps: digest and counts, or the error."""
    try:
        infos = [NodeInfo.model_validate(d) for d in dumps]
        graph = node_infos_to_gbc_graph(infos)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    if graph is None:
        return {"digest": digest(None), "img_size": None, "stats": None}

    # The image path as the case names it, so local folders do not enter the fixture
    img_size = None if graph.img_size is None else list(graph.img_size)
    graph.img_path = img_path
    dump = graph.model_dump()
    labels = defaultdict(int)
    for v in graph.vertices:
        labels[v.label] += 1
    try:
        depth = graph.depth
        roots = graph.n_roots
    except ValueError:
        depth = roots = None
    return {
        "digest": digest(dump),
        "img_size": img_size,
        "stats": {
            "vertices": graph.n_vertices,
            "labels": dict(sorted(labels.items())),
            "edges": graph.n_edges,
            "captions": graph.n_captions,
            "depth": depth,
            "roots": roots,
            "leaves": graph.n_leaves,
        },
    }


def case(name: str, img_path: str, dumps: list[dict]) -> dict:
    out = {"name": name, "img_path": img_path, "records": [from_node_info(d) for d in dumps]}
    out.update(mlgbc(dumps, img_path))
    return out


# ---- recorded runs ---------------------------------------------------------------------------


def recorded(path: Path) -> list[dict]:
    by_image = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            by_image[d["input_dict"]["image"]["img_path"]].append(d)
    # The case names the image by its file name; ml-gbc reads the size from the real path
    run = path.parent.parent.name
    return [
        case(f"{run}/{Path(img_path).name}", Path(img_path).name, dumps)
        for img_path, dumps in by_image.items()
    ]


# ---- generated record sets -------------------------------------------------------------------


def info(ids: list, box, descs: list, children: list, img: str) -> dict:
    """A NodeInfo dump: ids [(id, label, text)], descs [(text, label, assoc)], children."""
    entity_info = [{"label": lab, "text": t, "entity_id": i} for i, lab, t in ids]
    return {
        "input_type": ENTITY_INPUT,
        "input_dict": {
            "image": {"img_path": img, "img_url": None, "image": None},
            "entity_info": entity_info[0] if len(entity_info) == 1 else entity_info,
            "bbox": box,
        },
        "query_result": {
            "descs": [[{"text": t, "label": lab}, list(a)] for t, lab, a in descs],
            "entities": [[{"label": lab, "text": t, "entity_id": i}, []] for i, lab, t in children],
            "raw": None,
        },
    }


def box_in(rng: random.Random, outer: dict | None, conf=True) -> dict:
    o = outer or {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}
    w = o["right"] - o["left"]
    match rng.random():
        case r if r < 0.08:
            # Exactly the parent box, or a degenerate one
            b = dict(o)
        case r if r < 0.12:
            x = o["left"] + w * rng.random()
            b = {"left": x, "top": o["top"], "right": x, "bottom": o["bottom"]}
        case r if r < 0.2:
            # Partly outside, around the 0.85 overlap threshold
            x = o["left"] + w * rng.uniform(0.0, 0.3)
            b = {"left": x, "top": o["top"], "right": x + w * 0.8, "bottom": o["bottom"]}
        case _:
            a, c = sorted(rng.uniform(o["left"], o["right"]) for _ in range(2))
            t, d = sorted(rng.uniform(o["top"], o["bottom"]) for _ in range(2))
            b = {"left": a, "top": t, "right": c, "bottom": d}
    b["confidence"] = round(rng.random(), 3) if conf else None
    return b


NAMES = ["girl", "cup", "table", "hand", "hat", "window", "tree", "dog", "ball", "chair"]
# Ids Python's str.isdigit treats unusually: "y_²" is numbered, "z_½" is not
ODD = ["café", "x²", "a_b", "日本", "two  words", " ", "y_²", "z_½"]


def generated(rng: random.Random) -> list[dict]:
    img = "generated.png"
    dumps = []
    image_descs = [("A scene.", "detail", [])]
    if rng.random() < 0.6:
        image_descs.append(("Short.", "short", []))
    if rng.random() < 0.2:
        image_descs.append(("Orig.", "original", []))
    if rng.random() < 0.1:
        image_descs.append(("Second detail.", "detail", []))

    # Top-level names, some numbered instances with or without a composition vertex
    entities = []
    pool = NAMES + (ODD if rng.random() < 0.3 else [])
    for name in rng.sample(pool, k=min(len(pool), rng.randint(1, 6))):
        count = rng.choice([1, 1, 1, 2, 3])
        if count == 1:
            entities.append((name, name))
        else:
            entities += [(f"{name}_{k}", name) for k in range(1, count + 1)]
            if rng.random() < 0.5:
                entities.append((name, name + "s"))
    image_children = [(i, "entity", t) for i, t in entities]
    if rng.random() < 0.2:
        image_children.append(("ghost", "entity", "ghost"))
    if rng.random() < 0.2 and image_children:
        image_children.append(rng.choice(image_children))
    if rng.random() < 0.1:
        image_children.append(("blank", "entity", "  "))
    if rng.random() < 0.9:
        dumps.append(info([("", "image", None)], None, image_descs, image_children, img))

    boxes = {}
    ids_used = []
    for entity_id, text in entities:
        box = box_in(rng, None)
        boxes[entity_id] = box
        label = (
            "composition"
            if not entity_id.split("_")[-1].isdigit() and text.endswith("s") and text != entity_id
            else "entity"
        )
        if rng.random() < 0.05:
            label = "single"
        ids = [(entity_id, label, text)]
        # Joins: other regions of the same object answer under this vertex
        if label == "entity" and rng.random() < 0.25:
            ids += [(f"{entity_id}~{k}", label, text) for k in range(rng.randint(1, 2))]
        descs = [] if rng.random() < 0.1 else [(f"About {text}.", "detail", [])]
        children = []
        if label == "entity" and rng.random() < 0.5:
            for feat in rng.sample(NAMES, k=rng.randint(1, 3)):
                fid = f"{entity_id}_{feat}"
                children.append((fid, "entity", feat))
                if rng.random() < 0.7:
                    fbox = box_in(rng, box)
                    boxes[fid] = fbox
                    fdescs = [] if rng.random() < 0.1 else [(f"A {feat}.", "detail", [])]
                    dumps.append(info([(fid, "entity", feat)], fbox, fdescs, [], img))
                    if fdescs:
                        ids_used.append(fid)
        if label == "composition":
            children = [
                (i, "entity", t)
                for i, t in entities
                if i.rsplit("_", 1)[0] == entity_id and i != entity_id
            ]
        dumps.append(info(ids, box, descs, children, img))
        if descs:
            ids_used += [i for i, _, _ in ids]

    # Relational answers over existing ids (joined, numbered, stems), on the image or an entity
    described = [i for i in ids_used if i in boxes or "~" in i]
    for _ in range(rng.randint(0, 3)):
        if len(described) < 1:
            break
        parent = "" if rng.random() < 0.7 or not described else rng.choice(described)
        descs = []
        for _ in range(rng.randint(1, 3)):
            assoc = rng.sample(described, k=min(len(described), rng.randint(1, 3)))
            descs.append((f"Relation {assoc}.", rng.choice(["relation", "composition"]), assoc))
        if descs and rng.random() < 0.3:
            descs.append(descs[0])
        dumps.append(info([(parent, "relation", None)], None, descs, [], img))

    if rng.random() < 0.3:
        rng.shuffle(dumps)
    return dumps


def errors(rng: random.Random) -> list[tuple[str, list[dict]]]:
    img = "error.png"
    base = [info([("", "image", None)], None, [("A.", "detail", [])], [("a", "entity", "a")], img)]
    a = info(
        [("a", "entity", "a")],
        {"left": 0.1, "top": 0.1, "right": 0.5, "bottom": 0.5},
        [("A.", "detail", [])],
        [],
        img,
    )
    return [
        ("duplicate id", base + [a, a]),
        (
            "relation names an unknown id",
            base + [a, info([("", "relation", None)], None, [("R.", "relation", ["zz"])], [], img)],
        ),
        (
            "relation on an unknown vertex",
            base
            + [a, info([("zz", "relation", None)], None, [("R.", "relation", ["a"])], [], img)],
        ),
        (
            "bad desc label",
            base + [info([("a", "entity", "a")], None, [("A.", "nonsense", [])], [], img)],
        ),
        (
            "child without text",
            [
                info(
                    [("", "image", None)],
                    None,
                    [("A.", "detail", [])],
                    [("a", "entity", None)],
                    img,
                ),
                a,
            ],
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("node_infos", nargs="*")
    parser.add_argument("--generated", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cases = []
    for path in args.node_infos:
        cases += recorded(Path(path))
    rng = random.Random(args.seed)
    for k in range(args.generated):
        cases.append(case(f"generated/{k}", "generated.png", generated(rng)))
    for name, dumps in errors(rng):
        cases.append(case(f"error/{name}", "error.png", dumps))

    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    failed = sum(1 for c in cases if "error" in c)
    print(f"{len(cases)} cases, {failed} where ml-gbc raises -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
