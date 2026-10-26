"""GBC graphs of the same images from several runs, side by side in one self-contained HTML page.

Each run is a folder holding ``graphs.jsonl`` (one GBC graph per image, matched across runs by
the image's file name), and optionally ``meta.json`` (``label``, ``model``, ``quant``, ``note``)
and ``summary.json`` (the run's summary from ``scripts/bench``). Per image the page shows every
run's image captions, then one column per run: the image with every entity's box, the graph as
a tree with each vertex's captions (grounded phrases in their child's colour), and the
relations. Images are embedded downscaled.

    python scripts/demo/compare.py OUT.html IMAGES_DIR RUN_DIR [RUN_DIR ...] [--fragment]

``--fragment`` writes the page without the document skeleton (for hosts that add their own).
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

PAGE = Path(__file__).with_name("compare.html")


def load_run(folder: Path) -> dict:
    meta = {"label": folder.name}
    if (folder / "meta.json").exists():
        meta.update(json.loads((folder / "meta.json").read_text(encoding="utf-8")))
    summary = {}
    if (folder / "summary.json").exists():
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    graphs = {}
    with open(folder / "graphs.jsonl", encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            graphs[Path(g["img_path"]).name] = {"vertices": g["vertices"]}
    return {
        "meta": meta,
        "summary": {k: summary[k] for k in ("jobs", "run_s", "jobs_per_s") if k in summary},
        "graphs": graphs,
    }


def embed(path: Path, side: int) -> dict:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    image.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=85)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"src": f"data:image/jpeg;base64,{data}", "w": w, "h": h}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("images")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--side", type=int, default=1024, help="embedded image's longer side")
    parser.add_argument("--title", default="GBC model comparison")
    parser.add_argument("--fragment", action="store_true")
    args = parser.parse_args()

    runs = [load_run(Path(r)) for r in args.runs]
    keys = sorted(set().union(*(r["graphs"] for r in runs)))
    images = {k: embed(Path(args.images) / k, args.side) for k in keys}
    data = {"title": args.title, "keys": keys, "images": images, "runs": runs}
    # "</" inside the JSON would close the script element
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = PAGE.read_text(encoding="utf-8")
    page = page.replace("__TITLE__", args.title).replace("__DATA__", blob)
    if not args.fragment:
        page = (
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, '
            'viewport-fit=cover">\n</head>\n<body>\n' + page + "\n</body>\n</html>\n"
        )
    Path(args.out).write_text(page, encoding="utf-8")
    sizes = ", ".join(f"{r['meta']['label']}: {len(r['graphs'])}" for r in runs)
    print(f"{args.out}: {len(keys)} image(s); graphs per run: {sizes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
