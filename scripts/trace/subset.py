"""Cut a few jobs out of recorded traces, with their images, for the test fixtures.

    python scripts/trace/subset.py runs/traces/poc30 tests/fixtures/traces --arms text batch4 \
        --jobs cc12m_000.webp danbooru_007.webp nozomi_003.webp
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from kohakugbc.nodes.specs.trace import FILES


def image_of(name: str, record: dict) -> str | None:
    match name:
        case "jobs":
            return record["key"]
        case "vlm" | "detect":
            return json.loads(record["key"])["image"] if record.get("key") else None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces")
    parser.add_argument("out")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--jobs", nargs="+", required=True)
    args = parser.parse_args()
    keep = set(args.jobs)
    out_root = Path(args.out)
    images = out_root / "images"
    images.mkdir(parents=True, exist_ok=True)

    for arm in args.arms:
        src, dst = Path(args.traces) / arm, out_root / arm
        dst.mkdir(parents=True, exist_ok=True)
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        image_root = Path(meta["image_root"])
        meta["image_root"] = "../images"
        (dst / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Embeddings and same-object decisions are per text: keep those of the kept jobs' names
        names: set[str] = set()
        for name in FILES:
            path = src / f"{name}.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            kept = []
            for line in lines:
                record = json.loads(line)
                if name in ("embed", "same"):
                    texts = {record.get("text"), record.get("a"), record.get("b")} - {None}
                    if texts & names:
                        kept.append(line)
                    continue
                if image_of(name, record) in keep:
                    # System prompts repeat on every request; fixtures keep the query only
                    if name == "vlm":
                        record["request"].pop("system", None)
                        line = json.dumps(record, ensure_ascii=False)
                    kept.append(line)
                    if name == "jobs":
                        names |= {i[2] for r in record["records"] for i in r["ids"] if i[2]}
                        names |= {c[2] for r in record["records"] for c in r["children"] if c[2]}
            (dst / f"{name}.jsonl").write_text("".join(f"{k}\n" for k in kept), encoding="utf-8")
        for job in keep:
            if not (images / job).exists():
                shutil.copy(image_root / job, images / job)
    return 0


if __name__ == "__main__":
    sys.exit(main())
