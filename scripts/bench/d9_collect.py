"""One card's d9 results (``plans/detector/rigs/d9.sh`` output) as one markdown summary: model
walls per configuration, the engine path against transformers, precision against fp64, the
memory plans, the embedder, and per op the kernel against torch.compile; printed and written to
``RUNS/d9/<label>/summary.md``.

    python scripts/bench/d9_collect.py RUNS/d9/<label>
"""

import json
import sys
from pathlib import Path


def model_walls(root: Path) -> list[str]:
    lines = [
        "| config | workload | wall ms | vision | fusion | text | peak alloc GiB |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("model_eager", "model_compile", "model_ours"):
        for stages in sorted((root / name).glob("*/stages.json")):
            r = json.loads(stages.read_text(encoding="utf-8"))
            for wname, w in r["workloads"].items():
                if not wname.startswith("w2"):
                    continue
                s = w["stages"]
                lines.append(
                    f"| {r['box_source']} | {wname} | {w['wall_ms']:.0f} "
                    f"| {s.get('vision', {}).get('ms', 0):.0f} "
                    f"| {s.get('fusion', {}).get('ms', 0):.0f} "
                    f"| {s.get('text', {}).get('ms', 0):.0f} "
                    f"| {w['peak']['allocated_gib']:.2f} |"
                )
    return lines


def engine(root: Path) -> list[str]:
    lines = [
        "| impl | workload | wall ms | peak alloc GiB | errors |",
        "| --- | --- | --- | --- | --- |",
    ]
    for stages in sorted((root / "engine").glob("*/stages.json")):
        r = json.loads(stages.read_text(encoding="utf-8"))
        for wname, w in r["workloads"].items():
            lines.append(
                f"| {r['box_source']} | {wname} | {w['wall_ms']:.0f} "
                f"| {w['peak']['allocated_gib']:.2f} | {len(w['errors'])} |"
            )
    return lines


def table_rows(path: Path, keep) -> list[str]:
    if not path.exists():
        return [f"(missing: {path.name})"]
    return [line for line in path.read_text(encoding="utf-8").splitlines() if keep(line)]


def main() -> int:
    root = Path(sys.argv[1])
    out = [
        f"# d9: {root.name}",
        "",
        "## Model (W2)",
        "",
        *model_walls(root),
        "",
        "## Engine path",
        "",
        *engine(root),
        "",
        "## Precision against fp64 (W2, logits)",
        "",
    ]
    for summary in sorted((root / "reference").glob("*.md")) or [root / "reference/summary.md"]:
        rows = table_rows(summary, lambda line: "| logits |" in line or "| boxes |" in line)
        out += rows
    out += ["", "## Memory plan", ""]
    for md in sorted((root / "memory").glob("*.md")):
        out += table_rows(md, lambda line: line.startswith("|"))
    out += ["", "## Embedder", ""]
    for name in ("embed", "embed_fp32"):
        out += table_rows(root / name / "embed.md", lambda line: line.startswith("| "))
    out += ["", "## Ops (ms; kernel = the tuned entry)", ""]
    for name in ("ops", "ops_mask"):
        for md in sorted((root / name).glob("ops-*.md")):
            seen = set()
            for line in md.read_text(encoding="utf-8").splitlines():
                cells = [c.strip() for c in line.split("|")]
                if len(cells) > 6 and cells[3] in ("torch", "compile", "triton", "tilelang"):
                    key = (cells[1], cells[2], cells[3])
                    if key not in seen and cells[5] in ("0", ""):
                        seen.add(key)
                        out.append(
                            f"| {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} "
                            f"| abs mean {cells[7]} |"
                        )
    text = "\n".join(out) + "\n"
    (root / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
