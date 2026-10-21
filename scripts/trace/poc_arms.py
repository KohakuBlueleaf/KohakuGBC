"""Record the PoC arms the program is checked against, one trace each (``record_poc.py``).

Every arm runs the same images (the first ``--per_source`` seeds of each source) on the PoC's
text-seeded base configuration (Qwen3.5-4B behind vLLM, OWLv2, EmbeddingGemma same-object
check), changing one setting: the root (text, image and text, tags, image query), entity batch
4, or leaf captions from the parent. The vLLM server must be up (PoC
``scripts/vllm_serve.sh``).

    python scripts/trace/poc_arms.py --poc ../../../../nn/KohakuGBC/poc/gbc-poc --out runs/traces
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

BASE = [
    "--config", "configs/local_v3_json.yaml",
    "--prompt_dir", "prompts/json",
    "--controls", "target_min_nodes=8", "target_max_nodes=20",
    "--replace", "shared.detection_model=configs/components/detector_owlv2.yaml",
    "--override", "shared.detection_score_threshold=0.1", "shared.vlm.model=qwen3.5-4b",
    "shared.vlm.base_url=http://127.0.0.1:8000/v1", "shared.vlm.server=vllm",
    "--schedule", "stream",
]  # fmt: skip
SEED_FILES = ["data/seeds/seeds.jsonl", "data/seeds/seeds_danbooru.jsonl"]


def arms(poc: Path, per_source: int, image_list: Path) -> dict[str, list[str]]:
    seeds = ["--seeds", *SEED_FILES, "--seed_per_source", str(per_source)]
    leafcap_schema = (poc / "grammars" / "json" / "entity_leafcap.json").as_posix()
    return {
        "text": [*seeds, "--seed_elements", "text"],
        "image_text": [*seeds, "--seed_elements", "image"],
        "tags": [*seeds, "--seed_elements", "tags"],
        "query": ["--image_list", str(image_list)],
        "batch4": [*seeds, "--seed_elements", "text", "--controls", "entity_batch=4"],
        "leafcap": [
            *seeds,
            "--seed_elements", "text",
            "--controls", "leaf_caption=parent",
            "--prompt_dir", "prompts/json_leafcap",
            "--override", f"queries.entity_query.query_kwargs.json_schema_file={leafcap_schema}",
        ],
    }  # fmt: skip


def seed_images(poc: Path, per_source: int) -> list[str]:
    """The images of the seeded arms, in their order (first N of each source)."""
    out, count = [], {}
    for name in SEED_FILES:
        with open(poc / name, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seed = json.loads(line)
                    count[seed["source"]] = count.get(seed["source"], 0) + 1
                    if count[seed["source"]] <= per_source:
                        image = poc / "data" / "seeds" / "images" / seed["image"]
                        out.append(image.as_posix())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poc", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per_source", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--arms", nargs="*", default=None)
    args = parser.parse_args()
    poc, out = Path(args.poc).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    python = poc / ".venv" / "Scripts" / "python.exe"

    image_list = out / "images.txt"
    image_list.write_text("\n".join(seed_images(poc, args.per_source)) + "\n", encoding="utf-8")
    stamp = time.strftime("%m%d%H%M")
    for name, extra in arms(poc, args.per_source, image_list).items():
        if args.arms and name not in args.arms:
            continue
        if (out / name).exists():
            print(f"skip {name}: {out / name} exists", flush=True)
            continue
        argv = [
            str(python),
            str(HERE / "record_poc.py"),
            "--trace", str(out / name), "--poc", str(poc), "--arm", name,
            "--", *BASE, *extra,
            "--workers", str(args.workers), "--run_name", f"trace_{name}_{stamp}",
        ]  # fmt: skip
        start = time.time()
        with open(out / f"{name}.log", "w", encoding="utf-8") as log:
            code = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{name}: exit {code} in {time.time() - start:.0f} s", flush=True)
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
