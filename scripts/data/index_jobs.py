"""Jobs for ``gbc run`` (the ``jsonl`` source) from image index files: each line
``set<TAB>key<TAB>path``, each job ``{"key": "<set>/<key>", "image": path}``; the job key is the
graph's key in the vault (``kohakugbc.io.vault``), so a graph finds its image and back through
the index.

    python scripts/data/index_jobs.py OUT.jsonl INDEX.txt [INDEX.txt ...]
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("indexes", nargs="+")
    a = p.parse_args()
    seen: set[str] = set()
    n = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for index in a.indexes:
            for line in Path(index).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                group, key, path = line.split("\t")
                job_key = f"{group}/{key}"
                if job_key in seen:
                    raise ValueError(f"{index}: key {job_key!r} given twice")
                seen.add(job_key)
                out.write(json.dumps({"key": job_key, "image": path}) + "\n")
                n += 1
    print(n, "jobs")


if __name__ == "__main__":
    main()
