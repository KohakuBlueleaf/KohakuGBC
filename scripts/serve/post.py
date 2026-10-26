"""Stream images to a ``gbc serve`` service and keep its graphs as they come back.

    python scripts/serve/post.py URL IMAGES OUT.jsonl [--remote]

``IMAGES``: a folder (every picture under it) or a text file of paths. The pictures go as
NDJSON jobs with ``image_b64``; ``--remote``: the paths are on the serving host and go as
``image``. Each answer line (``key``, ``ok``, ``graph``, ``stats``, ``failure``) is appended to
``OUT.jsonl`` as it arrives.
"""

import argparse
import base64
import json
import mimetypes
import sys
import time
from pathlib import Path, PurePosixPath

import httpx

PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")


def paths_of(images: str) -> list[str]:
    """The pictures' paths as written (a remote path keeps the serving host's separators)."""
    p = Path(images)
    if p.is_dir():
        return sorted({str(f) for pat in PATTERNS for f in p.rglob(pat)})
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def jobs(paths: list[str], remote: bool):
    for name in paths:
        key = PurePosixPath(name.replace("\\", "/")).stem
        if remote:
            job = {"key": key, "image": name}
        else:
            path = Path(name)
            job = {
                "key": key,
                "image_b64": base64.b64encode(path.read_bytes()).decode(),
                "content_type": mimetypes.guess_type(path.name)[0] or "",
            }
        yield (json.dumps(job) + "\n").encode()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("url")
    p.add_argument("images")
    p.add_argument("out")
    p.add_argument("--remote", action="store_true", help="paths on the serving host")
    args = p.parse_args()

    paths = paths_of(args.images)
    t0, ok, failed = time.time(), 0, 0
    headers = {"Content-Type": "application/x-ndjson"}
    url = args.url.rstrip("/") + "/jobs"
    with (
        httpx.stream(
            "POST", url, content=jobs(paths, args.remote), headers=headers, timeout=None
        ) as r,
        open(args.out, "a", encoding="utf-8") as f,
    ):
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.strip():
                continue
            answer = json.loads(line)
            f.write(line + "\n")
            f.flush()
            ok += bool(answer.get("ok"))
            failed += not answer.get("ok")
            done = ok + failed
            rate = done / (time.time() - t0)
            print(f"{done}/{len(paths)} ok {ok} failed {failed} {rate:.2f} img/s", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
