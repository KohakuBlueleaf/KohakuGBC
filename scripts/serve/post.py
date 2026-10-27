"""Stream images to a ``gbc serve`` service and keep its graphs as they come back.

    python scripts/serve/post.py URL IMAGES OUT.jsonl [--remote] [--max-side N]

``IMAGES``: a folder (every picture under it) or a text file of paths. The pictures go as
NDJSON jobs with ``image_b64``; ``--remote``: the paths are on the serving host and go as
``image``; ``--max-side N``: each picture is read and shrunk here (long side at most N) and
sent as ``image_b64``, so the service never decodes a full-size original. Each answer line
(``key``, ``ok``, ``graph``, ``stats``, ``failure``) is appended to ``OUT.jsonl`` as it arrives.
"""

import argparse
import base64
import io
import json
import mimetypes
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import httpx
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")


def paths_of(images: str) -> list[str]:
    """The pictures' paths as written (a remote path keeps the serving host's separators)."""
    p = Path(images)
    if p.is_dir():
        return sorted({str(f) for pat in PATTERNS for f in p.rglob(pat)})
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def shrunk(path: Path, side: int) -> tuple[bytes, str]:
    """The picture at most ``side`` px on the long side (Lanczos), alpha on white, WEBP q95."""
    with Image.open(path) as img:
        img.draft("RGB", (side, side))
        if img.mode in ("RGBA", "LA", "P", "PA"):
            rgba = img.convert("RGBA")
            img = Image.new("RGB", rgba.size, "white")
            img.paste(rgba, mask=rgba.getchannel("A"))
        else:
            img = img.convert("RGB")
        if max(img.size) > side:
            scale = side / max(img.size)
            size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
            img = img.resize(size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "WEBP", quality=95, method=4)
    return buf.getvalue(), "image/webp"


def job_of(name: str, remote: bool, max_side: int) -> bytes:
    key = PurePosixPath(name.replace("\\", "/")).stem
    if max_side:
        data, ctype = shrunk(Path(name), max_side)
    elif remote:
        return (json.dumps({"key": key, "image": name}) + "\n").encode()
    else:
        data, ctype = Path(name).read_bytes(), mimetypes.guess_type(name)[0] or ""
    job = {"key": key, "image_b64": base64.b64encode(data).decode(), "content_type": ctype}
    return (json.dumps(job) + "\n").encode()


def jobs(paths: list[str], remote: bool, max_side: int, workers: int):
    """Jobs in path order, ``workers`` threads preparing up to ``4 * workers`` ahead."""
    with ThreadPoolExecutor(workers) as pool:
        ahead: deque = deque()
        for name in paths:
            ahead.append(pool.submit(job_of, name, remote, max_side))
            if len(ahead) >= 4 * workers:
                yield ahead.popleft().result()
        while ahead:
            yield ahead.popleft().result()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("url")
    p.add_argument("images")
    p.add_argument("out")
    p.add_argument("--remote", action="store_true", help="paths on the serving host")
    p.add_argument("--max-side", type=int, default=0, help="shrink here first, send as b64")
    p.add_argument("--workers", type=int, default=8, help="threads reading/shrinking images")
    args = p.parse_args()

    paths = paths_of(args.images)
    t0, ok, failed = time.time(), 0, 0
    headers = {"Content-Type": "application/x-ndjson"}
    url = args.url.rstrip("/") + "/jobs"
    with (
        httpx.stream(
            "POST",
            url,
            content=jobs(paths, args.remote, args.max_side, args.workers),
            headers=headers,
            timeout=None,
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
