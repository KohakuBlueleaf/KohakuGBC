"""Record the detector requests a GBC machine sends: the machine of the configuration layers over
a folder of images, every request the detect lane's model prepares once the machine runs (the
load-time example is not one) saved in arrival order.

Writes ``OUT/requests.jsonl`` (one line per request: ``name``, ``texts``, ``patch`` file, the
batch it came in and that batch's size, its seconds since the first) and ``OUT/patches/<n>.npy``,
up to ``--limit`` requests. The patches are written by a thread of their own, so the prep
threads that hand them over only queue them. The run goes on to its end. Records only an
in-process native SAM 3 detector; a run that records nothing fails.

    GBC_IMAGES=DIR python scripts/bench/detector/record.py OUT --limit 2000 --images 240 \
        configs/machine/qwen35_card.py configs/controls/mid.py configs/controls/sam3.py
"""

import argparse
import itertools
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from kohakugbc.engines.torch.modules import NativeSam3Detector
from kohakugbc.machine import build_parts


class Recorder:
    def __init__(self, out: Path, limit: int):
        self.out, self.limit = out, limit
        (out / "patches").mkdir(parents=True, exist_ok=True)
        self.lines = open(out / "requests.jsonl", "w", encoding="utf-8")  # noqa: SIM115
        self.lock = threading.Lock()
        self.active = False
        self.n = self.batches = 0
        self.t0 = None
        self.todo: queue.Queue = queue.Queue()
        self.writer = threading.Thread(target=self.write, daemon=True)
        self.writer.start()

    def take(self, payloads: list) -> None:
        if not self.active:
            return
        with self.lock:
            now = time.monotonic()
            self.t0 = now if self.t0 is None else self.t0
            batch = self.batches
            self.batches += 1
            for p in payloads:
                if self.n >= self.limit:
                    return
                line = {
                    "name": self.n,
                    "texts": list(p["request"]["texts"]),
                    "patch": f"{self.n:06d}.npy",
                    "batch": batch,
                    "batch_size": len(payloads),
                    "t": now - self.t0,
                }
                self.todo.put((line, p["patch"]))
                self.n += 1

    def write(self) -> None:
        while (item := self.todo.get()) is not None:
            line, patch = item
            np.save(self.out / "patches" / line["patch"], patch)
            self.lines.write(json.dumps(line) + "\n")

    def close(self) -> None:
        self.todo.put(None)
        self.writer.join()
        self.lines.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("layers", nargs="+")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--images", type=int, default=0, help="the source's first N (0: all)")
    args = parser.parse_args()
    out = Path(args.out)
    rec = Recorder(out, args.limit)

    # Every batch the detector prepares, as the engine hands it over
    prepare = NativeSam3Detector.prepare

    def recorded(self, payloads):
        # The engine's load-time warm-up passes the model's own example, never a request
        example = self.example()
        if not (len(payloads) == 1 and payloads[0]["request"] == example[0]["request"]):
            rec.take(payloads)
        return prepare(self, payloads)

    NativeSam3Detector.prepare = recorded

    parts = build_parts(*args.layers, {"sink": {"kind": "memory"}})
    source = parts.source
    if args.images:
        source = itertools.islice(source, args.images)
    with parts.machine:
        rec.active = True
        t0 = time.time()
        parts.machine.run(source, parts.sink)
        run_s = time.time() - t0
    rec.close()
    summary = {
        "jobs": len(parts.sink.records),
        "ok": sum(r.ok for r in parts.sink.records),
        "run_s": run_s,
        "requests": rec.n,
        "batches": rec.batches,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))
    if not rec.n:
        print("no request recorded (not an in-process native SAM 3 detector?)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
