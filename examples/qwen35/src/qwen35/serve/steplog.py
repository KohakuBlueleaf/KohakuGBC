"""One JSON line per engine step and vision call, when the unit is given a ``step_log`` path:
what each step held, what it cost on the host and on the device, and the pools after it."""

import json
import threading
import time


class StepLog:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")  # noqa: SIM115  (open for the unit's life)
        self.lock = threading.Lock()
        self.t0 = time.perf_counter()

    def write(self, row: dict) -> None:
        row = {"t": round(time.perf_counter() - self.t0, 4), **row}
        with self.lock:
            self.f.write(json.dumps(row) + "\n")

    def close(self) -> None:
        with self.lock:
            self.f.close()
