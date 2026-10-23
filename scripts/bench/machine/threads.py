"""py-spy dumps of one process, aggregated across its threads: per thread name, the share of
samples it was active and holding the GIL; for the GIL holder, its innermost frames and the
inclusive share of every function on its stack. Dumps are separated by ``=====`` lines.

    python scripts/bench/machine/threads.py DUMPS [N]
"""

import re
import sys
from collections import Counter

HEAD = re.compile(r'^Thread (\S+) \(([^)]*)\): "([^"]*)"', re.M)
FRAME = re.compile(r"^\s*(\S+) \((\S+?):(\d+)\)")


def frames_of(block: str) -> list[str]:
    out = []
    for line in block.splitlines():
        f = FRAME.match(line)
        if f:
            out.append(f"{f.group(1)} ({'/'.join(f.group(2).split('/')[-2:])})")
    return out


def main() -> int:
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
        text = f.read()
    active, gil, top, incl = Counter(), Counter(), Counter(), Counter()
    samples = holders = 0
    for dump in text.split("====="):
        heads = list(HEAD.finditer(dump))
        if not heads:
            continue
        samples += 1
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(dump)
            name = re.sub(r"\d+", "N", h.group(3))
            state = h.group(2)
            if "active" in state:
                active[name] += 1
            if "gil" in state:
                gil[name] += 1
                holders += 1
                frames = frames_of(dump[h.end() : end])
                if frames:
                    top[f"[{name}] {frames[0]}"] += 1
                    for fr in set(frames):
                        incl[f"[{name}] {fr}"] += 1
    print(f"samples {samples}, GIL held in {holders}")
    print("-- threads: active% gil%")
    for name in sorted(set(active) | set(gil), key=lambda k: -active[k]):
        print(f"{100 * active[name] / samples:6.1f} {100 * gil[name] / samples:6.1f}  {name}")
    print("-- GIL holder innermost")
    for fr, c in top.most_common(n // 2):
        print(f"{100 * c / samples:5.1f}%  {fr}")
    print("-- GIL holder inclusive")
    for fr, c in incl.most_common(n):
        print(f"{100 * c / samples:5.1f}%  {fr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
