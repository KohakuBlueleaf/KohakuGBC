"""Reply validity of replayed GBC requests (``replay.py`` rows), with or without a JSON schema in
the request: per action, strict ``json.loads`` of the whole reply, GBC's tolerant reader
(``parse.load_json``), the request's schema checked on what the tolerant reader returns, and
replies cut at max_tokens. Replay row i is trace row i % len(trace).

    python scripts/bench/vlm/parse_rate.py TRACE_DIR/vlm.jsonl REPLAY.jsonl [REPLAY.jsonl ...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from kohakugbc.nodes.specs import chat
from kohakugbc.nodes.specs.parse import load_json

TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int}


def valid(x, s: dict) -> bool:
    if "enum" in s and x not in s["enum"]:
        return False
    t = s.get("type")
    if t and not isinstance(x, TYPES.get(t, object)):
        return False
    if t == "object":
        if any(k not in x for k in s.get("required", [])):
            return False
        return all(valid(x[k], v) for k, v in s.get("properties", {}).items() if k in x)
    if t == "array":
        if len(x) > s.get("maxItems", 1 << 30) or len(x) < s.get("minItems", 0):
            return False
        return all(valid(i, s["items"]) for i in x) if "items" in s else True
    if t == "string":
        return s.get("minLength", 0) <= len(x) <= s.get("maxLength", 1 << 30)
    return True


def strict(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> int:
    trace = read_jsonl(sys.argv[1])
    for path in sys.argv[2:]:
        stats = defaultdict(lambda: [0, 0, 0, 0, 0])
        for i, r in enumerate(read_jsonl(path)):
            req = trace[i % len(trace)]["request"]
            schema = chat.schema(req["schema"]) if req["schema"] else None
            obj = load_json(r["text"])
            s = stats[r["action"].split(":")[0]]
            s[0] += 1
            s[1] += strict(r["text"])
            s[2] += obj is not None
            s[3] += obj is not None and (schema is None or valid(obj, schema))
            s[4] += r.get("finish") == "length"
        print(f"== {Path(path).name}")
        head = ("n", 5), ("strict", 7), ("tolerant", 8), ("schema ok", 9), ("cut", 5)
        print(f"  {'action':28s} " + " ".join(f"{h:>{w}s}" for h, w in head))
        tot = [0] * 5
        for action, s in sorted(stats.items()):
            tot = [x + y for x, y in zip(tot, s)]
            print(f"  {action:28s} {s[0]:5d} {s[1]:7d} {s[2]:8d} {s[3]:9d} {s[4]:5d}")
        n = tot[0]
        print(
            f"  {'all':28s} {n:5d} {tot[1]:7d} {tot[2]:8d} {tot[3]:9d} {tot[4]:5d}"
            f"   -> needs a retry: {100 * (n - tot[3]) / n:.1f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
