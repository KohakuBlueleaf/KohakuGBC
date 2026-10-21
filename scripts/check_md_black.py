"""Check that every ```python block in the markdown files of a tree passes black
(line length 100).
"""

import difflib
import re
import sys
from pathlib import Path

import black

BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
MODE = black.Mode(line_length=100)
SKIP = {".venv", "node_modules", "target", "plans", "runs"}


def main(root: str) -> int:
    bad = 0
    for path in sorted(Path(root).rglob("*.md")):
        if SKIP & set(path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        for m in BLOCK.finditer(text):
            src = m.group(1)
            line = text[: m.start()].count("\n") + 1
            try:
                out = black.format_str(src, mode=MODE)
            except Exception as exc:  # noqa: BLE001
                bad += 1
                print(f"{path}:{line}: black cannot parse: {exc}")
                continue
            if out != src:
                bad += 1
                print(f"{path}:{line}: not black-formatted")
                sys.stdout.writelines(
                    difflib.unified_diff(
                        src.splitlines(True), out.splitlines(True), "written", "black"
                    )
                )
    print(f"{bad} block(s) need work")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
