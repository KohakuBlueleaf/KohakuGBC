"""Release notes for a version of kohakugbc-data: its Change.md section, or the commit list.

Change.md (in the package) is the curated source. When it has no section for the version
being released (a patch or nightly cut by automation), the notes fall back to the commits that
touched the package since its previous stable tag, so a release never has empty notes.

Usage:
    python scripts/ci/release_notes.py --version 0.1.0
"""

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "packages/kohakugbc-data"
PREFIX = "data-v"
CHANGELOG = ROOT / PACKAGE / "Change.md"


def changelog_section(version: str) -> str | None:
    if not CHANGELOG.exists():
        return None
    text = CHANGELOG.read_text(encoding="utf-8")
    # "## <date> update to <version>" through the next "## " heading.
    pattern = re.compile(rf"(?ms)^##\s+[^\n]*update to {re.escape(version)}\s*$(.*?)(?=^##\s|\Z)")
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False, cwd=ROOT
    ).stdout.strip()


def commits_since_last_tag(version: str) -> str:
    tags = [
        t
        for t in git("tag", "--list", f"{PREFIX}*", "--sort=-creatordate").split()
        if "dev" not in t and t != f"{PREFIX}{version}"
    ]
    span = f"{tags[0]}..HEAD" if tags else "HEAD"
    log = git("log", "--no-merges", "--pretty=* %s (%h)", span, "--", PACKAGE)
    return log or "* No changes recorded."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    args = ap.parse_args()
    body = changelog_section(args.version)
    if body is None:
        body = f"### Changes\n\n{commits_since_last_tag(args.version)}"
    print(f"## kohakugbc-data {args.version}\n\n{body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
