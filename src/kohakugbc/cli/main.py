"""``gbc``: the command line of KohakuGBC."""

import argparse
import sys

from kohakugbc import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gbc", description=__doc__)
    parser.add_argument("--version", action="version", version=f"kohakugbc {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
