"""`vllm serve` with the engine arguments of a machine configuration (its ``_VLLM`` dict), so the
HTTP path runs the same engine as the in-process one. Runs in the foreground; the caller picks
the card with CUDA_VISIBLE_DEVICES and stops the server.

    python scripts/bench/machine/serve.py CONFIG PORT
"""

import json
import os
import runpy
import sys


def flags(args: dict) -> list[str]:
    """vLLM CLI flags of engine arguments: dicts as JSON, booleans as switches."""
    out = []
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        match value:
            case True:
                out.append(flag)
            case False:
                if key.startswith("enable_"):
                    out.append("--no-" + flag[2:])
            case dict():
                out += [flag, json.dumps(value)]
            case _:
                out += [flag, str(value)]
    return out


def main() -> int:
    config, port = sys.argv[1], sys.argv[2]
    args = dict(runpy.run_path(config)["_VLLM"])
    model = args.pop("model")
    argv = ["vllm", "serve", model, "--host", "127.0.0.1", "--port", port, *flags(args)]
    print(" ".join(argv), flush=True)
    os.execvp(argv[0], argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
