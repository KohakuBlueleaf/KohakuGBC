"""Check the import boundary between KohakuGBC's layers over every module of src/ (kohakugbc
and the standalone kohakugbc_models) and examples/."""

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

# The program layer: the GBC program, its nodes, the graph; no machine, no engine
PROGRAM = ("kohakugbc.program", "kohakugbc.nodes", "kohakugbc.graph", "kohakugbc._rs", "kohakunode")

# Package prefix -> allowed import prefixes (beyond the standard library)
RULES = {
    "kohakugbc.program": PROGRAM,
    "kohakugbc.nodes": PROGRAM,
    "kohakugbc.graph": PROGRAM,
    "kohakugbc.engines": (
        "kohakugbc.engines",
        "kohakugbc.nodes.specs",
        "kohakugbc_models.detect",
        "kohakugbc_models.embed",
        "togeari",
    ),
    # kohakugbc_models, never importing kohakugbc: the models GBC runs itself (below the
    # engines) above their kernels, the kernels above the numerics
    "kohakugbc_models.detect": (
        "kohakugbc_models.detect",
        "kohakugbc_models.kernels",
        "kohakugbc_models.numerics",
    ),
    "kohakugbc_models.embed": (
        "kohakugbc_models.embed",
        "kohakugbc_models.kernels",
        "kohakugbc_models.numerics",
    ),
    "kohakugbc_models.kernels": ("kohakugbc_models.kernels", "kohakugbc_models.numerics"),
    "kohakugbc_models.numerics": ("kohakugbc_models.numerics",),
    "kohakugbc.machine": (
        *PROGRAM,
        "kohakugbc.machine",
        "kohakugbc.engines",
        "kohakugbc.io",
        "togeari",
    ),
    "kohakugbc.io": (*PROGRAM, "kohakugbc.io", "togeari"),
    "kohakugbc.cli": ("kohakugbc", "togeari", "kohakunode"),
    "qwen35": ("qwen35", "kohakugbc.engines", "kohakugbc.nodes.specs", "togeari"),
}

# Third-party packages each part may use; torch, transformers and vllm only in the engines and
# the models below them
EXTRAS = {
    "kohakugbc.program": ("PIL", "numpy"),
    "kohakugbc.nodes": ("PIL", "numpy", "scipy", "hbutils", "supervision"),
    "kohakugbc.graph": ("PIL", "numpy", "pyarrow"),
    "kohakugbc.engines": (
        "PIL",
        "numpy",
        "httpx",
        "torch",
        "transformers",
        "sentence_transformers",
        "inflect",
        "vllm",
    ),
    "kohakugbc.io": ("PIL", "numpy", "pyarrow"),
    "kohakugbc_models.detect": (
        "PIL",
        "numpy",
        "torch",
        "transformers",
        "huggingface_hub",
        "safetensors",
    ),
    "kohakugbc_models.embed": ("numpy", "torch", "transformers", "huggingface_hub", "safetensors"),
    "kohakugbc_models.kernels": ("torch", "triton", "tilelang"),
    "kohakugbc_models.numerics": ("torch",),
    "qwen35": (
        "PIL",
        "numpy",
        "torch",
        "transformers",
        "huggingface_hub",
        "triton",
        "fla",
        "xgrammar",
        "tilelang",
        "safetensors",
    ),
}

# Source roots scanned; a module's name is its path under its root
ROOTS = [SRC, REPO / "examples" / "qwen35" / "src"]


def module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imports(tree: ast.AST, mod: str, is_pkg: bool) -> list[str]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # A package's own __init__ is level 1 of itself
                level = node.level - 1 if is_pkg else node.level
                base = mod.rsplit(".", level)[0] if level else mod
                source = f"{base}.{node.module}" if node.module else base
            else:
                source = node.module

            # Each imported name, as it may be a submodule (``from kohakugbc import _rs``)
            out += [f"{source}.{a.name}" for a in node.names]
    return out


def violations(roots: list[Path]) -> list[str]:
    """Every import under the roots that its layer may not make, as ``path: imports name``."""
    stdlib = set(sys.stdlib_module_names)
    bad = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            mod = module_name(root, path)
            rule = next((k for k in RULES if mod == k or mod.startswith(k + ".")), None)
            if rule is None:
                continue
            allowed = RULES[rule] + EXTRAS.get(rule, ())
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for name in imports(tree, mod, path.name == "__init__.py"):
                top = name.split(".")[0]
                if top in stdlib or top == "__future__":
                    continue
                if not any(name == a or name.startswith(a + ".") for a in allowed):
                    bad.append(f"{path.relative_to(root.parent)}: imports {name}")
    return bad


def main() -> int:
    bad = violations(ROOTS)
    for b in bad:
        print(b)
    print(f"{len(bad)} import boundary violation(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
