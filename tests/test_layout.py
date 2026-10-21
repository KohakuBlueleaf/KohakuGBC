"""The package layers import, and the import boundary holds and can fail."""

import importlib
import sys
from pathlib import Path

import pytest

import kohakugbc
from kohakugbc import _rs

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import check_imports  # noqa: E402

PACKAGES = [
    "kohakugbc",
    "kohakugbc._rs",
    "kohakugbc.program",
    "kohakugbc.nodes",
    "kohakugbc.graph",
    "kohakugbc.machine",
    "kohakugbc.engines",
    "kohakugbc.io",
    "kohakugbc.cli.main",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_imports(name):
    importlib.import_module(name)


def test_native_version_is_package_version():
    assert kohakugbc.__version__.split("+")[0].startswith(_rs.version())


def test_import_boundary_holds():
    assert check_imports.violations(check_imports.ROOTS) == []


def test_import_boundary_catches_each_layer(tmp_path):
    # One forbidden import per layer, absolute and relative, in modules and package inits
    files = {
        "kohakugbc/graph/__init__.py": "from ..machine import build\n",
        "kohakugbc/graph/resolve.py": "import torch\n",
        "kohakugbc/nodes/regions/crop.py": "from togeari import Use\n",
        "kohakugbc/engines/vlm/api.py": "from kohakugbc.machine import bindings\n",
        "kohakugbc/engines/vlm/ok.py": "from kohakugbc.nodes.specs import chat\nimport httpx\n",
        "kohakugbc/program/__init__.py": "from . import controls\n",
    }
    for rel, text in files.items():
        path = tmp_path / "src" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    bad = check_imports.violations([tmp_path / "src"])
    names = sorted(b.split(": imports ")[1] for b in bad)
    assert names == [
        "kohakugbc.machine.bindings",
        "kohakugbc.machine.build",
        "togeari.Use",
        "torch",
    ]
