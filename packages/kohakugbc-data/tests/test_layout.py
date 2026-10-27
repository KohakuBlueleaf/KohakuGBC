"""The package imports without the captioning machine, and its native part is its version."""

import importlib
import sys

import pytest

import kohakugbc_data
from kohakugbc_data import _rs


@pytest.mark.parametrize(
    "name",
    [
        "kohakugbc_data",
        "kohakugbc_data.prompt",
        "kohakugbc_data.prompt.detail",
        "kohakugbc_data.dataset",
    ],
)
def test_imports(name):
    importlib.import_module(name)


def test_no_machine_imported():
    for name in ("kohakugbc", "togeari", "kohakunode", "torch", "cv2", "scipy"):
        assert name not in sys.modules, name


def test_native_version_is_package_version():
    assert kohakugbc_data.__version__.split("+")[0].startswith(_rs.version().split("-")[0])
