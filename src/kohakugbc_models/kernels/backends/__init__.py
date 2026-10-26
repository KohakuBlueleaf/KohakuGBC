"""Kernel backends. Each backend package imports every public module of its tree on import
(``load_tree``); a module whose import raises (a missing library, or a toolchain whose API
differs, e.g. an older Gluon) is listed in the package's ``FAILED`` and the rest still
register."""

import importlib
import logging
import pkgutil

log = logging.getLogger(__name__)


def load_tree(package: str, path) -> dict[str, str]:
    """Import every module under ``package`` not starting with ``_``; the failed ones."""
    failed = {}
    for info in pkgutil.walk_packages(path, prefix=f"{package}."):
        if info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        try:
            importlib.import_module(info.name)
        except Exception as err:  # noqa: BLE001 - one module never takes the others down
            failed[info.name] = f"{type(err).__name__}: {err}"
            log.warning("kernel module %s not loaded: %s", info.name, err)
    return failed
