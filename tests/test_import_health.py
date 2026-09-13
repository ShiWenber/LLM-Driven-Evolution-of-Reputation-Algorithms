"""Guards against dangling import/reference breakage after refactors.

Two real incidents motivated these checks:

1. Moving a private helper between modules broke a caller that imported the
   private name. No test imported that module, so the breakage shipped silently.
2. A module was later deleted outright, which left its script target in
   ``pyproject.toml`` pointing at a module that no longer exists. Module-import
   checks cannot see that, because a script target is a string, not an import.

So: import every module, and resolve every declared script target.
"""

from __future__ import annotations

import importlib
import pkgutil
import tomllib
from pathlib import Path

import pytest

import experiments

PACKAGE_ROOT = Path(experiments.__file__).parent
PREFIX = "experiments."
PROJECT_ROOT = PACKAGE_ROOT.parent


def _source_backed_module_names() -> list[str]:
    """Modules under ``experiments/`` that have a real ``.py`` source.

    ``pkgutil`` also lists orphaned ``__pycache__/*.pyc`` left behind by deleted
    modules; those are build artifacts, and failing on them is a false positive.
    """
    names = []
    for info in pkgutil.walk_packages([str(PACKAGE_ROOT)], prefix=PREFIX):
        relative = info.name[len(PREFIX):].replace(".", "/")
        if (PACKAGE_ROOT / f"{relative}.py").is_file():
            names.append(info.name)
    return sorted(names)


def _script_targets() -> dict[str, str]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    return dict(data.get("project", {}).get("scripts", {}))


@pytest.mark.parametrize("name", _source_backed_module_names())
def test_module_imports_cleanly(name: str):
    assert importlib.import_module(name) is not None


@pytest.mark.parametrize("name,target", sorted(_script_targets().items()))
def test_script_target_resolves(name: str, target: str):
    """``module:attr`` in [project.scripts] must point at a real callable."""
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attr)), f"{name} -> {target} is not callable"


def test_guards_are_not_vacuous():
    """Parametrisation must actually find things, not silently find none."""
    assert len(_source_backed_module_names()) > 20
    assert len(_script_targets()) > 5
