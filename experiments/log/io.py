"""Canonical run layout plus atomic read/write of evolution-log records."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from .schema import RESULTS_FILENAME
from .validate import validate_evolution_results


def run_dir(output_root, label: str, seed: int) -> Path:
    """Canonical per-seed run directory: ``<output_root>/<label>_seed<N>``."""
    return Path(output_root) / f"{label}_seed{seed}"


def evolution_json_path(output_root, label: str, seed: int) -> Path:
    """Canonical result file for one seed run."""
    return run_dir(output_root, label, seed) / RESULTS_FILENAME


def write_evolution_json(
    path, data: Dict[str, Any], *, validate: bool = True
) -> Path:
    """Atomically persist a top-level record (tmp file + rename).

    Creates parent directories as needed. Returns the written path.
    """
    if validate:
        validate_evolution_results(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return path


def load_evolution_json(path, *, validate: bool = True) -> Dict[str, Any]:
    """Load a top-level record; optionally validate it against the contract."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if validate:
        validate_evolution_results(data)
    return data
