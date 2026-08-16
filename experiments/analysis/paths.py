"""Path resolution helpers for analysis commands.

Library functions accept explicit paths whenever possible.  The helpers here
only provide convenient defaults for CLI usage; they do not store a global
project-root path or depend on a machine-specific absolute directory.
"""

from __future__ import annotations

from pathlib import Path

from experiments.evolution_log import (
    evolution_json_path as _evolution_json_path,
    run_dir as _run_dir,
)


def project_root(start: Path | None = None) -> Path:
    """Find the repository root from ``start`` or the current directory."""
    # Prefer the caller's working tree, but fall back to this package's
    # location when a launcher does not preserve the requested cwd.
    current = (start or Path.cwd()).resolve()
    if not (current / "pyproject.toml").is_file():
        current = Path(__file__).resolve().parent
    candidates = (current, *current.parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "experiments").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the project root; run from the repository or pass an explicit path."
    )


def quantitative_results_dir(base: Path | None = None) -> Path:
    """Return the quantitative-baseline results directory."""
    return (base or project_root()) / "results" / "quantitative_baseline"


def invasion_results_dir(base: Path | None = None) -> Path:
    """Return the default Agent 2 invasion result directory."""
    return quantitative_results_dir(base) / "invasion" / "agent2_schmid_L1_L2_L7_L8_mainmatched"


def run_dir(output_root, label: str, seed: int) -> Path:
    """Canonical per-seed run directory ``<output_root>/<label>_seed<N>``.

    Delegates to ``experiments.evolution_log`` (the single source of
    truth for the on-disk layout shared with the v2/v3 writers).
    """
    return _run_dir(output_root, label, seed)


def evolution_json_path(output_root, label: str, seed: int) -> Path:
    """Canonical result file for one seed run (schema contract)."""
    return _evolution_json_path(output_root, label, seed)
