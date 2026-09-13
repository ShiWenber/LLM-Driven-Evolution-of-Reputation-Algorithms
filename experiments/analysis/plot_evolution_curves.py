"""Plot cooperation trajectories for one evolution run label.

Draws a single panel: the per-seed cooperation curves of
``<results-dir>/<label>_seed<N>/evolutionary.json`` for the requested seeds,
plus their mean and a one-standard-deviation band.
Any run label can be passed, so the same figure works for either agent type.
The module exposes a reusable function, module CLI, and ``uv`` console entry.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.evolution_log import (
    F_COOPERATION_RATE_MEAN,
    F_GENERATION,
    K_TRAJECTORY,
    load_evolution_json,
)

from .paths import evolution_json_path as canonical_json_path


DEFAULT_LABEL = "LLM_agent-type1_fermi_z_v3_g100_10000inter_N16_genreset_upd4_5seed"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_COLOR = "#2166ac"


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load generation and cooperation arrays from an evolution JSON."""
    if not path.exists():
        return None
    data: dict[str, Any] = load_evolution_json(path)
    trajectory = data.get(K_TRAJECTORY, [])
    if not trajectory:
        raise ValueError(f"Trajectory is empty: {path}")
    return (
        np.asarray([row[F_GENERATION] for row in trajectory]),
        np.asarray([row[F_COOPERATION_RATE_MEAN] for row in trajectory], dtype=float),
    )


def load_seed_runs(
    results_dir: Path,
    label: str,
    seeds: Sequence[int],
    *,
    require_all: bool = True,
) -> list[tuple[int, tuple[np.ndarray, np.ndarray]]]:
    """Load ``<label>_seedN/evolutionary.json`` runs in canonical layout."""
    runs = []
    missing = []
    for seed in seeds:
        path = canonical_json_path(results_dir, label, seed)
        trajectory = load_trajectory(path)
        if trajectory is None:
            print(f"  seed {seed}: MISSING ({path})")
            missing.append(path)
            continue
        print(f"  seed {seed}: OK")
        runs.append((seed, trajectory))
    if require_all and missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required runs:\n{formatted}")
    if not runs:
        raise FileNotFoundError(f"No trajectories found for {label}")
    return runs


def plot_evolution_curves(
    label: str = DEFAULT_LABEL,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    results_dir: Path | None = None,
    output_path: Path | None = None,
    title: str | None = None,
    color: str = DEFAULT_COLOR,
    dpi: int = 180,
    write_pdf: bool = True,
) -> Path:
    """Draw per-seed cooperation curves, their mean, and a ±1 std band."""
    from .paths import quantitative_results_dir

    results_dir = Path(results_dir) if results_dir is not None else quantitative_results_dir()
    output_path = (
        Path(output_path)
        if output_path is not None
        else results_dir / "plots" / f"{label}_evolution_curves.png"
    )

    print(f"{label}:")
    runs = load_seed_runs(results_dir, label, seeds)

    generations = runs[0][1][0]
    for seed, (seed_generations, _) in runs:
        if not np.array_equal(seed_generations, generations):
            raise ValueError(f"Generation indices differ for seed {seed}")

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for seed, (_, cooperation) in runs:
        ax.plot(
            generations,
            cooperation,
            linewidth=1.0,
            alpha=0.42,
            color=color,
            label=f"seed {seed}",
        )

    cooperation = np.vstack([trajectory[1] for _, trajectory in runs])
    mean = cooperation.mean(axis=0)
    std = cooperation.std(axis=0)
    ax.fill_between(
        generations,
        np.clip(mean - std, 0, 1),
        np.clip(mean + std, 0, 1),
        color=color,
        alpha=0.16,
        label="mean ± std",
    )
    ax.plot(generations, mean, linewidth=2.6, color=color, label=f"{len(runs)}-seed mean")

    ax.set_title(title or f"{label}\n{len(runs)} seeds", fontsize=13, fontweight="semibold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cooperation rate")
    ax.set_xlim(generations.min(), generations.max())
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    if write_pdf:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output_path}")
    if write_pdf:
        print(f"saved: {pdf_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--label", default=DEFAULT_LABEL, help="Run label to plot")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--title", default=None, help="Override the figure title")
    parser.add_argument("--color", default=DEFAULT_COLOR)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-pdf", action="store_true", help="Do not write a PDF copy")
    args = parser.parse_args()
    plot_evolution_curves(
        label=args.label,
        seeds=args.seeds,
        results_dir=args.results_dir,
        output_path=args.output,
        title=args.title,
        color=args.color,
        dpi=args.dpi,
        write_pdf=not args.no_pdf,
    )


if __name__ == "__main__":
    main()
