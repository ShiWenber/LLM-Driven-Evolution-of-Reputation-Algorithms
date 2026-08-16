"""Plot cooperation evolution curves for quantitative-baseline runs.

This is the packaged version of the former repository-root plotting script.
It uses paths relative to the project root and exposes a reusable function,
module CLI, and ``uv`` console entry point.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.evolution_log import (
    F_COOPERATION_RATE_MEAN, F_FITNESS_MEAN, F_GENERATION, K_TRAJECTORY,
    RESULTS_FILENAME, load_evolution_json,
)

from .paths import evolution_json_path as canonical_json_path

DEFAULT_BASELINE_LABEL = "LLM_v3_g100_thinking_off_seed0"
DEFAULT_PRODUCTION_LABEL = "LLM_v3_fermi_z_v3_g100_1000inter"


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load generation, cooperation, and fitness arrays from an evolution JSON."""
    if not path.exists():
        return None
    data: dict[str, Any] = load_evolution_json(path)
    trajectory = data.get(K_TRAJECTORY, [])
    if not trajectory:
        raise ValueError(f"Trajectory is empty: {path}")
    return (
        np.asarray([row[F_GENERATION] for row in trajectory]),
        np.asarray([row[F_COOPERATION_RATE_MEAN] for row in trajectory]),
        np.asarray([row[F_FITNESS_MEAN] for row in trajectory]),
    )


def load_seed_runs(
    results_dir: Path, label: str, seeds: list[int] | tuple[int, ...]
) -> list[tuple[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Load available ``<label>_seedN/evolutionary.json`` runs (canonical layout)."""
    runs = []
    for seed in seeds:
        path = canonical_json_path(results_dir, label, seed)
        trajectory = load_trajectory(path)
        if trajectory is None:
            print(f"  production seed {seed}: MISSING ({path})")
            continue
        print(f"  production seed {seed}: OK")
        runs.append((seed, trajectory))
    return runs


def plot_evolution_curves(
    results_dir: Path | None = None,
    output_path: Path | None = None,
    baseline_label: str = DEFAULT_BASELINE_LABEL,
    production_label: str = DEFAULT_PRODUCTION_LABEL,
    seeds: list[int] | tuple[int, ...] = (0, 1, 2),
    dpi: int = 180,
) -> Path:
    """Create and save the baseline-versus-production evolution figure."""
    from .paths import quantitative_results_dir
    results_dir = Path(results_dir) if results_dir is not None else quantitative_results_dir()
    output_path = (Path(output_path) if output_path is not None
                   else results_dir / "plots" / "evolution_curves.png")
    # NOTE: `baseline_label` is the FULL seed-directory name (the seed suffix
    # is baked in, e.g. LLM_v3_g100_thinking_off_seed0), so it is not passed
    # through evolution_json_path (which appends `_seed{N}` itself).
    baseline_path = results_dir / baseline_label / RESULTS_FILENAME
    baseline = load_trajectory(baseline_path)
    print(f"baseline: {'OK' if baseline is not None else 'MISSING'} ({baseline_path})")
    production = load_seed_runs(results_dir, production_label, seeds)
    if not production:
        raise FileNotFoundError(f"No production trajectories found for {production_label}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    if baseline is not None:
        generations, cooperation, _ = baseline
        ax.plot(generations, cooperation, color="#1f77b4", linewidth=1.5,
                label=f"{baseline_label} (seed 0)")
        for generation in (15, 30, 60, 95):
            ax.axvline(generation, color="gray", linestyle=":", alpha=0.4)
    ax.set_title("Baseline evolution", fontsize=12)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cooperation rate")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    minimum = min(len(trajectory[0]) for _, trajectory in production)
    generations = production[0][1][0][:minimum]
    cooperation = np.asarray([trajectory[1][:minimum] for _, trajectory in production])
    mean = cooperation.mean(axis=0)
    std = cooperation.std(axis=0)
    ax.plot(generations, mean, color="#d62728", linewidth=1.5,
            label=f"mean (n={len(production)} seeds)")
    if len(production) > 1:
        ax.fill_between(generations, mean - std, mean + std,
                        color="#d62728", alpha=0.2, label="± std")
    for index, (seed, trajectory) in enumerate(production):
        ax.plot(trajectory[0], trajectory[1], color=plt.cm.Reds(0.4 + 0.2 * index),
                linewidth=0.7, alpha=0.5, label=f"seed {seed}")
    if baseline is not None:
        ax.axhline(baseline[1][-1], color="#1f77b4", linestyle="--", alpha=0.5,
                   label=f"baseline final ({baseline[1][-1]:.3f})")
    ax.set_title("Production evolution", fontsize=12)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cooperation rate")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Quantitative-baseline evolution curves", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"saved: {output_path}")
    print(f"saved: {pdf_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL)
    parser.add_argument("--production-label", default=DEFAULT_PRODUCTION_LABEL)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    plot_evolution_curves(
        results_dir=args.results_dir,
        output_path=args.output,
        baseline_label=args.baseline_label,
        production_label=args.production_label,
        seeds=args.seeds,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
