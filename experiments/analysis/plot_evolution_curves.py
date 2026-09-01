"""Plot matched cooperation trajectories for agent-type1 and agent-type2.

The default figure compares six current agent-type1 seeds with the three
available agent-type2 seeds under the 100-generation, 1,000-interaction,
population-16 generation-reset experiment used in README.
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
    F_FITNESS_MEAN,
    F_GENERATION,
    K_TRAJECTORY,
    load_evolution_json,
)

from .paths import evolution_json_path as canonical_json_path


DEFAULT_AGENT_TYPE1_LABEL = "LLM_agent-type1_fermi_z_v3_g100_1000inter_N16_genreset"
DEFAULT_AGENT_TYPE2_LABEL = "LLM_v3_fermi_z_v3_g100_1000inter_N16_genreset"

AGENT_STYLES = {
    "agent-type1": ("#2166ac", "two-function strategy"),
    "agent-type2": ("#b2182b", "stateful LLMAgent class"),
}


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
        np.asarray([row[F_COOPERATION_RATE_MEAN] for row in trajectory], dtype=float),
        np.asarray([row[F_FITNESS_MEAN] for row in trajectory], dtype=float),
    )


def load_seed_runs(
    results_dir: Path,
    label: str,
    seeds: Sequence[int],
    *,
    require_all: bool = True,
) -> list[tuple[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
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
        raise FileNotFoundError(f"Missing required comparison runs:\n{formatted}")
    if not runs:
        raise FileNotFoundError(f"No trajectories found for {label}")
    return runs


def _plot_agent_panel(
    ax: plt.Axes,
    agent_type: str,
    runs: list[tuple[int, tuple[np.ndarray, np.ndarray, np.ndarray]]],
) -> None:
    """Draw individual seeds, their mean, and one-standard-deviation band."""
    color, description = AGENT_STYLES[agent_type]
    lengths = {len(trajectory[0]) for _, trajectory in runs}
    if len(lengths) != 1:
        raise ValueError(f"Mismatched trajectory lengths for {agent_type}: {lengths}")

    reference_generations = runs[0][1][0]
    for seed, trajectory in runs:
        generations, cooperation, _ = trajectory
        if not np.array_equal(generations, reference_generations):
            raise ValueError(f"Generation indices differ for {agent_type}, seed {seed}")
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
        reference_generations,
        np.clip(mean - std, 0, 1),
        np.clip(mean + std, 0, 1),
        color=color,
        alpha=0.16,
        label="mean ± std",
    )
    ax.plot(
        reference_generations,
        mean,
        linewidth=2.6,
        color=color,
        label=f"{len(runs)}-seed mean",
    )
    ax.set_title(f"{agent_type}\n{description}", fontsize=13, fontweight="semibold")
    ax.set_xlabel("Generation")
    ax.set_xlim(reference_generations.min(), reference_generations.max())
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)


def plot_evolution_curves(
    results_dir: Path | None = None,
    output_path: Path | None = None,
    agent_type1_label: str = DEFAULT_AGENT_TYPE1_LABEL,
    agent_type2_label: str = DEFAULT_AGENT_TYPE2_LABEL,
    agent_type1_seeds: Sequence[int] = (0, 1, 2, 3, 4, 5),
    agent_type2_seeds: Sequence[int] = (0, 1, 2),
    dpi: int = 180,
    write_pdf: bool = True,
) -> Path:
    """Create the matched agent-type1 versus agent-type2 evolution figure."""
    from .paths import quantitative_results_dir

    results_dir = Path(results_dir) if results_dir is not None else quantitative_results_dir()
    output_path = (
        Path(output_path)
        if output_path is not None
        else results_dir / "plots" / "evolution_curves.png"
    )

    print("agent-type1:")
    type1_runs = load_seed_runs(results_dir, agent_type1_label, agent_type1_seeds)
    print("agent-type2:")
    type2_runs = load_seed_runs(results_dir, agent_type2_label, agent_type2_seeds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharex=True, sharey=True)
    _plot_agent_panel(axes[0], "agent-type1", type1_runs)
    _plot_agent_panel(axes[1], "agent-type2", type2_runs)
    axes[0].set_ylabel("Cooperation rate")
    fig.suptitle(
        "Evolution of cooperation by agent type "
        f"({len(agent_type1_seeds)} agent-type1 seeds; "
        f"{len(agent_type2_seeds)} agent-type2 seeds)",
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.01,
        "100 generations · 1,000 target interactions per generation · "
        "population size 16 · generation reset",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))

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
    parser.add_argument("--agent-type1-label", default=DEFAULT_AGENT_TYPE1_LABEL)
    parser.add_argument("--agent-type2-label", default=DEFAULT_AGENT_TYPE2_LABEL)
    parser.add_argument("--agent-type1-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--agent-type2-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-pdf", action="store_true", help="Do not write a PDF copy")
    args = parser.parse_args()
    plot_evolution_curves(
        results_dir=args.results_dir,
        output_path=args.output,
        agent_type1_label=args.agent_type1_label,
        agent_type2_label=args.agent_type2_label,
        agent_type1_seeds=args.agent_type1_seeds,
        agent_type2_seeds=args.agent_type2_seeds,
        dpi=args.dpi,
        write_pdf=not args.no_pdf,
    )


if __name__ == "__main__":
    main()
