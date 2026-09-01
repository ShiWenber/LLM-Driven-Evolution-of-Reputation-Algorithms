"""Render the Agent 2 versus Schmid-norm bidirectional invasion dashboard."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import invasion_results_dir

NORMS = ("L1", "L2", "L7", "L8")
DIRECTIONS = ("agent2_invades_norm", "norm_invades_agent2")


def _records(input_dir: Path) -> list[dict[str, Any]]:
    """Read all completed raw invasion JSON files."""
    rows: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        for norm in NORMS:
            for path in (input_dir / direction / norm).glob("n*_seed*/invasion.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    trajectory = data["trajectory"]
                    if not trajectory:
                        continue
                    name = path.parent.name
                    n_text, seed_text = name.split("_seed", 1)
                    n = int(n_text.removeprefix("n"))
                    seed = int(seed_text)
                    rows.append({
                        "direction": direction,
                        "norm": norm,
                        "n": n,
                        "seed": seed,
                        "trajectory": trajectory,
                        "final": float(trajectory[-1]["invader_frequency"]),
                        "fixed": bool(data.get("invader_fixed", trajectory[-1]["invader_frequency"] >= 0.9)),
                    })
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    print(f"Skipping malformed result {path}: {exc}")
    if not rows:
        raise FileNotFoundError(f"No invasion JSON files found under {input_dir}")
    return rows


def _mean_trajectory(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    by_generation: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        for point in row["trajectory"]:
            by_generation[int(point["generation"])].append(float(point["invader_frequency"]))
    generations = np.array(sorted(by_generation))
    values = np.array([np.mean(by_generation[g]) for g in generations])
    return generations, values


def plot_dashboard(
    input_dir: Path | None = None,
    output: Path | None = None,
    dpi: int = 180,
) -> Path:
    """Create the four-panel dashboard and return the PNG path."""
    input_dir = Path(input_dir) if input_dir is not None else invasion_results_dir()
    output = Path(output) if output else input_dir / "agent2_schmid_bidirectional_invasion_dashboard.png"
    rows = _records(input_dir)
    by_direction = {direction: [r for r in rows if r["direction"] == direction] for direction in DIRECTIONS}

    fig, axes = plt.subplots(2, 2, figsize=(19, 12), constrained_layout=True)
    fig.patch.set_facecolor("#f7f9fb")
    colors = {"agent2_invades_norm": "#d1495b", "norm_invades_agent2": "#286f9f"}
    labels = {"agent2_invades_norm": "Agent 2 invades norm", "norm_invades_agent2": "Norm invades Agent 2"}

    # A: fixation probability by initial invader count.
    ax = axes[0, 0]
    for direction in DIRECTIONS:
        direction_rows = by_direction[direction]
        xs, ys = [], []
        for n in range(1, 15):
            subset = [r for r in direction_rows if r["n"] == n]
            if subset:
                xs.append(n)
                ys.append(np.mean([r["fixed"] for r in subset]))
        ax.plot(xs, ys, marker="o" if direction == DIRECTIONS[0] else "s", linewidth=3,
                color=colors[direction], label=labels[direction])
    ax.set_title("A  Fixation probability after 50 generations", loc="left", weight="bold")
    ax.set_xlabel("Initial invader count n (out of 15)")
    ax.set_ylabel("Fixation probability across seeds")
    ax.set_ylim(-0.03, 1.05); ax.set_xticks(range(1, 15)); ax.grid(alpha=.25); ax.legend()

    # B: final frequency heatmap, one row per direction and seed.
    ax = axes[0, 1]
    matrix, row_labels = [], []
    for direction in DIRECTIONS:
        for seed in (0, 1, 2):
            values = []
            for n in range(1, 15):
                subset = [r["final"] for r in by_direction[direction] if r["seed"] == seed and r["n"] == n]
                values.append(np.mean(subset) if subset else np.nan)
            matrix.append(values)
            row_labels.append(("A2→Norm" if direction == DIRECTIONS[0] else "Norm→A2") + f"  s{seed}")
    image = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="RdYlBu_r")
    ax.set_title("B  Final invader frequency by seed", loc="left", weight="bold")
    ax.set_xlabel("Initial invader count n"); ax.set_xticks(range(14), range(1, 15)); ax.set_yticks(range(6), row_labels)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if not np.isnan(value): ax.text(j, i, f"{value:.1g}", ha="center", va="center", fontsize=8,
                                             color="white" if value > .55 else "#34495e", weight="bold")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Final invader frequency")

    # C: Agent 2 frequency trajectories for rare initial mutants.
    ax = axes[1, 0]
    for n, style in ((1, "-"), (2, "--")):
        for seed in (0, 1, 2):
            subset = [r for r in by_direction[DIRECTIONS[0]] if r["n"] == n and r["seed"] == seed]
            if subset:
                x, y = _mean_trajectory(subset)
                ax.plot(x, y, style, color="#d1495b", alpha=.45 + .15 * seed, linewidth=2,
                        label=f"n={n}, seed={seed}")
    ax.set_title("C  Agent 2 invades: rare mutants", loc="left", weight="bold")
    ax.set_xlabel("Generation"); ax.set_ylabel("Agent 2 frequency"); ax.set_ylim(-.03, 1.05); ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2)

    # D: norm frequency for high initial counts.
    ax = axes[1, 1]
    for n, color in ((12, "#7ba6bd"), (13, "#397b9f"), (14, "#164d70")):
        subset = [r for r in by_direction[DIRECTIONS[1]] if r["n"] == n]
        if subset:
            x, y = _mean_trajectory(subset)
            ax.plot(x, y, color=color, linewidth=3, label=f"n={n} mean ({len({r['seed'] for r in subset})} seeds)")
    ax.set_title("D  Norm invades Agent 2: high initial counts", loc="left", weight="bold")
    ax.set_xlabel("Generation"); ax.set_ylabel("Norm frequency"); ax.set_ylim(-.03, 1.05); ax.grid(alpha=.25); ax.legend()

    fig.suptitle("Agent 2 vs. Schmid Robust Norms — Bidirectional Invasion", fontsize=22, weight="bold", x=.03, ha="left")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor=fig.get_facecolor())
    fig.savefig(output.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved: {output}")
    print(f"saved: {output.with_suffix('.pdf')}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="invasion result directory (default: project results)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    plot_dashboard(args.input_dir, args.output, args.dpi)


if __name__ == "__main__":
    main()
