"""Plot N=100 bidirectional invasion-count response curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import project_root


ROOT = project_root()
DEFAULT_SUMMARY = ROOT / "results" / "quantitative_baseline" / "invasion" / "n100_invasion_count_sweep" / "summary.json"
DEFAULT_OUTPUT = ROOT / "README.assets" / "n100_invasion_count_sweep.png"
NORMS = ("IS", "SS", "SJ", "SC", "SH", "IS+", "SS+", "SJ+")
AGENTS = ("agent-type1", "agent-type2")
DIRECTIONS = ("evolved_invades_norm", "norm_invades_evolved")
COLORS = ("#2878B5", "#D1495B", "#2A9D8F", "#E9C46A", "#7B2CBF", "#F77F00", "#5F6F52", "#6C757D")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")


def plot(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("completed_or_cached_runs") != 1248:
        raise ValueError("The N=100 README figure requires all 1,248 runs")
    if summary.get("population_size") != 100:
        raise ValueError("Expected population size 100")
    if summary.get("selection") != "synchronous_deterministic_payoff_imitation":
        raise ValueError("Expected deterministic payoff imitation")

    counts = np.asarray(summary["initial_invader_counts"], dtype=float)
    x = counts / 100.0
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for row, direction in enumerate(DIRECTIONS):
        for col, agent in enumerate(AGENTS):
            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], color="#222222", linewidth=1.2, linestyle="--", label="No frequency change")
            for norm, color, marker in zip(NORMS, COLORS, MARKERS, strict=True):
                cells = summary["groups"][agent][direction][norm]
                means = np.asarray([cells[str(int(n))]["mean_final_invader_frequency"] for n in counts])
                ax.plot(x, means, color=color, marker=marker, markersize=4, linewidth=1.7, label=norm)
            title_direction = "Evolved strategy invades" if direction == "evolved_invades_norm" else "Leading Eight norm invades"
            ax.set_title(f"{agent} — {title_direction}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks(np.linspace(0, 1, 6), [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
            ax.set_yticks(np.linspace(0, 1, 6), [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
            ax.grid(color="#D9D9D9", linewidth=0.7, alpha=0.75)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if row == 1:
                ax.set_xlabel("Initial invader share")
            if col == 0:
                ax.set_ylabel("Mean final invader share (3 seeds)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=9, frameon=False)
    action_error = float(summary.get("action_error_probability", 0.0))
    observation_error = float(summary.get("observation_error_probability", 0.0))
    if action_error or observation_error:
        title = (
            "N=100 invasion ability with "
            f"{action_error:.0%} action error + {observation_error:.0%} observation error"
        )
    else:
        title = "N=100 invasion ability across initial invader counts"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0.03, 0.04, 0.98, 0.88), h_pad=2.2, w_pad=1.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.summary.resolve(), args.output.resolve())
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
