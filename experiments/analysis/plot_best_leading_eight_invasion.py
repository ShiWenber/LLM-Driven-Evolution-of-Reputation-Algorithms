"""Plot the single-invader Leading Eight experiment as a bidirectional comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    ROOT
    / "results"
    / "quantitative_baseline"
    / "invasion"
    / "best_vs_leading_eight"
    / "summary.json"
)
DEFAULT_OUTPUT = ROOT / "README.assets" / "best_vs_leading_eight_invasion.png"
NORMS = ("IS", "SS", "SJ", "SC", "SH", "IS+", "SS+", "SJ+")
AGENT_TYPES = ("agent-type1", "agent-type2")


def _fixation_rates(summary: dict, agent_type: str, direction: str) -> list[float]:
    groups = summary["groups"][agent_type][direction]
    rates = []
    for norm in NORMS:
        cell = groups[norm]
        if cell["runs"] != 3:
            raise ValueError(
                f"Expected three seeds for {agent_type}/{direction}/{norm}; "
                f"found {cell['runs']}"
            )
        rates.append(cell["fixations"] / cell["runs"])
    return rates


def plot(summary_path: Path, output_path: Path) -> None:
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("completed_or_cached_runs") != 96:
        raise ValueError("The README figure requires the complete 96-run n=1 batch")
    if summary.get("generation_lifecycle") != "fresh_agent_and_reputation_reset":
        raise ValueError("The README figure requires population-aligned generation reset")
    if summary.get("selection") != "synchronous_deterministic_payoff_imitation":
        raise ValueError("The README figure requires deterministic payoff imitation")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 200,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)
    y = np.arange(len(NORMS))
    evolved_color = "#2878B5"
    norm_color = "#D1495B"

    for ax, agent_type in zip(axes, AGENT_TYPES, strict=True):
        evolved = np.asarray(
            _fixation_rates(summary, agent_type, "evolved_invades_norm")
        )
        norms = np.asarray(
            _fixation_rates(summary, agent_type, "norm_invades_evolved")
        )
        ax.barh(y, -norms, height=0.62, color=norm_color, label="Leading Eight invades")
        ax.barh(y, evolved, height=0.62, color=evolved_color, label="Evolved strategy invades")
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(agent_type)
        ax.set_yticks(y, NORMS)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xticks(
            [-1, -2 / 3, -1 / 3, 0, 1 / 3, 2 / 3, 1],
            ["3/3", "2/3", "1/3", "0", "1/3", "2/3", "3/3"],
        )
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for yi, left, right in zip(y, norms, evolved, strict=True):
            if left:
                ax.text(-left + 0.035, yi, f"{round(left * 3)}/3", va="center", ha="left", color="white", fontweight="bold")
            else:
                ax.text(-0.035, yi, "0/3", va="center", ha="right", color=norm_color, fontweight="bold")
            if right:
                ax.text(right - 0.035, yi, f"{round(right * 3)}/3", va="center", ha="right", color="white", fontweight="bold")
            else:
                ax.text(0.035, yi, "0/3", va="center", ha="left", color=evolved_color, fontweight="bold")

    axes[0].invert_yaxis()
    fig.suptitle("Bidirectional invasion: best evolved strategies vs. Leading Eight", fontsize=15)
    fig.supxlabel("Fixations across three seeds  ←  Leading Eight invades | Evolved strategy invades  →", y=0.035)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False)
    fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.86), w_pad=2.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
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
