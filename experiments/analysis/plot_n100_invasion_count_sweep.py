"""Plot invasion-count response curves.

The host population size is read from the summary, so both the archived N=100
sweeps and the current N=20 sweeps plot through the same code path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .invasion.core import ARCHIVED_DIRECTION_LABEL
from .paths import project_root


ROOT = project_root()
DEFAULT_SUMMARY = ROOT / "results" / "quantitative_baseline" / "invasion" / "n100_invasion_count_sweep" / "summary.json"
DEFAULT_OUTPUT = ROOT / "README.assets" / "n100_invasion_count_sweep.png"
COLORS = ("#2878B5", "#D1495B", "#2A9D8F", "#E9C46A", "#7B2CBF", "#F77F00", "#5F6F52", "#6C757D")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
# Unconditional baselines are drawn as thick black/grey reference curves so
# they never collide with a leading-eight colour.
UNCONDITIONAL_STYLE = {"ALLC": ("#111111", ":"), "ALLD": ("#999999", ":")}


def _styles(norms: tuple[str, ...]) -> dict[str, dict[str, object]]:
    """Assign a distinct colour/marker to every norm, special-casing ALLC/ALLD."""
    from matplotlib import colormaps

    palette = [colormaps["tab20"](i) for i in range(20)]
    styles: dict[str, dict[str, object]] = {}
    index = 0
    for norm in norms:
        if norm in UNCONDITIONAL_STYLE:
            color, linestyle = UNCONDITIONAL_STYLE[norm]
            styles[norm] = {
                "color": color, "linestyle": linestyle, "marker": "o",
                "markersize": 4, "linewidth": 2.4, "zorder": 5,
            }
            continue
        styles[norm] = {
            "color": COLORS[index % len(COLORS)] if len(norms) <= len(COLORS) + 2
            else palette[index % len(palette)],
            "linestyle": "-",
            "marker": MARKERS[index % len(MARKERS)],
            "markersize": 4,
            "linewidth": 1.7,
        }
        index += 1
    return styles


def _is_cell(value: object) -> bool:
    return isinstance(value, dict) and "runs" in value


def _norm_groups(summary: dict, agent: str) -> dict:
    """Return ``{norm: {count: cell}}`` for one source.

    Archived summaries nest a direction level that the experiment no longer has
    (``groups[label][direction][norm][count]``). Collapsing it here keeps those
    datasets plottable without reintroducing the concept.
    """
    by_norm = summary["groups"][agent]
    if not by_norm:
        raise ValueError("Summary contains no norm groups")
    first = next(iter(by_norm.values()))
    flat = not isinstance(first, dict) or any(
        _is_cell(value) for value in first.values()
    )
    if flat:
        return by_norm
    archived = by_norm.get(ARCHIVED_DIRECTION_LABEL)
    return archived if archived is not None else next(iter(by_norm.values()))


def plot(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    agents = tuple(summary.get("sources", {}).keys())
    if not agents:
        raise ValueError("Summary contains no strategy sources")
    # Norms are read dynamically from the summary (no hard-coded labels); the
    # key order of the first source cell defines the plotting order.
    first_agent = agents[0]
    groups = {agent: _norm_groups(summary, agent) for agent in agents}
    norms = tuple(groups[first_agent].keys())
    if not norms:
        raise ValueError("Summary contains no norm groups")
    # Completeness is checked per sweep, so an archived summary that also stored
    # a second direction still validates.
    per_sweep = (
        len(agents) * len(norms)
        * len(summary["initial_invader_counts"]) * len(summary["seeds"])
    )
    completed = summary.get("completed_or_cached_runs")
    if not completed or completed < per_sweep or completed % per_sweep:
        raise ValueError(
            f"Incomplete sweep: expected a positive multiple of {per_sweep} "
            f"runs, found {completed}"
        )
    if summary.get("selection") != "synchronous_deterministic_payoff_imitation":
        raise ValueError("Expected deterministic payoff imitation")
    population_size = summary.get("population_size")
    if not isinstance(population_size, int) or population_size < 3:
        raise ValueError(
            f"Summary records an unusable population_size: {population_size!r}"
        )
    if any(not 1 <= count < population_size for count in summary["initial_invader_counts"]):
        raise ValueError(
            f"initial_invader_counts must be in 1..{population_size - 1} "
            f"for population_size {population_size}"
        )

    counts = np.asarray(summary["initial_invader_counts"], dtype=float)
    x = counts / float(population_size)
    fig, axes = plt.subplots(
        1, len(agents), figsize=(5.8 * len(agents), 4.5),
        sharex=True, sharey=True, squeeze=False,
    )
    norm_styles = _styles(norms)
    for col, agent in enumerate(agents):
        ax = axes[0, col]
        ax.plot([0, 1], [0, 1], color="#222222", linewidth=1.2, linestyle="--", label="No frequency change")
        for norm in norms:
            cells = groups[agent][norm]
            means = np.asarray([cells[str(int(n))]["mean_final_invader_frequency"] for n in counts])
            ax.plot(x, means, label=norm, **norm_styles[norm])
        title = "Candidate strategy invades host"
        ax.set_title(f"{agent} — {title}" if len(agents) > 1 else title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6), [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
        ax.set_yticks(np.linspace(0, 1, 6), [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
        ax.grid(color="#D9D9D9", linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_xlabel("Initial candidate share")
        if col == 0:
            ax.set_ylabel(
                f"Mean final candidate share ({len(summary['seeds'])} seeds)"
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol = min(len(labels), 6 if len(labels) > 10 else 9)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.86),
               ncol=ncol, frameon=False)
    action_error = float(summary.get("action_error_probability", 0.0))
    observation_error = float(summary.get("observation_error_probability", 0.0))
    if action_error or observation_error:
        title = (
            f"N={population_size} invasion ability with "
            f"{action_error:.0%} action error + {observation_error:.0%} observation error"
        )
    else:
        title = (
            f"N={population_size} invasion ability across initial invader counts"
        )
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0.03, 0.04, 0.98, 0.82), h_pad=2.2, w_pad=1.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
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
