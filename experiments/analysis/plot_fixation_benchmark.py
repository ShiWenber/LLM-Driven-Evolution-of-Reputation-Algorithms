"""Plot a fixation benchmark: payoff-difference curves and fixation probabilities.

The payoff-difference panel is the informative one. For a two-type mixture with
``k`` candidates and ``N-k`` probes it plots

    d(k) = pi_candidate(k) - pi_probe(k)

against the candidate's share ``k/N``. Above zero the candidate is the fitter
type at that composition. A curve that crosses zero tells you the pair has an
INTERIOR equilibrium: the candidate is favoured only while rare (coexistence)
or only while common (bistability). That sign structure, not the endpoint of a
short imitation run, is what the fixation probability integrates.
"""

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
DEFAULT_OUTPUT = ROOT / "README.assets" / "fixation_benchmark.png"
PROBE_COLORS = {
    "ALLC": "#111111",
    "ALLD": "#999999",
}


def _color(index: int, probe: str) -> object:
    if probe in PROBE_COLORS:
        return PROBE_COLORS[probe]
    return plt.get_cmap("tab10")(index % 10)


def plot(summary_path: Path, output_path: Path) -> None:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    neutral = float(data["config"]["neutral_fixation_probability"])
    n = int(data["config"]["population_size"])
    label = data["candidate"]["label"]
    ae = float(data["config"]["action_error_probability"])
    oe = float(data["config"]["observation_error_probability"])
    probes = list(data["results"].keys())

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))

    # ---- panel A: payoff-difference curves -------------------------------
    ax = axes[0]
    ax.axhline(0.0, color="#222222", linewidth=1.4, linestyle="--")
    for index, probe in enumerate(probes):
        curve = data["results"][probe]["candidate_invades_probe"]["curve"]
        ks = np.asarray([c["mutant_count"] for c in curve], dtype=float)
        diffs = np.asarray([c["payoff_difference"] for c in curve], dtype=float)
        heavy = probe in PROBE_COLORS
        color = _color(index, probe)
        stds = np.asarray(
            [c.get("payoff_difference_std", 0.0) for c in curve], dtype=float
        )
        if np.any(stds > 0):
            # Replicate spread: a wide band means the composition is bimodal
            # (more than one reputation basin), so its mean is fragile.
            ax.fill_between(
                ks / n, diffs - stds, diffs + stds, color=color, alpha=0.13,
                linewidth=0, zorder=1,
            )
        ax.plot(
            ks / n, diffs, label=probe, color=color,
            linewidth=2.6 if heavy else 1.5,
            marker="o" if heavy else None, markersize=4,
            alpha=1.0 if heavy else 0.85, zorder=3 if heavy else 2,
        )
    ax.set_xlabel("Candidate share $k/N$")
    ax.set_ylabel(r"$\pi_{\rm candidate}(k) - \pi_{\rm probe}(k)$")
    stationarity = data.get("stationarity", {})
    drifting = {
        probe: rep["max_half_to_half_gap"]
        for probe, rep in stationarity.items()
        if rep.get("max_half_to_half_gap", 0.0) > 0.10
    }
    title = (
        "Payoff advantage of the candidate, per composition\n"
        "above 0 = candidate fitter at that mix; crossing 0 = interior equilibrium"
    )
    reps = data["config"].get("replicates", 1)
    if reps > 1:
        title += f"\nshaded band = $\\pm$1 s.d. over {reps} replicates"
    if drifting:
        worst_probe, worst_gap = max(drifting.items(), key=lambda kv: kv[1])
        title += (
            f"\n(!) {len(drifting)} probe(s) still drifting "
            f"(worst {worst_probe} at {worst_gap:.2f}); raise --burn-in"
        )
    ax.set_title(title, fontsize=11)
    for probe in drifting:
        worst = stationarity[probe]["worst_composition"]
        if worst and worst.get("mutant_count") is not None:
            ax.axvline(worst["mutant_count"] / n, color="#D1495B",
                       linewidth=0.8, alpha=0.35, zorder=0)
    ax.grid(color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9, ncol=2, frameon=False)

    # ---- panel B: fixation probabilities ---------------------------------
    ax = axes[1]
    x = np.arange(len(probes))
    forward = np.asarray(
        [data["results"][p]["candidate_invades_probe"]["rho"] for p in probes]
    )
    width = 0.5
    ax.axhline(neutral, color="#222222", linewidth=1.6, linestyle="--",
               label=f"neutral $1/N$ = {neutral:.3f}")
    ax.bar(x, forward, width, label=f"{label} invades probe", color="#2878B5")
    # Show how far rho would move if each composition had landed in its other
    # basin; a tall whisker is a warning, not a confidence interval.
    los, his = [], []
    for probe, value in zip(probes, forward, strict=True):
        side = data["results"][probe]["candidate_invades_probe"]
        if "rho_at_minus_1sd" in side:
            los.append(max(0.0, value - side["rho_at_minus_1sd"]))
            his.append(max(0.0, side["rho_at_plus_1sd"] - value))
        else:
            los.append(0.0)
            his.append(0.0)
    if any(los) or any(his):
        ax.errorbar(
            x, forward, yerr=[los, his], fmt="none",
            ecolor="#333333", elinewidth=1.1, capsize=3, zorder=5,
        )
    ax.set_xticks(x, probes)
    ax.set_ylabel(r"Fixation probability $\rho$")
    if reps > 1:
        ax.set_title(
            "Fixation probability vs the neutral benchmark\n"
            f"whiskers = rho at $\\pm$1 s.d. of the {reps}-replicate payoff spread",
            fontsize=11,
        )
    else:
        ax.set_title(
            "Fixation probability vs the neutral benchmark\n"
            "bar above dashed = candidate can invade the probe",
            fontsize=11,
        )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

    fig.suptitle(
        f"Schmid-style fixation benchmark — {label} "
        f"(N={n}, {ae:.0%} action + {oe:.0%} observation error, "
        f"$\\beta$={data['config']['beta']:g})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.93), w_pad=2.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.summary.resolve(), args.output.resolve())
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
