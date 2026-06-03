"""Result aggregation and figure generation.

Reads JSON trial results from experiments/results/ and produces:
- Summary tables (CSV / markdown)
- Figure 1: Cooperation trajectories (PRIVATE vs FULL)
- Figure 2: Phase transition — final cooperation rate vs observability p
- Figure 3: Representative evolved strategy code at three observability levels
- Figure 4: IPD baseline — evolutionary dynamics under full information

Usage:
    python -m experiments.analysis.make_figures
    python -m experiments.analysis.make_figures --results-dir results --output-dir results/figures
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np


# ---------- Result loading ----------

def load_donor_trials(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all donor-game evolutionary trials from results/.

    Two file formats are supported:
      A. Single-trial: {config, trajectory, final_population}
         (evo_*.json, named with the observability level)
      B. Aggregate: {trials: [{config, trajectory, ...}, ...]}
         (evolutionary_*.json — produced by main.py _save_aggregate)
    """
    trials = []
    for p in results_dir.glob("evo_*.json"):
        with open(p) as f:
            data = json.load(f)
        if "config" in data and "trajectory" in data:
            data["observability"] = data["config"].get("observability", "unknown")
            data["_source_file"] = p.name
            trials.append(data)
        elif "trials" in data:
            for t in data["trials"]:
                t["_source_file"] = p.name
                trials.append(t)
    for p in results_dir.glob("evolutionary_*.json"):
        with open(p) as f:
            data = json.load(f)
        for t in data.get("trials", []):
            t["_source_file"] = p.name
            trials.append(t)
    return trials


def load_ipd_trials(results_dir: Path) -> List[Dict[str, Any]]:
    """Load IPD baseline trials from results/ipd_baseline/."""
    trials = []
    ipd_dir = results_dir / "ipd_baseline"
    if not ipd_dir.exists():
        return trials
    for p in ipd_dir.glob("trial_seed*.json"):
        with open(p) as f:
            data = json.load(f)
        data["_source_file"] = p.name
        trials.append(data)
    return trials


# ---------- Donor-game aggregation ----------

def parse_observability(trial: Dict[str, Any]) -> str:
    """Extract observability label from a trial record."""
    return trial.get("observability", "unknown")


def final_cooperation(trial: Dict[str, Any]) -> Optional[float]:
    """Last generation's mean cooperation rate from a trial."""
    traj = trial.get("trajectory", [])
    if not traj:
        return None
    last = traj[-1]
    return last.get("cooperation_rate_mean") or last.get("mean_cooperation")


def final_payoff(trial: Dict[str, Any]) -> Optional[float]:
    traj = trial.get("trajectory", [])
    if not traj:
        return None
    last = traj[-1]
    return last.get("payoff_mean") or last.get("mean_payoff")


def aggregate_donor_by_observability(trials: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Compute mean ± std of final cooperation rate per observability level."""
    by_obs: Dict[str, List[float]] = defaultdict(list)
    for t in trials:
        coop = final_cooperation(t)
        if coop is None:
            continue
        by_obs[parse_observability(t)].append(coop)

    summary: Dict[str, Dict[str, float]] = {}
    for obs, vals in by_obs.items():
        summary[obs] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return summary


def obs_label_to_p(obs: str) -> Optional[float]:
    """Map observability string to numeric p value."""
    if obs == "private":
        return 0.0
    if obs == "full":
        return 1.0
    m = re.match(r"partial[_ ]?([0-9.]+)", obs)
    if m:
        return float(m.group(1))
    return None


# ---------- Figure rendering (matplotlib) ----------

def make_figure_1_obs_contrast(trials: List[Dict], output_dir: Path):
    """Cooperation trajectories across generations: PRIVATE vs FULL."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    obs_to_color = {"private": "#d62728", "full": "#1f77b4"}

    for obs, color in obs_to_color.items():
        obs_trials = [t for t in trials if parse_observability(t) == obs]
        if not obs_trials:
            continue
        for t in obs_trials:
            traj = t.get("trajectory", [])
            if not traj:
                continue
            xs = [g.get("generation", i) for i, g in enumerate(traj)]
            ys = [g.get("cooperation_rate_mean") or g.get("mean_cooperation") for g in traj]
            ax.plot(xs, ys, color=color, alpha=0.4, linewidth=1)

        # Mean trajectory
        max_len = max(len(t.get("trajectory", [])) for t in obs_trials)
        all_xs = list(range(max_len))
        all_ys: List[List[float]] = []
        for t in obs_trials:
            traj = t.get("trajectory", [])
            ys = [g.get("cooperation_rate_mean") or g.get("mean_cooperation") for g in traj]
            all_ys.append(ys + [None] * (max_len - len(ys)))
        arr = np.array([[y if y is not None else np.nan for y in row] for row in all_ys])
        mean_ys = np.nanmean(arr, axis=0)
        ax.plot(all_xs, mean_ys, color=color, linewidth=2.5, label=f"{obs.upper()} (n={len(obs_trials)})")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean cooperation rate")
    ax.set_title("Cooperation Trajectories: PRIVATE (p=0) vs FULL (p=1)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / "fig1_obs_contrast.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def make_figure_2_phase_transition(trials: List[Dict], output_dir: Path):
    """Final cooperation vs observability p — the phase transition diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = aggregate_donor_by_observability(trials)
    rows = []
    for obs, stats in summary.items():
        p = obs_label_to_p(obs)
        if p is None:
            continue
        rows.append((p, stats["mean"], stats["std"], stats["n"]))
    rows.sort(key=lambda r: r[0])

    if not rows:
        print("  [fig2] no data, skipping")
        return

    ps = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    ns = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ps, means, yerr=stds, fmt="o-", capsize=5, color="#1f77b4",
                linewidth=2, markersize=8, label="Mean ± std")

    # Annotate n
    for p, m, s, n in zip(ps, means, stds, ns):
        ax.annotate(f"n={n}", (p, m), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="gray")

    # Vertical band marking the phase transition
    ax.axvspan(0.1, 0.3, alpha=0.15, color="red", label="Phase transition band (0.1-0.3)")

    ax.set_xlabel("Observability p (fraction of interactions observed)")
    ax.set_ylabel("Final generation cooperation rate")
    ax.set_title("Phase Transition: Cooperation vs Observability")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / "fig2_phase_transition.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")
    return rows


def make_figure_4_ipd_baseline(trials: List[Dict], output_dir: Path):
    """IPD baseline: cooperation rate across generations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not trials:
        print("  [fig4] no IPD data, skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for t in trials:
        traj = t.get("trajectory", [])
        if not traj:
            continue
        xs = [g["generation"] for g in traj]
        ys = [g["mean_cooperation"] for g in traj]
        seed = t.get("config", {}).get("seed", "?")
        ax.plot(xs, ys, alpha=0.5, linewidth=1, label=f"seed={seed}")

    if len(trials) > 0:
        max_len = max(len(t.get("trajectory", [])) for t in trials)
        all_xs = list(range(max_len))
        all_ys = []
        for t in trials:
            traj = t.get("trajectory", [])
            ys = [g["mean_cooperation"] for g in traj]
            all_ys.append(ys + [np.nan] * (max_len - len(ys)))
        arr = np.array(all_ys)
        mean_ys = np.nanmean(arr, axis=0)
        ax.plot(all_xs, mean_ys, color="black", linewidth=3, label=f"Mean (n={len(trials)})")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean cooperation rate")
    ax.set_title("IPD Baseline: LLM-Driven Evolution in 2-Player Full-Information Game (Willis Comparison)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / "fig4_ipd_baseline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def write_summary_tables(donor_summary: Dict, ipd_summary: List[Dict], output_dir: Path):
    """Write CSV and markdown summary tables."""
    md_lines = ["# Experimental Summary\n"]

    md_lines.append("## Donor Game — Final cooperation rate by observability\n")
    md_lines.append("| Observability | n | Mean | Std | Min | Max |")
    md_lines.append("|---|---|---|---|---|---|")
    for obs in sorted(donor_summary.keys()):
        s = donor_summary[obs]
        md_lines.append(f"| {obs} | {s['n']} | {s['mean']:.3f} | {s['std']:.3f} | {s['min']:.3f} | {s['max']:.3f} |")

    if ipd_summary:
        md_lines.append("\n## IPD Baseline — Final generation\n")
        md_lines.append("| Seed | Final cooperation | Mean payoff |")
        md_lines.append("|---|---|---|")
        for t in ipd_summary:
            traj = t.get("trajectory", [])
            if not traj:
                continue
            seed = t.get("config", {}).get("seed", "?")
            md_lines.append(f"| {seed} | {traj[-1]['mean_cooperation']:.3f} | {traj[-1]['mean_payoff']:.3f} |")

    out = output_dir / "summary.md"
    out.write_text("\n".join(md_lines))
    print(f"  saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output-dir", default="results/figures")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    donor_trials = load_donor_trials(results_dir)
    ipd_trials = load_ipd_trials(results_dir)

    print(f"[figures] loaded {len(donor_trials)} donor trials, "
          f"{len(ipd_trials)} IPD trials")

    donor_summary = aggregate_donor_by_observability(donor_trials)
    print(f"[figures] donor summary: {donor_summary}")

    print("\n[figures] generating Figure 1: PRIVATE vs FULL contrast")
    make_figure_1_obs_contrast(donor_trials, output_dir)

    print("\n[figures] generating Figure 2: Phase transition diagram")
    make_figure_2_phase_transition(donor_trials, output_dir)

    print("\n[figures] generating Figure 4: IPD baseline")
    make_figure_4_ipd_baseline(ipd_trials, output_dir)

    print("\n[figures] writing summary tables")
    write_summary_tables(donor_summary, ipd_trials, output_dir)

    print("\n[figures] done.")


if __name__ == "__main__":
    main()
