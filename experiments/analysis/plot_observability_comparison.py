"""Cross-condition comparison: observability rate (p) x agent type.

Loads the Fermi 100-gen / 1000-interaction / N=16 / generation-reset
results for agent-type1 (seeds 0-5) and agent-type2 (seeds 0-2) under
three observability settings:

  * p = 1.0 (full)  — agent-type1: LLM_agent-type1_..._N16_genreset
                      agent-type2: LLM_v3_..._N16_genreset (legacy label)
  * p = 0.5 (partial) — ..._partial0p5
  * p = 0.1 (partial) — ..._partial0p1

For each condition it computes final-generation cooperation rate and
fitness (mean +/- SE), runs pairwise Mann-Whitney U tests (hand-rolled,
no scipy) between observation levels within each agent type, and
reports Cohen's d effect sizes.  Produces a two-panel figure with bar +
per-seed scatter + SE error bars.

Outputs:
  * console table (mean +/- SE, U, p, d)
  * <plots>/observability_comparison.png
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.evolution_log import (
    F_COOPERATION_RATE_MEAN,
    F_FITNESS_MEAN,
    K_TRAJECTORY,
    load_evolution_json,
)
from .paths import project_root, quantitative_results_dir


AT1_FULL = "LLM_agent-type1_fermi_z_v3_g100_1000inter_N16_genreset"
AT2_FULL_LEGACY = "LLM_v3_fermi_z_v3_g100_1000inter_N16_genreset"  # pre-rename full run
AT2_BASE = "LLM_agent-type2_fermi_z_v3_g100_1000inter_N16_genreset"

# (label, observability p, seeds)
CONDITIONS = [
    ("agent-type1", 1.0, AT1_FULL, [0, 1, 2, 3, 4, 5]),
    ("agent-type1", 0.5, AT1_FULL + "_partial0p5", [0, 1, 2, 3, 4, 5]),
    ("agent-type1", 0.1, AT1_FULL + "_partial0p1", [0, 1, 2, 3, 4, 5]),
    ("agent-type2", 1.0, AT2_FULL_LEGACY, [0, 1, 2]),
    ("agent-type2", 0.5, AT2_BASE + "_partial0p5", [0, 1, 2]),
    ("agent-type2", 0.1, AT2_BASE + "_partial0p1", [0, 1, 2]),
]

COLORS = {
    "agent-type1": "#2166ac",
    "agent-type2": "#b2182b",
}


# ---------------------------------------------------------------------------
# Statistics (zero-dependency: numpy only)
# ---------------------------------------------------------------------------
def _se(x: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    if a.size < 2:
        return 0.0
    return float(a.std(ddof=1) / math.sqrt(a.size))


def mann_whitney_u(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U with tie correction (hand-rolled).

    Returns (u_stat, p_value) using the normal approximation with
    continuity correction, which is fine for the small seed counts here.
    """
    nx, ny = len(x), len(y)
    combined = [(float(v), 0) for v in x] + [(float(v), 1) for v in y]
    combined.sort(key=lambda t: t[0])
    # Ranks with ties -> average rank
    ranks = [0.0] * (nx + ny)
    i = 0
    n = nx + ny
    while i < n:
        j = i
        while j < n and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    r1 = sum(r for k, r in enumerate(ranks) if combined[k][1] == 0)
    u1 = r1 - nx * (nx + 1) / 2.0
    u2 = nx * ny - u1
    u = min(u1, u2)
    mu = nx * ny / 2.0
    # Variance with tie correction
    tie_counts = {}
    for v, _ in combined:
        tie_counts[v] = tie_counts.get(v, 0) + 1
    tie_adj = sum(t**3 - t for t in tie_counts.values() if t > 1)
    var = (nx * ny / 12.0) * ((n + 1) - tie_adj / (n * (n - 1)))
    if var <= 0:
        return u, 1.0
    z = (u - mu - 0.5) / math.sqrt(var)  # continuity correction
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return u, p


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = a.size, b.size
    if na + nb <= 2:
        return 0.0
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((ma - mb) / pooled)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_seed_final(results_dir: Path, label: str, seed: int) -> dict[str, float] | None:
    path = canonical_json_path(results_dir, label, seed)
    if not path.exists():
        return None
    data: dict[str, Any] = load_evolution_json(path)
    last = data.get(K_TRAJECTORY, [])[-1]
    return {
        "coop": float(last[F_COOPERATION_RATE_MEAN]),
        "fitness": float(last[F_FITNESS_MEAN]),
    }


def canonical_json_path(results_dir: Path, label: str, seed: int) -> Path:
    """Per-seed evolution JSON: <results_dir>/<label>_seed<N>/evolutionary.json."""
    return results_dir / f"{label}_seed{seed}" / "evolutionary.json"


def collect(results_dir: Path) -> dict[tuple[str, float], list[dict[str, float]]]:
    out: dict[tuple[str, float], list[dict[str, float]]] = {}
    for agent_type, p, label, seeds in CONDITIONS:
        rows = []
        for s in seeds:
            row = _load_seed_final(results_dir, label, s)
            if row is not None:
                rows.append(row)
        out[(agent_type, p)] = rows
    return out


# ---------------------------------------------------------------------------
# Reporting / plotting
# ---------------------------------------------------------------------------
def _report(metric: str, data: dict[tuple[str, float], list[dict[str, float]]]) -> str:
    lines = []
    header = (f"{metric:>8} | {'agent-type':<11} | {'p':>5} | {'n':>2} | "
              f"{'mean':>7} | {'SE':>6} | {'min':>6} | {'max':>6}")
    lines.append(header)
    lines.append("-" * len(header))
    for agent_type in ("agent-type1", "agent-type2"):
        for p in (1.0, 0.5, 0.1):
            vals = [r[metric] for r in data[(agent_type, p)]]
            if not vals:
                continue
            mean = float(np.mean(vals))
            se = _se(vals)
            lines.append(
                f"{metric:>8} | {agent_type:<11} | {p:>5} | {len(vals):>2} | "
                f"{mean:>7.3f} | {se:>6.3f} | {min(vals):>6.3f} | {max(vals):>6.3f}"
            )
    return "\n".join(lines)


def _pairwise(metric: str, data: dict[tuple[str, float], list[dict[str, float]]]) -> str:
    lines = []
    lines.append(f"Pairwise Mann-Whitney U / Cohen's d  ({metric})")
    for agent_type in ("agent-type1", "agent-type2"):
        for (pa, pb) in ((1.0, 0.5), (0.5, 0.1), (1.0, 0.1)):
            xa = [r[metric] for r in data[(agent_type, pa)]]
            xb = [r[metric] for r in data[(agent_type, pb)]]
            u, p = mann_whitney_u(xa, xb)
            d = cohens_d(xa, xb)
            lines.append(
                f"  {agent_type:<11} p={pa:>3} vs p={pb:>3}: "
                f"U={u:.1f}, p={p:.3f}, d={d:+.3f} (n={len(xa)} vs {len(xb)})"
            )
    # agent-type1 vs agent-type2 at matched p
    lines.append("agent-type1 vs agent-type2 (matched p)")
    for p in (1.0, 0.5, 0.1):
        xa = [r[metric] for r in data[("agent-type1", p)]]
        xb = [r[metric] for r in data[("agent-type2", p)]]
        u, pv = mann_whitney_u(xa, xb)
        d = cohens_d(xa, xb)
        lines.append(
            f"  p={p:>3}: U={u:.1f}, p={pv:.3f}, d={d:+.3f} "
            f"(n={len(xa)} vs {len(xb)})"
        )
    return "\n".join(lines)


def _plot(data: dict[tuple[str, float], list[dict[str, float]]], out_path: Path):
    ps = [1.0, 0.5, 0.1]
    xpos = np.arange(len(ps))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, metric, ylabel in (
        (axes[0], "coop", "final cooperation rate"),
        (axes[1], "fitness", "final fitness"),
    ):
        for j, agent_type in enumerate(("agent-type1", "agent-type2")):
            means, ses = [], []
            for p in ps:
                vals = [r[metric] for r in data[(agent_type, p)]]
                means.append(float(np.mean(vals)) if vals else float("nan"))
                ses.append(_se(vals) if vals else float("nan"))
            off = (j - 0.5) * width
            ax.bar(
                xpos + off, means, width, yerr=ses,
                color=COLORS[agent_type], alpha=0.85, capsize=4,
                label=f"{agent_type} (n={len(data[(agent_type, 1.0)])})",
            )
            # per-seed scatter
            for k, p in enumerate(ps):
                vals = [r[metric] for r in data[(agent_type, p)]]
                xs = np.full(len(vals), xpos[k] + off)
                ax.scatter(xs, vals, color="black", s=18, zorder=5,
                           edgecolors="white", linewidths=0.4)
        ax.set_xticks(xpos, [f"p=1.0\n(full)", "p=0.5", "p=0.1"])
        ax.set_xlabel("observability (third-party observation probability)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle("100 gen x 1000 interactions, N=16, Fermi Z-like (mu=0.1, beta=5.0)", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {out_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Quantitative-baseline results root (default: results/quantitative_baseline)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Where to write the PNG (default: <results>/plots)")
    args = parser.parse_args(argv)

    root = project_root()
    results_dir = Path(args.results_dir) if args.results_dir else quantitative_results_dir(root)
    data = collect(results_dir)

    print("=== final-cooperation / fitness by observability p ===\n")
    print(_report("coop", data))
    print()
    print(_report("fitness", data))
    print()
    print(_pairwise("coop", data))
    print()
    print(_pairwise("fitness", data))
    print()

    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "plots"
    _plot(data, out_dir / "observability_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
