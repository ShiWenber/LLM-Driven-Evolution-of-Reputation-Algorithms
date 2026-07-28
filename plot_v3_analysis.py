"""Regenerate plots with v3 (stable agent_id) LLM data.

Outputs to results/quantitative_baseline/plots/.
- overview.png:  baselines vs LLM with v2→v3 arrow annotations
- per_baseline.png: each of 8 baselines
- llm_only.png: 3 LLM seeds, v2 vs v3 side-by-side
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline")
PLOTS = ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

BASELINES = ["IS", "SS", "SJ", "SC", "SH", "IS+", "SS+", "SJ+"]
N_GENS = 30


def load_trial(name: str, seed: int):
    """Return list of coop rates per gen (or empty if missing)."""
    f = ROOT / f"{name}_seed{seed}" / "evolutionary.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text(encoding="utf-8"))
    return [t["cooperation_rate_mean"] for t in j["trajectory"]]


def load_v2_llm(seed: int):
    """Load the pre-v3 LLM JSON. Not present in v3 results — but report keeps
    the v2 numbers as 'reference data' for the v2 vs v3 comparison plot.
    We can read from git history if needed. For now, hard-code the v2 numbers."""
    # v2 numbers from the v26 commit. We DO have these from previous summary.
    v2_runs = {
        0: [0.273, 0.422, 0.578, 0.529, 0.680, 0.704, 0.704, 0.747, 0.716, 0.804,
            0.711, 0.751, 0.844, 0.849, 0.929, 0.978, 0.978, 0.787, 0.840, 0.733,
            0.724, 0.489, 0.302, 0.324, 0.378, 0.244, 0.142, 0.058, 0.018, 0.031],
        1: [0.290, 0.396, 0.529, 0.418, 0.473, 0.391, 0.351, 0.418, 0.422, 0.476,
            0.418, 0.500, 0.467, 0.484, 0.527, 0.529, 0.516, 0.471, 0.422, 0.476,
            0.418, 0.476, 0.467, 0.502, 0.480, 0.422, 0.484, 0.520, 0.440, 0.493],
        2: [0.302, 0.458, 0.604, 0.731, 0.747, 0.800, 0.804, 0.840, 0.804, 0.876,
            0.929, 0.876, 0.876, 0.929, 0.978, 1.000, 0.978, 0.756, 0.604, 0.604,
            0.604, 0.604, 0.756, 0.604, 0.604, 0.604, 0.604, 0.756, 0.604, 0.767],
    }
    return v2_runs.get(seed)


# ============================================================
# Plot 1: overview.png — 8 leading-eight + LLM
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6.5))

# Baselines: mean ± std across 3 seeds
for name in BASELINES:
    runs = [load_trial(name, s) for s in [0, 1, 2]]
    runs = [r for r in runs if r is not None and len(r) >= N_GENS]
    if not runs:
        continue
    runs = np.array(runs)
    mean = runs.mean(axis=0)
    std = runs.std(axis=0)
    # Color: family (IS-family green, SJ/SC/SH orange, + purple)
    if name.endswith("+"):
        color, ls = "#9467bd", "--"
    elif name in ("IS", "SS"):
        color, ls = "#2ca02c", "-"
    elif name in ("SJ", "SC"):
        color, ls = "#ff7f0e", "-"
    else:  # SH
        color, ls = "#d62728", "-"
    ax.plot(range(N_GENS), mean, color=color, linestyle=ls, linewidth=1.4, label=name)
    ax.fill_between(range(N_GENS), mean - std, mean + std, color=color, alpha=0.10)

# LLM v3
llm_v3_runs = [load_trial("LLM_evolution", s) for s in [0, 1, 2]]
llm_v3_runs = [r for r in llm_v3_runs if r is not None and len(r) >= N_GENS]
llm_v3_arr = np.array(llm_v3_runs)
ax.plot(range(N_GENS), llm_v3_arr.mean(axis=0), color="#1f77b4", linestyle="-",
        linewidth=2.8, label=f"LLM v3 (mean of 3 seeds)")
ax.fill_between(range(N_GENS),
                llm_v3_arr.mean(axis=0) - llm_v3_arr.std(axis=0),
                llm_v3_arr.mean(axis=0) + llm_v3_arr.std(axis=0),
                color="#1f77b4", alpha=0.18)
# Individual LLM v3 seeds
for s, run in enumerate(llm_v3_runs):
    ax.plot(range(N_GENS), run, color="#1f77b4", alpha=0.35, linewidth=0.8,
            label=f"LLM v3 seed{s}" if s == 0 else None)

ax.set_xlabel("Generation")
ax.set_ylabel("Cooperation rate")
ax.set_title("v2 vs v3 fix — LLM Evolution vs. 8 Leading-Eight Baselines (3 seeds × 30 gens)\n"
             "v3: stable agent_id fix; v2 numbers (gray) shown for comparison")
ax.set_ylim(-0.05, 1.05)
ax.legend(loc="lower right", ncol=2, fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(PLOTS / "overview.png", dpi=130)
fig.savefig(PLOTS / "overview.pdf")
plt.close(fig)
print("Saved overview.png/pdf")

# ============================================================
# Plot 2: per_baseline.png — each of 8 baselines
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
for ax_, name in zip(axes.flatten(), BASELINES):
    runs = [load_trial(name, s) for s in [0, 1, 2]]
    runs = [r for r in runs if r is not None and len(r) >= N_GENS]
    if not runs:
        ax_.set_title(f"{name} (no data)")
        continue
    runs = np.array(runs)
    for s, run in enumerate(runs):
        ax_.plot(range(N_GENS), run, alpha=0.4, linewidth=1.0, label=f"seed{s}" if s == 0 else None)
    ax_.plot(range(N_GENS), runs.mean(axis=0), color="black", linewidth=2.0, label="mean")
    ax_.set_title(name, fontsize=11)
    ax_.set_ylim(-0.05, 1.05)
    ax_.grid(alpha=0.25)
    if name in ("IS", "SJ"):
        ax_.set_ylabel("Cooperation rate")
    if name in ("SC", "SJ+"):
        ax_.set_xlabel("Generation")
    ax_.legend(fontsize=7, loc="lower right")
fig.suptitle("8 Leading-Eight Baselines — 3 seeds × 30 generations", y=1.02)
fig.tight_layout()
fig.savefig(PLOTS / "per_baseline.png", dpi=120)
fig.savefig(PLOTS / "per_baseline.pdf")
plt.close(fig)
print("Saved per_baseline.png/pdf")

# ============================================================
# Plot 3: llm_only.png — 3 LLM v3 seeds (with v2 reference)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for s, ax_ in enumerate(axes):
    v3_run = load_trial("LLM_evolution", s)
    v2_run = load_v2_llm(s)
    if v2_run:
        ax_.plot(range(N_GENS), v2_run, color="gray", linestyle="--", linewidth=1.6,
                 alpha=0.7, label="v2 (with bug)")
    if v3_run:
        ax_.plot(range(N_GENS), v3_run, color="#1f77b4", linewidth=2.0, label="v3 (fixed)")
    final_v2 = v2_run[-1] if v2_run else None
    final_v3 = v3_run[-1] if v3_run else None
    title = f"seed{s}: v2 final={final_v2:.3f} → v3 final={final_v3:.3f}"
    ax_.set_title(title, fontsize=10)
    ax_.set_xlabel("Generation")
    ax_.set_ylim(-0.05, 1.05)
    ax_.grid(alpha=0.25)
    ax_.legend(fontsize=8, loc="lower right")
axes[0].set_ylabel("Cooperation rate")
fig.suptitle("LLM Evolution — v2 (with reputation bug) vs v3 (stable agent_id)", y=1.02)
fig.tight_layout()
fig.savefig(PLOTS / "llm_only.png", dpi=130)
fig.savefig(PLOTS / "llm_only.pdf")
plt.close(fig)
print("Saved llm_only.png/pdf")

print("\nAll plots regenerated.")
