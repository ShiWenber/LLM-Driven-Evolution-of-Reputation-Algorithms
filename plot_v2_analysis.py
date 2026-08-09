"""Generate v2 quantitative baseline analysis plots.

Compares LLM-evolved strategies (3 seeds, 30 generations) against the
8 leading-eight baselines (IS, SS, SJ, SC, SH, IS+, SS+, SJ+) from
Ohtsuki-Iwasa 2006 under full observability.

Outputs:
  results/quantitative_baseline/plots/overview.png     - all curves overlaid
  results/quantitative_baseline/plots/per_baseline.png - 8-panel grid
  results/quantitative_baseline/plots/llm_only.png     - LLM individual seeds
"""
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/quantitative_baseline")
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

BASELINE_ORDER = ["IS", "SS", "SJ", "SC", "SH", "IS+", "SS+", "SJ+"]
N_GENS = 30


def load_trial(name: str, seed: int):
    path = OUT / f"{name}_seed{seed}" / "evolutionary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("trajectory", [])


def load_baseline(name: str):
    """Load 3 seeds and return list of value-only trajectories."""
    trajs = []
    for seed in range(3):
        t = load_trial(name, seed)
        if t is not None and len(t) >= N_GENS:
            trajs.append([x["cooperation_rate_mean"] for x in t[:N_GENS]])
    return trajs


def load_llm_values():
    """Return list of value-only LLM trajectories."""
    vals = []
    for s in range(3):
        t = load_trial("LLM_evolution", s)
        if t is not None and len(t) >= N_GENS:
            vals.append([x["cooperation_rate_mean"] for x in t[:N_GENS]])
    return vals


def mean_std_band(trajs):
    """Given a list of per-seed trajectories, return (mean, std) arrays."""
    n = len(trajs)
    if n == 0:
        return [], []
    arr = [list(t)[:N_GENS] for t in trajs]
    mean = [statistics.mean(arr[s][g] for s in range(n)) for g in range(N_GENS)]
    if n > 1:
        std = [statistics.stdev(arr[s][g] for s in range(n)) for g in range(N_GENS)]
    else:
        std = [0.0] * N_GENS
    return mean, std


# ---- Load all data ----
baselines = {name: load_baseline(name) for name in BASELINE_ORDER}
llm_seeds = load_llm_values()
llm_mean, llm_std = mean_std_band(llm_seeds)
gens = list(range(N_GENS))

print(f"Loaded {sum(len(v) for v in baselines.values())} baseline trials, {len(llm_seeds)} LLM trials")
for name in BASELINE_ORDER:
    print(f"  {name}: {len(baselines[name])} seeds, final mean = {mean_std_band(baselines[name])[0][-1]:.3f}")
print(f"  LLM: {len(llm_seeds)} seeds, final mean = {llm_mean[-1]:.3f}")

# ---- Plot 1: overview ----
fig, ax = plt.subplots(figsize=(10, 6))
colors_b = plt.cm.tab10([0, 1, 2, 3, 4, 5, 6, 7])
for i, name in enumerate(BASELINE_ORDER):
    if not baselines[name]:
        continue
    if len(baselines[name]) >= 2:
        m, s = mean_std_band(baselines[name])
        ax.plot(gens, m, label=f"{name} (mean±std, {len(baselines[name])} seeds)", color=colors_b[i], lw=1.5, alpha=0.85)
        ax.fill_between(gens, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)], color=colors_b[i], alpha=0.08)
    else:
        ax.plot(gens, baselines[name][0], label=f"{name} (1 seed)", color=colors_b[i], lw=1.2, alpha=0.7)
# LLM
ax.plot(gens, llm_mean, label=f"LLM (mean±std, {len(llm_seeds)} seeds)", color="black", lw=2.5)
ax.fill_between(gens, [a - b for a, b in zip(llm_mean, llm_std)], [a + b for a, b in zip(llm_mean, llm_std)], color="black", alpha=0.15)
# LLM individual seeds
for s_idx, traj in enumerate(llm_seeds):
    ax.plot(gens, traj, color="gray", lw=0.8, alpha=0.5,
            label=f"LLM seed{s_idx}" if s_idx == 0 else None)
ax.axhline(1.0, color="green", ls="--", lw=0.6, alpha=0.4)
ax.axhline(0.0, color="red", ls="--", lw=0.6, alpha=0.4)
ax.set_xlabel("Generation")
ax.set_ylabel("Mean cooperation rate")
ax.set_title("v2 quantitative baseline: 8 leading-eight (Ohtsuki-Iwasa 2006) vs LLM-evolved\n(full observability, n=15, 30 rounds/gen, 30 generations)")
ax.set_ylim(-0.05, 1.1)
ax.legend(loc="lower right", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "overview.png", dpi=120)
fig.savefig(PLOTS / "overview.pdf")
plt.close(fig)
print(f"  -> {PLOTS / 'overview.png'}")

# ---- Plot 2: per-baseline subplots ----
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
for idx, name in enumerate(BASELINE_ORDER):
    ax = axes[idx // 4][idx % 4]
    # Plot individual baseline seeds (flat 1.0 typically)
    for s_idx, traj in enumerate(baselines[name]):
        ax.plot(gens, traj, color=colors_b[idx], lw=1.2, alpha=0.6,
                label=f"{name} seed{s_idx}" if len(baselines[name]) <= 1 else None)
    if len(baselines[name]) > 1:
        m, s = mean_std_band(baselines[name])
        ax.plot(gens, m, color=colors_b[idx], lw=2.0, label=f"{name} mean")
    # Plot LLM mean with band
    ax.plot(gens, llm_mean, color="black", lw=1.8, label=f"LLM mean ({len(llm_seeds)} seeds)")
    ax.fill_between(gens, [a - b for a, b in zip(llm_mean, llm_std)], [a + b for a, b in zip(llm_mean, llm_std)], color="black", alpha=0.15)
    # LLM individual seeds (light)
    for s_idx, traj in enumerate(llm_seeds):
        ax.plot(gens, traj, color="gray", lw=0.6, alpha=0.4)
    ax.set_title(name)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    if idx % 4 == 0:
        ax.set_ylabel("Cooperation rate")
    if idx // 4 == 1:
        ax.set_xlabel("Generation")
    ax.legend(loc="lower right", fontsize=7)
fig.suptitle("Per-baseline comparison: 8 leading-eight (Ohtsuki-Iwasa 2006) vs LLM-evolved (3 seeds each, 30 generations)", fontsize=12)
fig.tight_layout()
fig.savefig(PLOTS / "per_baseline.png", dpi=120)
fig.savefig(PLOTS / "per_baseline.pdf")
plt.close(fig)
print(f"  -> {PLOTS / 'per_baseline.png'}")

# ---- Plot 3: LLM only, individual seeds ----
fig, ax = plt.subplots(figsize=(10, 6))
for s_idx, traj in enumerate(llm_seeds):
    ax.plot(gens, traj, marker="o", ms=4, lw=1.5, label=f"LLM seed{s_idx} (final={traj[-1]:.2f})")
ax.plot(gens, llm_mean, color="black", lw=2.5, ls="--", label=f"LLM mean")
ax.fill_between(gens, [a - b for a, b in zip(llm_mean, llm_std)], [a + b for a, b in zip(llm_mean, llm_std)], color="black", alpha=0.15, label="LLM ±1 std")
ax.axhline(1.0, color="green", ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Generation")
ax.set_ylabel("Mean cooperation rate")
ax.set_title("LLM-evolved strategies: 3 seeds, 30 generations\n(shows high seed-to-seed variance: 1.0 vs 0.0 vs 0.0 final coop)")
ax.set_ylim(-0.05, 1.1)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "llm_only.png", dpi=120)
fig.savefig(PLOTS / "llm_only.pdf")
plt.close(fig)
print(f"  -> {PLOTS / 'llm_only.png'}")

print("\nAll plots generated.")
