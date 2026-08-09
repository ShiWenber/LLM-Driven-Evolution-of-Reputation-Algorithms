"""Plot G=10 vs G=30 trajectories: side-by-side per obs, mean ± std across 3 seeds.
Saves to results/figures/g10_vs_g30.png and .pdf
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
G10 = ROOT / "results" / "exp1_method"
G30 = ROOT / "results" / "exp12_g30_n15"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OBS = ["private", "partial_0.3", "partial_0.7", "full"]
SEEDS = [0, 1, 2]


def load_traj(d):
    for f in (d / "evolutionary.json",):
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                traj = data.get("trajectory", [])
                if traj:
                    return traj
            except Exception:
                pass
    for f in d.glob("evo_*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            traj = data.get("trajectory", [])
            if traj:
                return traj
        except Exception:
            pass
    return None


fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, obs in zip(axes, OBS):
    # G=10
    g10_trajs = []
    for seed in SEEDS:
        traj = load_traj(G10 / f"{obs}_seed{seed}")
        if traj:
            g10_trajs.append([g.get("cooperation_rate_mean", 0) for g in traj])
    # G=30
    g30_trajs = []
    for seed in SEEDS:
        traj = load_traj(G30 / f"{obs}_seed{seed}")
        if traj:
            g30_trajs.append([g.get("cooperation_rate_mean", 0) for g in traj])

    # G=10 individual seeds + mean
    if g10_trajs:
        for t in g10_trajs:
            ax.plot(range(len(t)), t, color="C0", alpha=0.25, linewidth=0.8)
        max_g = max(len(t) for t in g10_trajs)
        if g10_trajs:
            # Pad to max length with last value
            padded = [t + [t[-1]] * (max_g - len(t)) for t in g10_trajs]
            mean = np.mean(padded, axis=0)
            std = np.std(padded, axis=0)
            ax.plot(range(max_g), mean, color="C0", linewidth=2, label="G=10")
            ax.fill_between(range(max_g), mean - std, mean + std, color="C0", alpha=0.15)
    # G=30 individual seeds + mean
    if g30_trajs:
        for t in g30_trajs:
            ax.plot(range(len(t)), t, color="C3", alpha=0.25, linewidth=0.8)
        max_g = max(len(t) for t in g30_trajs)
        padded = [t + [t[-1]] * (max_g - len(t)) for t in g30_trajs]
        mean = np.mean(padded, axis=0)
        std = np.std(padded, axis=0)
        ax.plot(range(max_g), mean, color="C3", linewidth=2, label="G=30")
        ax.fill_between(range(max_g), mean - std, mean + std, color="C3", alpha=0.15)

    ax.set_title(obs, fontsize=11)
    ax.set_xlabel("Generation")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)

axes[0].set_ylabel("Cooperation rate (mean across 15 agents)")
axes[0].legend(loc="upper right", fontsize=9)
fig.suptitle("G=10 (blue) vs G=30 (red) trajectories: 3 seeds each, N=15, DeepSeek-V4-Flash, b/c=2",
             fontsize=11)
fig.tight_layout()
png = OUT_DIR / "g10_vs_g30.png"
pdf = OUT_DIR / "g10_vs_g30.pdf"
fig.savefig(png, dpi=150, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"Saved {png}")
print(f"Saved {pdf}")
