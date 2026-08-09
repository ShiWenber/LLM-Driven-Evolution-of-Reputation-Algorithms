"""Three-way trajectory comparison: M3 30gen (FALLBACK init) vs M4 30gen smoke
(FALLBACK init) vs M4 100gen thinking=off (real LLM init)."""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(
    r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline"
)
PLOTS = ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def load_traj(name):
    f = ROOT / name / "evolutionary.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text(encoding="utf-8"))
    return [(int(t["generation"]), float(t["cooperation_rate_mean"]), float(t["fitness_mean"])) for t in j["trajectory"]]


# Three runs
runs = {
    "M3 v3 30gen (FALLBACK init)": load_traj("LLM_evolution_seed0"),
    "M4 v3 30gen smoke (FALLBACK init)": load_traj("LLM_v3_g30_smoke_seed0"),
    "M4 v3 100gen thinking-off (real LLM init)": load_traj("LLM_v3_g100_thinking_off_seed0"),
}

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)

colors = {
    "M3 v3 30gen (FALLBACK init)": "#1f77b4",
    "M4 v3 30gen smoke (FALLBACK init)": "#ff7f0e",
    "M4 v3 100gen thinking-off (real LLM init)": "#d62728",
}
markers = {
    "M3 v3 30gen (FALLBACK init)": "o",
    "M4 v3 30gen smoke (FALLBACK init)": "s",
    "M4 v3 100gen thinking-off (real LLM init)": "^",
}
lws = {
    "M3 v3 30gen (FALLBACK init)": 1.4,
    "M4 v3 30gen smoke (FALLBACK init)": 1.4,
    "M4 v3 100gen thinking-off (real LLM init)": 2.0,
}
labelsizes = {
    "M3 v3 30gen (FALLBACK init)": 5,
    "M4 v3 30gen smoke (FALLBACK init)": 5,
    "M4 v3 100gen thinking-off (real LLM init)": 4,
}

# Top panel: cooperation rate, gen 0-30 (all three runs share x-axis)
ax = axes[0]
for name, traj in runs.items():
    if traj is None:
        continue
    arr = np.array(traj)
    gens, coop, fit = arr[:, 0], arr[:, 1], arr[:, 2]
    mask = gens <= 30
    ax.plot(
        gens[mask],
        coop[mask],
        color=colors[name],
        marker=markers[name],
        markersize=labelsizes[name],
        linewidth=lws[name],
        label=f"{name} (final={coop[-1]:.3f})",
        alpha=0.85,
    )
ax.axhspan(0.9, 1.0, alpha=0.10, color="green")
ax.set_ylabel("Mean cooperation rate", fontsize=11)
ax.set_xlabel("Generation", fontsize=11)
ax.set_title("Gen 0-30: All three LLM runs (init phase)", fontsize=12)
ax.set_ylim(-0.05, 1.10)
ax.set_xticks(range(0, 31, 5))
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)

# Bottom panel: 100 gen trajectory, with phases annotated
ax = axes[1]
traj = runs["M4 v3 100gen thinking-off (real LLM init)"]
arr = np.array(traj)
gens, coop, fit = arr[:, 0], arr[:, 1], arr[:, 2]
ax.plot(
    gens,
    coop,
    color=colors["M4 v3 100gen thinking-off (real LLM init)"],
    marker="^",
    markersize=4,
    linewidth=2.0,
    label="100gen thinking-off (real LLM init)",
    alpha=0.85,
)
# Phase shading
ax.axvspan(0, 30, alpha=0.10, color="green", label="high-coop init (gen 0-29)")
ax.axvspan(30, 43, alpha=0.10, color="orange", label="regime collapse (gen 30-43)")
ax.axvspan(43, 58, alpha=0.10, color="red", label="defection basin (gen 43-58)")
ax.axvspan(58, 100, alpha=0.12, color="blue", label="multi-basin recovery (gen 58-99)")
# Add fitness on right axis
ax2 = ax.twinx()
ax2.plot(
    gens,
    fit,
    color="#2ca02c",
    linewidth=1.0,
    alpha=0.5,
    linestyle="--",
    label="fitness",
)
ax.set_xlabel("Generation", fontsize=11)
ax.set_ylabel("Mean cooperation rate", fontsize=11, color="black")
ax2.set_ylabel("Mean fitness (proxy for sum of payoffs)", fontsize=11, color="green")
ax.set_title("100gen thinking-off: full trajectory (real LLM init)", fontsize=12)
ax.set_ylim(-0.05, 1.10)
ax2.set_ylim(-5, 35)
ax.set_xticks(range(0, 101, 10))
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
ax2.legend(loc="lower right", fontsize=9)

plt.suptitle(
    "Three-way LLM trajectory comparison: 30gen (FALLBACK init) vs 30gen smoke (FALLBACK init) vs 100gen (real LLM init)\n"
    "Note: M3 / 30gen smoke both FALLBACK init -> single-basin convergence (~1.0). 100gen real LLM init -> multi-basin dynamics.",
    fontsize=12,
    y=1.00,
)
plt.tight_layout()

out_pdf = PLOTS / "v3_three_way_traj.pdf"
out_png = PLOTS / "v3_three_way_traj.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=180)
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
print()
print("Final coop:")
for name, traj in runs.items():
    if traj is None:
        print(f"  {name}: MISSING")
    else:
        print(f"  {name}: coop={traj[-1][1]:.3f}, fitness={traj[-1][2]:.2f}, gens={len(traj)}")
