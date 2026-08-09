"""Compare G=10 (v15 main plan) vs G=30 (exp12) trajectories per (obs, seed).

For each (obs, seed) shared between the two plans, compute:
- final coop
- time to first drop below 0.05 (collapse time)
- max coop achieved across the run
- area under the curve (cumulative cooperation)

Then aggregate per obs: mean, std, and seed-by-seed delta (G=30 - G=10).
"""
import json, statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")

G10_DIR = ROOT / "results" / "exp1_method"  # v15 main plan: N=15, 4 obs, 3 seeds, G=10
G30_DIR = ROOT / "results" / "exp12_g30_n15"  # exp12: N=15, 4 obs, 3 seeds, G=30

OBS = ["private", "partial_0.3", "partial_0.7", "full"]
SEEDS = [0, 1, 2]


def load_trial(d: Path):
    """Load trajectory from a trial dir, looking for evolutionary.json or evo_*.json."""
    if not d.exists():
        return None
    for fname in ("evolutionary.json",):
        f = d / fname
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                traj = data.get("trajectory", [])
                if traj:
                    return traj
            except Exception:
                pass
    # Fall back: scan for evo_*.json
    for f in d.glob("evo_*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            traj = data.get("trajectory", [])
            if traj:
                return traj
        except Exception:
            pass
    return None


def collapse_time(traj, threshold=0.05):
    """First generation where coop drops below threshold for the rest of the run."""
    below_from = None
    for i, g in enumerate(traj):
        c = g.get("cooperation_rate_mean", 0)
        if c < threshold:
            below_from = i
            break
    if below_from is None:
        return None
    # Verify it stays below
    for g in traj[below_from:]:
        if g.get("cooperation_rate_mean", 0) >= threshold:
            return None  # recovered
    return below_from


def auc(traj):
    """Sum of cooperation rates across all generations (proxy for cumulative cooperation)."""
    return sum(g.get("cooperation_rate_mean", 0) for g in traj)


def max_coop(traj):
    return max(g.get("cooperation_rate_mean", 0) for g in traj)


def final_coop(traj):
    return traj[-1].get("cooperation_rate_mean", 0) if traj else None


def gen_count(traj):
    return len(traj)


# Per (obs, seed) comparison
print(f"{'obs':<12} {'seed':<5} {'G10_final':<10} {'G30_final':<10} {'delta':<8} {'G10_auc':<10} {'G30_auc':<10} {'G10_drop':<10} {'G30_drop':<10}")
print("-" * 100)

per_obs_g10 = defaultdict(list)
per_obs_g30 = defaultdict(list)
per_obs_drop_g10 = defaultdict(list)
per_obs_drop_g30 = defaultdict(list)

for obs in OBS:
    for seed in SEEDS:
        g10 = load_trial(G10_DIR / f"{obs}_seed{seed}")
        g30 = load_trial(G30_DIR / f"{obs}_seed{seed}")
        f10 = final_coop(g10) if g10 else None
        f30 = final_coop(g30) if g30 else None
        delta = (f30 - f10) if (f10 is not None and f30 is not None) else None
        a10 = auc(g10) if g10 else None
        a30 = auc(g30) if g30 else None
        d10 = collapse_time(g10) if g10 else None
        d30 = collapse_time(g30) if g30 else None
        print(f"{obs:<12} {seed:<5} {f10 if f10 is not None else 'MISS':<10} {f30 if f30 is not None else 'MISS':<10} "
              f"{delta if delta is not None else '-':<8.3f} {a10 if a10 is not None else '-':<10.2f} {a30 if a30 is not None else '-':<10.2f} "
              f"{d10 if d10 is not None else '-':<10} {d30 if d30 is not None else '-':<10}")
        if f10 is not None:
            per_obs_g10[obs].append(f10)
            per_obs_drop_g10[obs].append(d10)
        if f30 is not None:
            per_obs_g30[obs].append(f30)
            per_obs_drop_g30[obs].append(d30)

# Per-obs aggregates
print()
print("=== Per-obs aggregate (mean ± std) ===")
print(f"{'obs':<12} {'G10_mean':<10} {'G10_std':<10} {'G30_mean':<10} {'G30_std':<10} {'delta':<10} {'G10_drop':<10} {'G30_drop':<10}")
print("-" * 90)
for obs in OBS:
    g10 = per_obs_g10[obs]
    g30 = per_obs_g30[obs]
    d10 = per_obs_drop_g10[obs]
    d30 = per_obs_drop_g30[obs]
    if g10 and g30:
        m10, s10 = statistics.mean(g10), (statistics.stdev(g10) if len(g10) > 1 else 0)
        m30, s30 = statistics.mean(g30), (statistics.stdev(g30) if len(g30) > 1 else 0)
        delta = m30 - m10
        dm10 = statistics.mean([d for d in d10 if d is not None]) if any(d is not None for d in d10) else None
        dm30 = statistics.mean([d for d in d30 if d is not None]) if any(d is not None for d in d30) else None
        print(f"{obs:<12} {m10:<10.3f} {s10:<10.3f} {m30:<10.3f} {s30:<10.3f} {delta:<+10.3f} "
              f"{dm10 if dm10 is not None else 'N/A':<10} {dm30 if dm30 is not None else 'N/A':<10}")

# Save JSON
out = {
    "per_obs_g10": dict(per_obs_g10),
    "per_obs_g30": dict(per_obs_g30),
    "per_obs_drop_g10": dict(per_obs_drop_g10),
    "per_obs_drop_g30": dict(per_obs_drop_g30),
}
out_path = ROOT / "results" / "g10_vs_g30_comparison.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")
