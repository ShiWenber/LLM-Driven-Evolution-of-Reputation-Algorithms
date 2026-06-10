"""Drift analysis: gen-0 cooperation rates from LLM-evo trajectories.

The LLM-driven evolutionary run records a trajectory of mean
cooperation per generation. Gen 0 is the LLM-generated initial
population (no selection, no mutation) -- directly comparable to
the static control's gen-0.

Question: does the LLM's gen-0 initial population look the same
across observability conditions, and does it match the static
control's gen-0?

Hypothesis (from v9 paper finding): the LLM's gen-0 is approximately
constant across observability levels (like static). If so, the
dramatic difference at gen-10 is attributable to selection + LLM
mutation, not to the initial population.
"""
import json
import re
from pathlib import Path
from collections import defaultdict
import statistics

EXP1 = Path('results/exp1_method')
EXP3 = Path('results/exp3_static_g10_n10')  # static n=10 G=10


def load_trajectories(root):
    out = defaultdict(list)  # obs -> list of trajectories
    for trial_dir in sorted(root.iterdir()):
        if not trial_dir.is_dir(): continue
        m = re.match(r'([a-z_0-9.]+)_seed\d+', trial_dir.name)
        if not m: continue
        obs = m.group(1)
        # Find the aggregate file
        agg = list(trial_dir.glob('evolutionary_*.json')) + list(trial_dir.glob('static_control_*.json'))
        if not agg: continue
        d = json.loads(agg[0].read_text())
        ts = d.get('trials_summary', [{}])[0]
        traj = ts.get('trajectory', [])
        if traj:
            out[obs].append(traj)
    return out


def gen0_means(root):
    """Return dict[obs] -> list of gen-0 cooperation rates."""
    out = defaultdict(list)
    trajs = load_trajectories(root)
    for obs, trials in trajs.items():
        for traj in trials:
            if traj and 'cooperation_rate_mean' in traj[0]:
                out[obs].append(traj[0]['cooperation_rate_mean'])
    return out


# LLM-evo (n=3, exp1)
llm_gen0 = gen0_means(EXP1)
# Static (n=10, exp3 G=10)
static_gen0 = gen0_means(EXP3)


def report(name, gen0, n_per_obs):
    print(f"\n=== {name} (n_per_obs) ===")
    print(f"{'obs':<14} {'mean':>8} {'std':>8} {'range':>22} {'n':>4}")
    for obs in sorted(gen0):
        vals = gen0[obs]
        if not vals: continue
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rng = f"[{min(vals):.3f}, {max(vals):.3f}]"
        n = len(vals)
        print(f"  {obs:<14} {m:8.3f} {s:8.3f} {rng:>22} {n:>4}")


report("LLM-evo gen-0 (Exp 1, n=3)", llm_gen0, 3)
report("Static gen-0 (Exp 3, n=10)", static_gen0, 10)


# Direct comparison at common obs levels
print("\n=== Side-by-side: LLM-evo gen-0 vs Static gen-0 (both at gen 0, before any selection) ===")
print(f"{'obs':<14} {'LLM gen-0 mean':>18} {'Static gen-0 mean':>20} {'Δ':>8}")
for obs in sorted(set(llm_gen0) & set(static_gen0)):
    if llm_gen0[obs] and static_gen0[obs]:
        lm = statistics.mean(llm_gen0[obs])
        sm = statistics.mean(static_gen0[obs])
        print(f"  {obs:<14} {lm:18.3f} {sm:20.3f} {sm - lm:+8.3f}")


# Per-gen-9 / gen-10 comparison
def final_means(root):
    out = defaultdict(list)
    trajs = load_trajectories(root)
    for obs, trials in trajs.items():
        for traj in trials:
            if traj and 'cooperation_rate_mean' in traj[-1]:
                out[obs].append(traj[-1]['cooperation_rate_mean'])
    return out


llm_final = final_means(EXP1)
static_final = final_means(EXP3)
print("\n=== Side-by-side: LLM-evo gen-10 (n=3) vs Static gen-10 (n=10) ===")
print(f"{'obs':<14} {'LLM gen-10 mean':>18} {'Static gen-10 mean':>20} {'Δ':>8}")
for obs in sorted(set(llm_final) & set(static_final)):
    if llm_final[obs] and static_final[obs]:
        lm = statistics.mean(llm_final[obs])
        sm = statistics.mean(static_final[obs])
        print(f"  {obs:<14} {lm:18.3f} {sm:20.3f} {sm - lm:+8.3f}")


# Drift analysis: gen-0 -> gen-9 mean within each condition
print("\n=== Drift (gen-0 -> gen-9 mean change) ===")
def drift(trajs):
    out = defaultdict(list)
    for obs, trials in trajs.items():
        for traj in trials:
            if traj and 'cooperation_rate_mean' in traj[0] and 'cooperation_rate_mean' in traj[-1]:
                drift_val = traj[-1]['cooperation_rate_mean'] - traj[0]['cooperation_rate_mean']
                out[obs].append(drift_val)
    return out

llm_drift = drift(load_trajectories(EXP1))
static_drift = drift(load_trajectories(EXP3))
print(f"{'obs':<14} {'LLM-evo drift':>15} {'Static drift':>15}")
for obs in sorted(set(llm_drift) & set(static_drift)):
    if llm_drift[obs] and static_drift[obs]:
        lm = statistics.mean(llm_drift[obs])
        sm = statistics.mean(static_drift[obs])
        print(f"  {obs:<14} {lm:+15.3f} {sm:+15.3f}")
