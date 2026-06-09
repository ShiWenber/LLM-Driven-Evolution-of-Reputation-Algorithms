"""Side-by-side comparison of deepseek-v4-flash and deepseek-coder at the
observability levels both experiments tested: private, partial_0.3, full.

Standard plan (deepseek-v4-flash) covers all 4 levels of Exp 1 + 6 of Exp 2
threshold scan. Robustness (deepseek-coder) only covers partial_0.3 and
partial_0.7. We compare what we can.
"""
import json
import re
from pathlib import Path
from collections import defaultdict

FLASH = Path('results/exp1_method')  # 12 trials
ROBUST = Path('results/exp5_robustness')  # 6 trials


def load_trials(root):
    rows = []
    for trial_dir in sorted(root.iterdir()):
        if not trial_dir.is_dir():
            continue
        m = re.match(r'([a-z_0-9.]+)_seed(\d+)', trial_dir.name)
        if not m:
            continue
        obs, seed = m.group(1), int(m.group(2))
        agg = list(trial_dir.glob('evolutionary_*.json'))
        if not agg:
            continue
        d = json.loads(agg[0].read_text())
        ts = d.get('trials_summary', [{}])[0]
        final = ts.get('final_mean_cooperation')
        traj = ts.get('trajectory', [])
        gen0 = traj[0]['cooperation_rate_mean'] if traj else None
        rows.append({'obs': obs, 'seed': seed, 'final_coop': final, 'gen0_coop': gen0})
    return rows


flash = load_trials(FLASH)
coder = load_trials(ROBUST)


def obs_to_p(obs):
    if obs == 'private': return 0.0
    if obs == 'full': return 1.0
    m = re.match(r'partial[_ ]?([0-9.]+)', obs)
    return float(m.group(1)) if m else None


def group(rows):
    by = defaultdict(list)
    for r in rows:
        p = obs_to_p(r['obs'])
        if p is not None:
            by[p].append(r)
    return by


fg = group(flash)
cg = group(coder)

print("=" * 76)
print("EXP 1 / Exp 5: deepseek-v4-flash  vs  deepseek-coder")
print("Final-generation mean cooperation rate by observability p")
print("=" * 76)
print()
print(f"{'p':<6} {'flash mean±std':<22} {'flash range':<22} {'coder mean±std':<22} {'coder range':<22}")
print("-" * 96)
all_p = sorted(set(fg.keys()) | set(cg.keys()))
import statistics
for p in all_p:
    f = fg.get(p, [])
    c = cg.get(p, [])
    f_mean = f"{statistics.mean([r['final_coop'] for r in f]):.3f}±{statistics.stdev([r['final_coop'] for r in f]):.3f}" if len(f) > 1 else (f"{f[0]['final_coop']:.3f}" if f else "—")
    c_mean = f"{statistics.mean([r['final_coop'] for r in c]):.3f}±{statistics.stdev([r['final_coop'] for r in c]):.3f}" if len(c) > 1 else (f"{c[0]['final_coop']:.3f}" if c else "—")
    f_rng = f"[{min(r['final_coop'] for r in f):.3f}, {max(r['final_coop'] for r in f):.3f}]" if f else "—"
    c_rng = f"[{min(r['final_coop'] for r in c):.3f}, {max(r['final_coop'] for r in c):.3f}]" if c else "—"
    n_f = len(f)
    n_c = len(c)
    print(f"{p:<6} {f_mean + f' (n={n_f})':<22} {f_rng:<22} {c_mean + f' (n={n_c})':<22} {c_rng:<22}")

print()
print("Direct comparison at p=0.3 and p=0.7 (both experiments tested these):")
print("-" * 80)
for p in (0.3, 0.7):
    f = fg.get(p, [])
    c = cg.get(p, [])
    if f and c:
        fv = [r['final_coop'] for r in f]
        cv = [r['final_coop'] for r in c]
        delta = statistics.mean(cv) - statistics.mean(fv)
        print(f"  p={p}: flash mean {statistics.mean(fv):.3f}  coder mean {statistics.mean(cv):.3f}  delta {delta:+.3f}")

print()
print("=" * 76)
print("Same comparison: Gen-0 (initial population) cooperation rate")
print("=" * 76)
print()
print(f"{'p':<6} {'flash gen0 mean±std':<24} {'coder gen0 mean±std':<24}")
print("-" * 60)
for p in all_p:
    f = fg.get(p, [])
    c = cg.get(p, [])
    f0 = [r['gen0_coop'] for r in f if r['gen0_coop'] is not None]
    c0 = [r['gen0_coop'] for r in c if r['gen0_coop'] is not None]
    f0s = f"{statistics.mean(f0):.3f}±{statistics.stdev(f0):.3f}" if len(f0) > 1 else (f"{f0[0]:.3f}" if f0 else "—")
    c0s = f"{statistics.mean(c0):.3f}±{statistics.stdev(c0):.3f}" if len(c0) > 1 else (f"{c0[0]:.3f}" if c0 else "—")
    print(f"{p:<6} {f0s:<24} {c0s:<24}")

print()
print("=" * 76)
print("Test: are the gen-0 initial populations similar?")
print("(if yes: both models produce similar starting points; difference is in mutation)")
print("=" * 76)
for p in (0.3, 0.7):
    f0 = [r['gen0_coop'] for r in fg.get(p, []) if r['gen0_coop'] is not None]
    c0 = [r['gen0_coop'] for r in cg.get(p, []) if r['gen0_coop'] is not None]
    if f0 and c0:
        fmean, cmean = statistics.mean(f0), statistics.mean(c0)
        print(f"  p={p}: flash gen0 mean {fmean:.3f}  coder gen0 mean {cmean:.3f}  diff {cmean-fmean:+.3f}")
