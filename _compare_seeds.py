"""Compare seed 0 vs seed 2 trajectories to understand basin divergence.

Look for:
- First crossing below 0.5 in seed 0
- Mutated strategy that triggered the collapse
- What seed 2 did differently in the same gens
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
QB = ROOT / 'results' / 'quantitative_baseline'

seeds = {}
for s in [0, 1, 2]:
    p = QB / f'LLM_v3_g100_1000inter_seed{s}' / 'evolutionary.json'
    if p.exists():
        seeds[s] = json.loads(p.read_text())

# Find first gen where coop < 0.5
print("=== First crossing below 0.5 ===")
for s, d in seeds.items():
    for t in d['trajectory']:
        if t['cooperation_rate_mean'] < 0.5:
            print(f"  seed {s}: first dipped <0.5 at gen {t['generation']} (coop={t['cooperation_rate_mean']:.3f})")
            break
    else:
        print(f"  seed {s}: never went below 0.5")

# Show per-gen snapshots around the collapse point for seed 0
print("\n=== Seed 0 trajectory (collapse focus) ===")
print(f"{'gen':>3} | {'coop':>6} | {'fit':>6} | n_mut_fail")
for t in seeds[0]['trajectory']:
    gen = t['generation']
    coop = t['cooperation_rate_mean']
    fit = t['fitness_mean']
    n_fail = sum(1 for p in t['population'] if p['cooperation_rate'] < 0.3)
    print(f"{gen:>3} | {coop:>6.3f} | {fit:>6.1f} | {n_fail}/15 < 0.3")

# Compare snapshot at gen 30 (still in early phase) for all 3 seeds
print("\n=== Gen 30 snapshot comparison ===")
for s, d in seeds.items():
    t = d['trajectory'][30]
    n_low = sum(1 for p in t['population'] if p['cooperation_rate'] < 0.3)
    n_high = sum(1 for p in t['population'] if p['cooperation_rate'] > 0.7)
    print(f"  seed {s}: coop={t['cooperation_rate_mean']:.3f}, "
          f"<0.3 = {n_low}/15, >0.7 = {n_high}/15")

# Look at the strategy "classes" of the 15 agents in each seed at gen 30
# Use the cooperation rate per agent to identify strategy types
def classify_strategy(agent, threshold=0.3, high=0.7):
    coop = agent['cooperation_rate']
    if coop < threshold:
        return 'defector'
    elif coop > high:
        return 'cooperator'
    else:
        return 'mixed'

print("\n=== Strategy class distribution at gen 30 ===")
for s, d in seeds.items():
    t = d['trajectory'][30]
    classes = [classify_strategy(p) for p in t['population']]
    n_def = classes.count('defector')
    n_coop = classes.count('cooperator')
    n_mix = classes.count('mixed')
    print(f"  seed {s}: defector={n_def}, mixed={n_mix}, cooperator={n_coop}")

print("\n=== Strategy class distribution at gen 99 (final) ===")
for s, d in seeds.items():
    t = d['trajectory'][-1]
    classes = [classify_strategy(p) for p in t['population']]
    n_def = classes.count('defector')
    n_coop = classes.count('cooperator')
    n_mix = classes.count('mixed')
    print(f"  seed {s}: defector={n_def}, mixed={n_mix}, cooperator={n_coop}")

# Find the gen where seed 0 first had >= 7 defectors (defection basin)
print("\n=== First time >= 7/15 defectors (basin absorbed) ===")
for s, d in seeds.items():
    for t in d['trajectory']:
        n_def = sum(1 for p in t['population'] if p['cooperation_rate'] < 0.3)
        if n_def >= 7:
            print(f"  seed {s}: gen {t['generation']} (n_def={n_def}/15, coop={t['cooperation_rate_mean']:.3f})")
            break
    else:
        print(f"  seed {s}: never reached 7/15 defectors")

# Compare trajectory "valleys" - find local minima in cooperation
print("\n=== Cooperation trajectory (per gen) — all 3 seeds ===")
print(f"{'gen':>3} | {'seed0':>7} | {'seed1':>7} | {'seed2':>7} | {'mean':>7}")
gens0 = [t['cooperation_rate_mean'] for t in seeds[0]['trajectory']]
gens1 = [t['cooperation_rate_mean'] for t in seeds[1]['trajectory']]
gens2 = [t['cooperation_rate_mean'] for t in seeds[2]['trajectory']]
for i in range(0, 100, 5):
    print(f"{i:>3} | {gens0[i]:>7.3f} | {gens1[i]:>7.3f} | {gens2[i]:>7.3f} | "
          f"{(gens0[i]+gens1[i]+gens2[i])/3:>7.3f}")
