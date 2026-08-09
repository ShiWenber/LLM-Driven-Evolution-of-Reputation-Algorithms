"""Quick aggregation: Ablation A v1 (adversarial + canonical hints) vs NEUTRAL main."""
import json
import numpy as np
from pathlib import Path

QB = Path(r'C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline')

print('=== Ablation A v1 (adversarial + canonical hints) / 3 seeds / 100 gen ===')
v1_coops = []
v1_fits = []
v1_g0s = []
for s in [0, 1, 2]:
    p = QB / f'LLM_v3_g100_1000inter_ADVERSARIAL_seed{s}' / 'evolutionary.json'
    d = json.loads(p.read_text())
    final = d['trajectory'][-1]
    v1_coops.append(final['cooperation_rate_mean'])
    v1_fits.append(final['fitness_mean'])
    v1_g0s.append(d['trajectory'][0]['cooperation_rate_mean'])
    print(f"  seed {s}: gen0={v1_g0s[-1]:.3f}, "
          f"final={v1_coops[-1]:.3f}, fitness={v1_fits[-1]:.1f}, "
          f"FALLBACK init/mut={d['config']['fallback_init_count']}/{d['config']['fallback_mutation_count']}")

print()
print(f"  v1 ADV    final coop: {v1_coops}")
print(f"  v1 ADV    mean +/- std: {np.mean(v1_coops):.3f} +/- {np.std(v1_coops):.3f}")
print(f"  v1 ADV    gen0:        mean={np.mean(v1_g0s):.3f}")
print()
print('=== Compare to NEUTRAL main run ===')
n_coops = []
n_fits = []
for s in [0, 1, 2]:
    p = QB / f'LLM_v3_g100_1000inter_seed{s}' / 'evolutionary.json'
    d = json.loads(p.read_text())
    final = d['trajectory'][-1]
    n_coops.append(final['cooperation_rate_mean'])
    n_fits.append(final['fitness_mean'])
    print(f"  seed {s}: final coop={n_coops[-1]:.3f}, fitness={n_fits[-1]:.1f}")

print()
print(f"  NEUTRAL   final coop: {n_coops}")
print(f"  NEUTRAL   mean +/- std: {np.mean(n_coops):.3f} +/- {np.std(n_coops):.3f}")
print()
print("=== Headline ===")
print(f"  v1 ADV:  mean coop = {np.mean(v1_coops):.3f} +/- {np.std(v1_coops):.3f}, fitness = {np.mean(v1_fits):.1f}")
print(f"  NEUTRAL: mean coop = {np.mean(n_coops):.3f} +/- {np.std(n_coops):.3f}, fitness = {np.mean(n_fits):.1f}")
print(f"  delta coop (v1 - neutral): {np.mean(v1_coops) - np.mean(n_coops):+.3f}")
print()
print("  basin breakdown:")
for label, arr in [("v1 ADV", v1_coops), ("NEUTRAL", n_coops)]:
    high = sum(1 for x in arr if x >= 0.7)
    mid  = sum(1 for x in arr if 0.3 <= x < 0.7)
    low  = sum(1 for x in arr if x < 0.3)
    print(f"    {label}: high(>=0.7)={high}, mid={mid}, low(<0.3)={low}")
