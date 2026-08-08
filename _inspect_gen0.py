"""Inspect gen0 init: per-agent cooperation rate distribution."""
import json

for s in [0, 1, 2]:
    d = json.load(open(f'results/quantitative_baseline/LLM_v3_fermi_z_g100_1000inter_seed{s}/evolutionary.json'))
    g0 = d['trajectory'][0]
    pop = g0['population']
    n_int = g0['n_interactions']
    print(f'\n=== seed{s} gen0 ===')
    print(f'  global coop_rate = {g0["cooperation_rate_mean"]:.4f}, n_interactions = {n_int}')
    rates = [a.get('cooperation_rate', 0) for a in pop]
    fits = [a.get('fitness', 0) for a in pop]
    print(f'  per-agent coop rate: min={min(rates):.3f}, max={max(rates):.3f}, mean={sum(rates)/len(rates):.3f}')
    print(f'  per-agent fitness:   min={min(fits):.1f}, max={max(fits):.1f}, mean={sum(fits)/len(fits):.1f}')
    # classify
    high = sum(1 for r in rates if r >= 0.95)
    mid = sum(1 for r in rates if 0.5 <= r < 0.95)
    low = sum(1 for r in rates if r < 0.5)
    print(f'  distribution: high(>=0.95)={high}, mid(0.5-0.95)={mid}, low(<0.5)={low}')
    # also check legacy v3 g100 1000inter for comparison
    d2 = json.load(open(f'results/quantitative_baseline/LLM_v3_g100_1000inter_seed{s}/evolutionary.json'))
    g02 = d2['trajectory'][0]
    print(f'  [LEGACY v3 g100 seed{s} gen0: coop={g02["cooperation_rate_mean"]:.4f}]')
