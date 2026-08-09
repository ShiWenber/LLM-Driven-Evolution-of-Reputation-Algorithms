"""Aggregate v2 final results + compare to v1."""
import json
import statistics
from pathlib import Path

base = Path(r'results/quantitative_baseline')

print("=== v2 (loosened prompt) per-seed ===")
final_coops = []
final_fits = []
for s in [0, 1, 2]:
    p = base / f'LLM_v3_fermi_z_v2_g100_1000inter_seed{s}' / 'evolutionary.json'
    d = json.loads(p.read_text(encoding='utf-8'))
    t = d['trajectory']
    final_coop = t[-1]['cooperation_rate_mean']
    final_fit = t[-1]['fitness_mean']
    gen0_coop = t[0]['cooperation_rate_mean']
    final_coops.append(final_coop)
    final_fits.append(final_fit)
    print(f"  seed{s}: gen0={gen0_coop:.3f}, final_coop={final_coop:.3f}, final_fit={final_fit:.2f}")

print()
print(f"=== v2 SUMMARY (loosened prompt) ===")
print(f"final coop: {[f'{x:.3f}' for x in final_coops]}")
print(f"mean: {statistics.mean(final_coops):.3f}, std: {statistics.stdev(final_coops):.3f}")
print(f"fitness mean: {statistics.mean(final_fits):.2f} ± {statistics.stdev(final_fits):.2f}")
print()
print("=== v1 vs v2 comparison ===")
print(f"v1 (old prompt, over-constrained):  mean 0.892 ± 0.098  (finals 0.981/0.941/0.755)")
print(f"v2 (loosened, M9):                  mean {statistics.mean(final_coops):.3f} ± {statistics.stdev(final_coops):.3f}  (finals {final_coops[0]:.3f}/{final_coops[1]:.3f}/{final_coops[2]:.3f})")
print()
print("delta:", round(statistics.mean([0.981, 0.941, 0.755]) - statistics.mean(final_coops), 3))
