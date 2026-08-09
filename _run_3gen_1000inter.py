"""Smoke: 3 gen × target_interactions_per_gen=1000 × seed=0.

Verifies the new target_interactions_per_gen param works and gives
us a real wall-time reading before committing to a full 100 gen run.

Estimate (from M4 100 gen thinking=off):
  - 5 LLM calls/gen × 3 gen = 15 mutation calls
  - 15 init calls
  - 30 LLM calls × ~11s = ~330s = 5.5 min LLM time
  - Game loop: 5x the 5-gen/210-games smoke (5 min) -> ~25 min
  - Total: ~25-30 min
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r'C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
from experiments.config.load_env import get_api_key, get_base_url
from experiments.v2_quantitative.population import V2EvolutionaryPopulation

ROOT = Path(r'C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
OUT = ROOT / 'results' / 'quantitative_baseline' / 'LLM_v3_g3_1000inter_smoke_seed0'
OUT.mkdir(parents=True, exist_ok=True)
out_path = OUT / 'evolutionary.json'
if out_path.exists():
    out_path.unlink()

api_key = get_api_key('deepseek')
base_url = get_base_url('deepseek')

print(f"=== 3 gen × 1000 inter smoke (Fix C/D/E + n_inter param) ===", flush=True)
print(f"  api_key: {api_key[:8]}...{api_key[-4:]}", flush=True)
print(f"  out: {out_path}", flush=True)

pop = V2EvolutionaryPopulation(
    population_size=15,
    target_interactions_per_gen=1000,  # NEW param
    benefit=2.0,
    cost=1.0,
    observability='full',
    observability_p=1.0,
    elite_count=2,
    num_eliminate=5,
    tournament_size=3,
    llm_model='deepseek-v4-flash',
    api_key=api_key,
    api_base_url=base_url,
    mutation_temperature=0.8,
    seed=0,
    results_dir=str(OUT.parent),
    use_baseline=None,
    agent_type='v3',
    llm_thinking=False,
)
print(f"  effective: rounds={pop.num_rounds_per_gen}, games/gen={pop.num_rounds_per_gen * 7}", flush=True)
print(f"  llm_thinking={pop.llm_thinking}, max_tokens={pop._llm_max_tokens}", flush=True)

t0 = time.time()
result = pop.run_evolution(num_generations=3)
elapsed = time.time() - t0
print(f"\n=== done in {elapsed:.1f}s ({elapsed/60:.1f} min) ===", flush=True)
print(f"  final coop = {result['trajectory'][-1]['cooperation_rate_mean']:.3f}", flush=True)
print(f"  final fitness_mean = {result['trajectory'][-1]['fitness_mean']:.1f}", flush=True)
print(f"  fallback_init_count = {result['config']['fallback_init_count']}", flush=True)
print(f"  fallback_mutation_count = {result['config']['fallback_mutation_count']}", flush=True)

# Per-gen breakdown
print(f"\n  Per-gen:", flush=True)
for t in result['trajectory']:
    print(f"    gen {t['generation']:2d}: coop={t['cooperation_rate_mean']:.3f}, "
          f"fitness={t['fitness_mean']:.1f}, n_inter={t['n_interactions']}", flush=True)

with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n  saved to {out_path}", flush=True)

# Project to 100 gen
proj = elapsed * 100 / 3
print(f"  [projection] 100 gen would take ~{proj/60:.0f} min ({proj/3600:.1f} h)", flush=True)
