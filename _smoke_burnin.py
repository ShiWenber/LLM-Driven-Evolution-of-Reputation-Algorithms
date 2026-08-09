"""Smoke test: verify burn-in fitness window works.

Tests:
  1. ALLC under windowed fitness: with everyone cooperating, every
     agent's per-interaction payoff is +1, so windowed fitness per
     agent = 200 (last 200 interactions all give +1 each).
  2. ALLD under windowed fitness: per-interaction payoff = 0, so
     windowed fitness = 0.
  3. IS (leading-eight baseline): some cooperators / some defectors,
     should produce non-trivial windowed fitness.
  4. Windowed vs cumulative sanity: with target=1000 and window=200,
     cumulative fitness should be ~5x the windowed fitness if
     behavior is roughly stationary.

All runs are 3 generations at G=5 to keep wall time minimal.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r'C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation')

from experiments.v2_quantitative.population import V2EvolutionaryPopulation


def run_smoke(baseline: str, window: int = 200) -> dict:
    pop = V2EvolutionaryPopulation(
        population_size=15,
        num_rounds_per_gen=30,
        target_interactions_per_gen=1000,
        benefit=2.0,
        cost=1.0,
        observability='full',
        observability_p=1.0,
        elite_count=2,
        num_eliminate=5,
        tournament_size=3,
        seed=42,
        results_dir='results/_smoke_burnin',
        use_baseline=baseline,
        agent_type='v2',
        fitness_window_interactions=window,
    )
    result = pop.run_evolution(num_generations=3)
    final = result['trajectory'][-1]
    return {
        'baseline': baseline,
        'window': window,
        'gen0_coop': result['trajectory'][0]['cooperation_rate_mean'],
        'gen0_fit_cum': sum(result['trajectory'][0]['population'][i]['fitness'] for i in range(15)) / 15,
        'gen0_n_inter': result['trajectory'][0]['n_interactions'],
        'gen2_coop': final['cooperation_rate_mean'],
        'gen2_fit_mean': final['fitness_mean'],
        'gen2_fitness_per_agent': [a['fitness'] for a in final['population']],
    }


for baseline in ['ALLC', 'ALLD', 'IS', 'IS+', 'SS']:
    r = run_smoke(baseline, window=200)
    print(f"\n[{baseline}] (window=200)")
    print(f"  gen 0: coop={r['gen0_coop']:.3f}  "
          f"fitness_mean={r['gen0_fit_cum']:.2f}  "
          f"n_inter={r['gen0_n_inter']}")
    print(f"  gen 2: coop={r['gen2_coop']:.3f}  "
          f"fitness_mean={r['gen2_fit_mean']:.2f}")
    fits = r['gen2_fitness_per_agent']
    print(f"  fitness (per agent, gen 2): "
          f"min={min(fits):.2f}  max={max(fits):.2f}  mean={sum(fits)/len(fits):.2f}")

print("\n--- Legacy mode (window=None, use ALL interactions) ---")
for baseline in ['ALLC', 'ALLD']:
    r = run_smoke(baseline, window=None)
    print(f"\n[{baseline}] (window=None)")
    print(f"  gen 2 fitness_mean={r['gen2_fit_mean']:.2f}  (should be ~5x the 200-window value)")
