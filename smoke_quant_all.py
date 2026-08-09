"""Smoke test all 8 baselines at G=10.
"""
import sys, time
from pathlib import Path
ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
sys.path.insert(0, str(ROOT))

from experiments.v2_quantitative.population import V2EvolutionaryPopulation
from experiments.v2_quantitative.baselines import BASELINES

for name in BASELINES:
    t0 = time.time()
    pop = V2EvolutionaryPopulation(
        population_size=15,
        num_rounds_per_gen=30,
        observability="full",
        seed=0,
        use_baseline=name,
    )
    res = pop.run_evolution(num_generations=10)
    elapsed = time.time() - t0
    traj = res["trajectory"]
    print(f"{name:<25} final coop g9={traj[-1]['cooperation_rate_mean']:.3f}, "
          f"mean coop={[g['cooperation_rate_mean'] for g in traj]}, "
          f"({elapsed:.1f}s)")
