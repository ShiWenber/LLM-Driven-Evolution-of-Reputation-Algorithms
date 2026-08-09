"""Smoke test the v2 quantitative pipeline with one baseline at G=10.
"""
import sys, time
from pathlib import Path
ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
sys.path.insert(0, str(ROOT))

from experiments.config.load_env import get_api_key, get_base_url
from experiments.v2_quantitative.population import V2EvolutionaryPopulation

t0 = time.time()
pop = V2EvolutionaryPopulation(
    population_size=15,
    num_rounds_per_gen=30,
    observability="full",
    seed=0,
    use_baseline="QuantIS_Schmid2023",
)
print("Running 10 generations with QuantIS baseline...")
res = pop.run_evolution(num_generations=10)
print(f"Done in {time.time()-t0:.1f}s")
for g in res["trajectory"]:
    print(f"  gen {g['generation']}: coop={g['cooperation_rate_mean']:.3f}, fit_mean={g['fitness_mean']:.1f}")
