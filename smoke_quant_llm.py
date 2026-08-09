"""Smoke test LLM evolution with the v2 interface.
5 generations, 1 seed. Verify the prompt is OK and LLM produces valid code.
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
    api_key=get_api_key("deepseek"),
    api_base_url=get_base_url("deepseek"),
    use_baseline=None,  # LLM mode
)
print("Running 5 generations with LLM evolution (this will call LLM ~30+15=45 times)...")
res = pop.run_evolution(num_generations=5)
print(f"Done in {time.time()-t0:.1f}s")
for g in res["trajectory"]:
    print(f"  gen {g['generation']}: coop={g['cooperation_rate_mean']:.3f}, fit_mean={g['fitness_mean']:.1f}")
print(f"\nFinal pop size: {len(res['final_population'])}")
print(f"Final pop coop range: {min(s['cooperation_rate'] for s in res['final_population']):.3f} to {max(s['cooperation_rate'] for s in res['final_population']):.3f}")
