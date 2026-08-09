"""Smoke test: run a single trial (full_seed0) at G=30 to verify the pipeline.
If this works, we launch the full 12-trial parallel run.
"""
import sys, subprocess, os
from pathlib import Path
ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
OUT = ROOT / "results" / "smoke_exp12"
OUT.mkdir(parents=True, exist_ok=True)

inline = f'''
import sys, json
sys.path.insert(0, r"{ROOT}")
from experiments.evolution.population import EvolutionaryPopulation
from experiments.config.load_env import get_api_key, get_base_url
api_key = get_api_key("deepseek")
api_base = get_base_url("deepseek")
assert api_key, "DEEPSEEK_API_KEY not set"
print(f"api_key prefix: {{api_key[:6]}}")
print(f"api_base: {{api_base}}")
pop = EvolutionaryPopulation(
    population_size=15, num_rounds_per_gen=30,
    benefit=2, cost=1, observability="full", observability_p=1.0,
    elite_count=2, num_eliminate=5, tournament_size=3,
    llm_provider="openai", llm_model="deepseek-v4-flash",
    api_key=api_key, api_base_url=api_base,
    mutation_temperature=0.8, seed=0, results_dir=r"{OUT}",
)
print("Running G=5 smoke (4 gens)...")
results = pop.run_evolution(num_generations=4)
final = results["trajectory"][-1] if results["trajectory"] else {{}}
print(f"Smoke OK. Final coop: {{final.get('cooperation_rate_mean', 'N/A')}}")
print(f"Final pop size: {{len(results.get('final_population', []))}}")
with open(r"{OUT}\\smoke_result.json", "w", encoding="utf-8") as f:
    json.dump({{"trajectory": results["trajectory"],
                "n_final": len(results.get("final_population", [])),
                "elapsed_proxy": "smoke"}}, f, indent=2)
'''

print("Launching smoke test (4 generations, full obs, seed 0)...")
r = subprocess.run([sys.executable, "-c", inline], cwd=str(ROOT))
print(f"\nSmoke rc={r.returncode}")
if r.returncode == 0:
    print("PASS — ready to launch full exp12_g30_n15")
else:
    print("FAIL — fix and retry")
