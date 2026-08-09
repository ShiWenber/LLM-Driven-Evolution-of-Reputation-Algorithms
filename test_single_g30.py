"""Sequential test: 1 trial at G=30 to verify timing.
"""
import sys, time, json
from pathlib import Path
ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
sys.path.insert(0, str(ROOT))
from experiments.evolution.population import EvolutionaryPopulation
from experiments.config.load_env import get_api_key, get_base_url

OUT = ROOT / "results" / "test_g30_single"
OUT.mkdir(parents=True, exist_ok=True)
api_key = get_api_key("deepseek")
api_base = get_base_url("deepseek")
print(f"api_key prefix: {api_key[:6]}", flush=True)
print(f"api_base: {api_base}", flush=True)

start = time.time()
pop = EvolutionaryPopulation(
    population_size=15, num_rounds_per_gen=30,
    benefit=2, cost=1, observability="partial_0.7", observability_p=0.7,
    elite_count=2, num_eliminate=5, tournament_size=3,
    llm_provider="openai", llm_model="deepseek-v4-flash",
    api_key=api_key, api_base_url=api_base,
    mutation_temperature=0.8, seed=0, results_dir=str(OUT),
)
print(f"[{time.time()-start:.1f}s] Starting G=30 evolution...", flush=True)
results = pop.run_evolution(num_generations=30)
elapsed = time.time() - start
final = results["trajectory"][-1] if results["trajectory"] else {}
print(f"[{elapsed:.1f}s] Done. Final coop: {final.get('cooperation_rate_mean', 'N/A')}", flush=True)
print(f"[{elapsed:.1f}s] Final pop: {len(results.get('final_population', []))} strategies", flush=True)
out_json = OUT / "test_result.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({"elapsed_sec": elapsed, "n_gens": 30, "final_coop": final.get("cooperation_rate_mean", None), "trajectory": results["trajectory"]}, f, indent=2)
print(f"Saved to {out_json}", flush=True)
