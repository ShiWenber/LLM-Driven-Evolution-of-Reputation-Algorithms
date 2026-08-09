"""Quick 2-parallel test: 2 trials, max 2 workers.
If 2-parallel takes ~16 min, it's safe. If much longer (>25), it's hitting limits.
"""
import subprocess, sys, time, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
OUT = ROOT / "results" / "test_2parallel"
OUT.mkdir(parents=True, exist_ok=True)


def run_trial(obs, seed):
    trial_dir = OUT / f"{obs}_seed{seed}"
    trial_dir.mkdir(exist_ok=True)
    log = OUT / f"{obs}_seed{seed}.log"
    obs_p = 0.0 if obs == "private" else 1.0 if obs == "full" else float(obs.split("_")[1])
    inline = f"""
import sys, json
sys.path.insert(0, r"{ROOT}")
from experiments.evolution.population import EvolutionaryPopulation
from experiments.config.load_env import get_api_key, get_base_url
api_key = get_api_key("deepseek")
api_base = get_base_url("deepseek")
pop = EvolutionaryPopulation(
    population_size=15, num_rounds_per_gen=30,
    benefit=2, cost=1, observability="{obs}", observability_p={obs_p},
    elite_count=2, num_eliminate=5, tournament_size=3,
    llm_provider="openai", llm_model="deepseek-v4-flash",
    api_key=api_key, api_base_url=api_base,
    mutation_temperature=0.8, seed={seed}, results_dir=r"{trial_dir}",
)
results = pop.run_evolution(num_generations=30)
with open(r"{trial_dir}\\evolutionary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("DONE", flush=True)
"""
    start = time.time()
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, "-u", "-c", inline],
                          cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, timeout=3600)
    return obs, seed, r.returncode, time.time() - start


t0 = time.time()
with ThreadPoolExecutor(max_workers=2) as ex:
    futs = [ex.submit(run_trial, "partial_0.7", 0), ex.submit(run_trial, "full", 0)]
    for f in as_completed(futs):
        obs, seed, rc, elapsed = f.result()
        print(f"  {obs}_seed{seed}: rc={rc} {elapsed/60:.1f} min", flush=True)
print(f"Total {(time.time()-t0)/60:.1f} min")
