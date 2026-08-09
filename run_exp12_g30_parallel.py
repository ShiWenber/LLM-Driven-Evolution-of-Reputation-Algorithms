"""Run 12 trials (4 obs x 3 seeds) at G=30, max 4 in parallel.

Each trial is a separate Python subprocess that directly instantiates
EvolutionaryPopulation (bypassing experiments.main's CLI seed-count loop).
This gives us per-trial seed control without wasted re-runs.

Concurrency: 4 (one per observability, matching v15 4-obs parallel pattern).
"""
import subprocess, sys, os, time, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
OUT_DIR = ROOT / "results" / "exp12_g30_n15"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OBS = ["private", "partial_0.3", "partial_0.7", "full"]
SEEDS = [0, 1, 2]
POP = 15
G = 30
T = 30
B = 2.0
C = 1.0
ELITE = 2
ELIM = 5
TOUR = 3
TEMP = 0.8
MODEL = "deepseek-v4-flash"

LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.json"


def run_trial(obs: str, seed: int):
    """Run one trial as a Python subprocess with explicit seed control."""
    trial_dir = OUT_DIR / f"{obs}_seed{seed}"
    trial_dir.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{obs}_seed{seed}.log"
    if obs == "private":
        obs_p = 0.0
    elif obs == "full":
        obs_p = 1.0
    else:
        obs_p = float(obs.split("_")[1])
    # Inline script: import EvolutionaryPopulation, run, dump json
    inline = f"""
import sys, json, os
sys.path.insert(0, r"{ROOT}")
from experiments.evolution.population import EvolutionaryPopulation
from experiments.config.load_env import get_api_key, get_base_url
api_key = get_api_key("deepseek")
api_base = get_base_url("deepseek")
assert api_key, "DEEPSEEK_API_KEY not set in .env"
pop = EvolutionaryPopulation(
    population_size={POP},
    num_rounds_per_gen={T},
    benefit={B},
    cost={C},
    observability="{obs}",
    observability_p={obs_p},
    elite_count={ELITE},
    num_eliminate={ELIM},
    tournament_size={TOUR},
    llm_provider="openai",
    llm_model="{MODEL}",
    api_key=api_key,
    api_base_url=api_base,
    mutation_temperature={TEMP},
    seed={seed},
    results_dir=r"{trial_dir}",
)
results = pop.run_evolution(num_generations={G})
results["model"] = "{MODEL}"
results["provider"] = "openai"
results["observability"] = "{obs}"
results["seed"] = {seed}
out = r"{trial_dir}\\evolutionary.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Saved to {{out}}")
print(f"Final gen coop: {{results['trajectory'][-1].get('cooperation_rate_mean', 'N/A')}}")
"""
    env = os.environ.copy()
    start = time.time()
    print(f"[START] {obs}_seed{seed} @ {time.strftime('%H:%M:%S')}", flush=True)
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            result = subprocess.run(
                [sys.executable, "-c", inline], cwd=str(ROOT), env=env,
                stdout=logf, stderr=subprocess.STDOUT,
                timeout=7200,  # 2h safety margin per trial
            )
        elapsed = time.time() - start
        success = result.returncode == 0
        print(f"[END]   {obs}_seed{seed} @ {time.strftime('%H:%M:%S')} "
              f"({elapsed/60:.1f} min, rc={result.returncode})", flush=True)
        return (obs, seed, success, elapsed, str(log_path))
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {obs}_seed{seed}", flush=True)
        return (obs, seed, False, time.time() - start, str(log_path))
    except Exception as e:
        print(f"[ERROR]  {obs}_seed{seed}: {e}", flush=True)
        return (obs, seed, False, time.time() - start, str(log_path))


def main():
    print(f"=== G=30 main experiment ===", flush=True)
    print(f"Output: {OUT_DIR}", flush=True)
    print(f"Trials: {len(OBS)} obs x {len(SEEDS)} seeds = {len(OBS) * len(SEEDS)}", flush=True)
    print(f"Concurrency: 4 (one per obs)", flush=True)
    print(flush=True)

    trials = [(obs, seed) for obs in OBS for seed in SEEDS]
    manifest = {
        "experiment": "G=30 main plan replay (v15 setup, N=15)",
        "model": MODEL,
        "population": POP, "generations": G, "rounds": T,
        "benefit": B, "cost": C,
        "obs": OBS, "seeds": SEEDS,
        "trials": [],
    }
    overall_start = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_trial, obs, seed): (obs, seed) for obs, seed in trials}
        for fut in as_completed(futs):
            obs, seed, success, elapsed, log = fut.result()
            manifest["trials"].append({
                "obs": obs, "seed": seed,
                "success": success, "elapsed_sec": elapsed,
                "log": log,
            })
            with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(flush=True)
    print(f"=== Done in {(time.time() - overall_start)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
