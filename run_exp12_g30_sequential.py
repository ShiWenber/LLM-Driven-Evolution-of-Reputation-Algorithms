"""Run 12 trials (4 obs x 3 seeds) at G=30 SEQUENTIALLY.

Sequential to avoid DeepSeek rate-limit contention. Total wall time ~3h.
Each trial is a subprocess.run of a python -c snippet (per-trial seed).
Logs go to per-trial .log files; aggregate manifest at end.
"""
import subprocess, sys, os, time, json
from pathlib import Path

ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
OUT_DIR = ROOT / "results" / "exp12_g30_n15"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.json"

OBS = ["private", "partial_0.3", "partial_0.7", "full"]
SEEDS = [0, 1, 2]
POP, G, T, B, C = 15, 30, 30, 2.0, 1.0
ELITE, ELIM, TOUR, TEMP = 2, 5, 3, 0.8
MODEL = "deepseek-v4-flash"


def run_trial(obs, seed):
    trial_dir = OUT_DIR / f"{obs}_seed{seed}"
    trial_dir.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{obs}_seed{seed}.log"
    if obs == "private":
        obs_p = 0.0
    elif obs == "full":
        obs_p = 1.0
    else:
        obs_p = float(obs.split("_")[1])
    inline = f"""
import sys, json
sys.path.insert(0, r"{ROOT}")
from experiments.evolution.population import EvolutionaryPopulation
from experiments.config.load_env import get_api_key, get_base_url
api_key = get_api_key("deepseek")
api_base = get_base_url("deepseek")
assert api_key, "DEEPSEEK_API_KEY not set"
pop = EvolutionaryPopulation(
    population_size={POP}, num_rounds_per_gen={T},
    benefit={B}, cost={C}, observability="{obs}", observability_p={obs_p},
    elite_count={ELITE}, num_eliminate={ELIM}, tournament_size={TOUR},
    llm_provider="openai", llm_model="{MODEL}",
    api_key=api_key, api_base_url=api_base,
    mutation_temperature={TEMP}, seed={seed}, results_dir=r"{trial_dir}",
)
results = pop.run_evolution(num_generations={G})
results["model"] = "{MODEL}"
results["provider"] = "openai"
results["observability"] = "{obs}"
results["seed"] = {seed}
with open(r"{trial_dir}\\evolutionary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"FINAL coop={{results['trajectory'][-1].get('cooperation_rate_mean', 'N/A')}}")
"""
    start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {obs}_seed{seed}", flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        r = subprocess.run(
            [sys.executable, "-u", "-c", inline], cwd=str(ROOT),
            stdout=logf, stderr=subprocess.STDOUT, timeout=3600,
        )
    elapsed = time.time() - start
    success = r.returncode == 0
    print(f"[{time.strftime('%H:%M:%S')}] END   {obs}_seed{seed} "
          f"({elapsed/60:.1f} min, rc={r.returncode})", flush=True)
    return (obs, seed, success, elapsed)


def main():
    trials = [(obs, seed) for obs in OBS for seed in SEEDS]
    # Skip trials that already have a complete evolutionary.json with 30 generations
    to_run = []
    for obs, seed in trials:
        evo = OUT_DIR / f"{obs}_seed{seed}" / "evolutionary.json"
        if evo.exists():
            try:
                data = json.loads(evo.read_text(encoding="utf-8"))
                traj = data.get("trajectory", [])
                if len(traj) >= 30:
                    print(f"[SKIP] {obs}_seed{seed} (already complete, {len(traj)} gens)", flush=True)
                    continue
            except Exception:
                pass
        to_run.append((obs, seed))
    print(f"=== Sequential G=30 main experiment: {len(to_run)} new + {len(trials)-len(to_run)} cached = {len(trials)} total ===", flush=True)
    print(f"Expected total: ~{len(to_run)*15/60:.1f} hours", flush=True)
    manifest = {
        "experiment": "G=30 main plan replay (v15 setup, N=15), sequential",
        "model": MODEL, "population": POP, "generations": G, "rounds": T,
        "benefit": B, "cost": C, "obs": OBS, "seeds": SEEDS,
        "trials": [],
    }
    overall_start = time.time()
    for obs, seed in to_run:
        result = run_trial(obs, seed)
        manifest["trials"].append({
            "obs": obs, "seed": seed,
            "success": result[2], "elapsed_sec": result[3],
        })
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    total = (time.time() - overall_start) / 60
    print(f"=== All done in {total:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
