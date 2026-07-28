"""Run only the 3 LLM evolution seeds with the v3 (stable agent_id) fix.
Will overwrite existing LLM JSONs since they are schema_version=2.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
sys.path.insert(0, str(ROOT))

from experiments.config.load_env import get_api_key, get_base_url
from experiments.v2_quantitative.population import V2EvolutionaryPopulation

OUT = ROOT / "results" / "quantitative_baseline"


def run_one(seed: int, num_gens: int = 30):
    """Run LLM_evolution seed, overwrite existing v2 JSON."""
    trial_dir = OUT / f"LLM_evolution_seed{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    out_path = trial_dir / "evolutionary.json"
    # Delete old v2 JSON to force re-run
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            sv = existing.get("config", {}).get("schema_version", 1)
            if sv < 3:
                print(f"[LLM_evolution seed{seed}] Deleting old v{sv} JSON to force re-run with v3 (stable agent_id) fix.")
                out_path.unlink()
            else:
                print(f"[LLM_evolution seed{seed}] Already v3+ ({sv}), skipping.")
                return
        except Exception as e:
            print(f"[LLM_evolution seed{seed}] Could not read existing JSON ({e}); re-running.")

    pop = V2EvolutionaryPopulation(
        population_size=15,
        num_rounds_per_gen=30,
        benefit=2.0,
        cost=1.0,
        observability="full",
        observability_p=1.0,
        elite_count=2,
        num_eliminate=5,
        tournament_size=3,
        llm_provider="openai",
        llm_model="deepseek-v4-flash",
        api_key=get_api_key("deepseek"),
        api_base_url=get_base_url("deepseek"),
        mutation_temperature=0.8,
        seed=seed,
        results_dir=str(trial_dir),
        use_baseline=None,
    )
    print(f"\n[LLM_evolution seed{seed}] Starting (v3, stable agent_id)...")
    t0 = time.time()
    res = pop.run_evolution(num_generations=num_gens)
    elapsed = time.time() - t0
    res["elapsed_sec"] = elapsed
    out_path.write_text(
        json.dumps(res, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    final = res["trajectory"][-1]["cooperation_rate_mean"] if res["trajectory"] else None
    print(f"[LLM_evolution seed{seed}] Done in {elapsed/60:.1f} min. Final coop = {final:.3f}")
    print(f"  schema_version={res['config'].get('schema_version')}, "
          f"final agent_ids sample={sorted([a['agent_id'] for a in res['final_population']])}")
    return res


if __name__ == "__main__":
    for seed in [0, 1, 2]:
        run_one(seed, 30)
