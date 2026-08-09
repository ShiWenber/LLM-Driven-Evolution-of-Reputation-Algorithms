"""100 gen × 1 seed × thinking=disabled.
Label: LLM_v3_g100_thinking_off.
Overwrites LLM_v3_g100_thinking_off_seed0/.
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

def run_one(seed: int, num_gens: int, agent_type: str, label: str):
    trial_dir = OUT / f"{label}_seed{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    out_path = trial_dir / "evolutionary.json"
    if out_path.exists():
        out_path.unlink()
        print(f"  removed existing {out_path}")

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
        agent_type=agent_type,
    )
    print(f"\n[{label} seed{seed}] Starting (agent_type={agent_type}, thinking=disabled, n_gens={num_gens})...", flush=True)
    t0 = time.time()
    res = pop.run_evolution(num_generations=num_gens)
    elapsed = time.time() - t0
    res["elapsed_sec"] = elapsed
    out_path.write_text(
        json.dumps(res, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    final = res["trajectory"][-1]["cooperation_rate_mean"] if res["trajectory"] else None
    print(f"[{label} seed{seed}] Done in {elapsed/60:.1f} min. Final coop = {final:.3f}", flush=True)
    init_codes = [a.get("code", "") for a in res["trajectory"][0]["population"]]
    n_real = sum(1 for c in init_codes if "def decide" in c and "def observe" in c)
    n_fallback = sum(1 for c in init_codes if c.count("def ") == 3 and "return True" in c and "return None" in c)
    print(f"  init: {n_real}/15 real LLM classes, {n_fallback}/15 FALLBACK")
    print(f"  schema_version={res['config'].get('schema_version')}, agent_type={res['config'].get('agent_type')}")
    print(f"  final agent_ids sample={sorted([a['agent_id'] for a in res['final_population']])}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-gens", type=int, default=100)
    p.add_argument("--label", default="LLM_v3_g100_thinking_off")
    args = p.parse_args()
    run_one(seed=args.seed, num_gens=args.n_gens, agent_type="v3", label=args.label)
