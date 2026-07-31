"""Run only the LLM evolution seeds.

Default `agent_type="v3"` (full LLMAgent class). Use `--agent-type v2` to
re-run the two-function-interface variant. Will overwrite any existing
JSON whose config does not match the requested agent_type.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
sys.path.insert(0, str(ROOT))

from experiments.config.load_env import get_api_key, get_base_url
from experiments.v2_quantitative.population import V2EvolutionaryPopulation

OUT = ROOT / "results" / "quantitative_baseline"


def run_one(seed: int, num_gens: int, agent_type: str, label: str = "LLM_evolution"):
    """Run one LLM seed; overwrite if existing JSON has wrong agent_type."""
    trial_dir = OUT / f"{label}_seed{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    out_path = trial_dir / "evolutionary.json"
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            sv = existing.get("config", {}).get("schema_version", 1)
            existing_at = existing.get("config", {}).get("agent_type", "v2")
            if sv >= 3 and existing_at == agent_type:
                t = existing.get("trajectory", [])
                if len(t) >= num_gens:
                    print(f"[{label} seed{seed}] Already v3+ with agent_type={existing_at} ({len(t)} gens), skipping.")
                    return
                print(f"[{label} seed{seed}] Existing v{sv} agent_type={existing_at} has only {len(t)} gens, re-running.")
            else:
                print(f"[{label} seed{seed}] Existing v{sv} agent_type={existing_at} (want agent_type={agent_type}), re-running.")
                out_path.unlink()
        except Exception as e:
            print(f"[{label} seed{seed}] Could not read existing JSON ({e}); re-running.")

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
    print(f"\n[{label} seed{seed}] Starting (agent_type={agent_type})...")
    t0 = time.time()
    res = pop.run_evolution(num_generations=num_gens)
    elapsed = time.time() - t0
    res["elapsed_sec"] = elapsed
    out_path.write_text(
        json.dumps(res, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    final = res["trajectory"][-1]["cooperation_rate_mean"] if res["trajectory"] else None
    print(f"[{label} seed{seed}] Done in {elapsed/60:.1f} min. Final coop = {final:.3f}")
    print(f"  schema_version={res['config'].get('schema_version')}, agent_type={res['config'].get('agent_type')}, "
          f"final agent_ids sample={sorted([a['agent_id'] for a in res['final_population']])}")
    return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-type", choices=["v2", "v3"], default="v3",
                        help="v2 = two-function interface, v3 = full LLMAgent class (default)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-gens", type=int, default=30)
    parser.add_argument("--label", default="LLM_evolution",
                        help="Trial-dir name suffix (default: LLM_evolution). "
                             "Use e.g. 'LLM_v3_evolution' to keep v2 and v3 results separate.")
    args = parser.parse_args()
    for seed in args.seeds:
        run_one(seed, args.n_gens, args.agent_type, label=args.label)
