"""Run the v2 quantitative-baseline experiment.

For each baseline strategy (8 of them), run 3 seeds × 30 generations
with full observability. Also run the LLM-driven evolution (3 seeds ×
30 generations). Save raw JSON trajectories + a comparison plot.

Output: results/quantitative_baseline/
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
sys.path.insert(0, str(ROOT))

from experiments.config.load_env import get_api_key, get_base_url
from experiments.v2_quantitative.population import V2EvolutionaryPopulation
from experiments.v2_quantitative.baselines import BASELINES

OUT = ROOT / "results" / "quantitative_baseline"
OUT.mkdir(parents=True, exist_ok=True)


def run_one(name: str, seed: int, num_gens: int = 30, mode: str = "baseline", agent_type: str = "v2"):
    """Run one trial and save JSON. Skip if result already exists."""
    trial_dir = OUT / f"{name}_seed{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    out_path = trial_dir / "evolutionary.json"
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            t = existing.get("trajectory", [])
            schema_version = existing.get("config", {}).get("schema_version", 1)
            config_agent_type = existing.get("config", {}).get("agent_type", "v2")
            if (len(t) >= num_gens and schema_version >= 3
                    and config_agent_type == agent_type):
                print(f"[{name} seed{seed}] Already done (v{schema_version}, {len(t)} gens, agent_type={config_agent_type}, final coop = {t[-1].get('cooperation_rate_mean', 'n/a')}). Skipping.")
                return existing
            else:
                reason = "missing per-gen code" if len(t) >= num_gens else f"only {len(t)} gens"
                if config_agent_type != agent_type:
                    reason = f"agent_type={config_agent_type} != {agent_type}"
                print(f"[{name} seed{seed}] Existing JSON is v{schema_version} ({reason}), re-running.")
        except Exception as e:
            print(f"[{name} seed{seed}] Existing JSON unreadable, re-running. ({e})")
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
        use_baseline=(name if mode == "baseline" else None),
        agent_type=agent_type,
    )
    print(f"\n[{name} seed{seed}] Starting ({mode})...")
    t0 = time.time()
    res = pop.run_evolution(num_generations=num_gens)
    elapsed = time.time() - t0
    res["elapsed_sec"] = elapsed
    out_path.write_text(
        json.dumps(res, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    final = res["trajectory"][-1]["cooperation_rate_mean"] if res["trajectory"] else None
    print(f"[{name} seed{seed}] Done in {elapsed/60:.1f} min. Final coop = {final}")
    return res


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-type", choices=["v2", "v3"], default="v2",
                        help="v2 = two-function interface (default), v3 = full LLMAgent class")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-gens", type=int, default=30)
    args = parser.parse_args()
    seeds = args.seeds
    n_gens = args.n_gens
    summary = {}
    # Run all baselines
    for name in BASELINES:
        summary[name] = []
        for seed in seeds:
            try:
                res = run_one(name, seed, n_gens, mode="baseline", agent_type=args.agent_type)
                summary[name].append({
                    "seed": seed,
                    "elapsed_sec": res["elapsed_sec"],
                    "trajectory": res["trajectory"],
                })
            except Exception as e:
                print(f"  ERROR {name} seed{seed}: {e}")
                summary[name].append({"seed": seed, "error": str(e)})
    # Run LLM evolution (only if v2 — v3 LLM run has its own script)
    if args.agent_type == "v2":
        summary["LLM_evolution"] = []
        for seed in seeds:
            try:
                res = run_one("LLM_evolution", seed, n_gens, mode="llm", agent_type="v2")
                summary["LLM_evolution"].append({
                    "seed": seed,
                    "elapsed_sec": res["elapsed_sec"],
                    "trajectory": res["trajectory"],
                })
            except Exception as e:
                print(f"  ERROR LLM seed{seed}: {e}")
                summary["LLM_evolution"].append({"seed": seed, "error": str(e)})
    # Save summary
    out_summary = OUT / f"summary_{args.agent_type}.json"
    out_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSummary saved to {out_summary}")


if __name__ == "__main__":
    main()
