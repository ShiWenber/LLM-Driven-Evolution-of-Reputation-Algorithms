"""Run all 10 PD baselines (ALLC, ALLD, IS, SS, SJ, SC, SH, IS+, SS+, SJ+)
across 3 seeds × 30 generations. No LLM involvement (that's M3).

This is the M1 verification: it confirms that the rewritten 2-player
simultaneous-PD game, the 3-arg evaluate() interface, and the 8 leading-eight
+ 2 simple baselines all behave as expected (ALLC/leading-eight -> 1.0,
ALLD -> 0.0) when wired through the full V2EvolutionaryPopulation stack.

Output: results/quantitative_baseline/{name}_seed{seed}/evolutionary.json
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
sys.path.insert(0, str(ROOT))

from experiments.v2_quantitative.population import V2EvolutionaryPopulation
from experiments.v2_quantitative.baselines import BASELINES

OUT = ROOT / "results" / "quantitative_baseline"
OUT.mkdir(parents=True, exist_ok=True)


def run_one(name: str, seed: int, num_gens: int = 30) -> dict:
    """Run one trial and save JSON. Skip if a schema_version>=3 result exists."""
    trial_dir = OUT / f"{name}_seed{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    out_path = trial_dir / "evolutionary.json"
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            t = existing.get("trajectory", [])
            schema_version = existing.get("config", {}).get("schema_version", 1)
            if len(t) >= num_gens and schema_version >= 3:
                print(f"[{name} seed{seed}] Already done (v{schema_version}, {len(t)} gens, final coop = {t[-1].get('cooperation_rate_mean', 'n/a')}). Skipping.")
                return existing
            else:
                reason = "missing per-gen code" if len(t) >= num_gens else f"only {len(t)} gens"
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
        api_key="",
        api_base_url="",
        mutation_temperature=0.8,
        seed=seed,
        results_dir=str(trial_dir),
        use_baseline=name,
    )
    print(f"\n[{name} seed{seed}] Starting (baseline)...")
    t0 = time.time()
    res = pop.run_evolution(num_generations=num_gens)
    elapsed = time.time() - t0
    res["elapsed_sec"] = elapsed
    out_path.write_text(
        json.dumps(res, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    final = res["trajectory"][-1]["cooperation_rate_mean"] if res["trajectory"] else None
    print(f"[{name} seed{seed}] Done in {elapsed/60:.2f} min. Final coop = {final}")
    return res


def main():
    seeds = [0, 1, 2]
    n_gens = 30
    summary = {}
    n_skipped = 0
    n_ran = 0
    for name in BASELINES:
        summary[name] = []
        for seed in seeds:
            try:
                # Detect skip ahead of time for the summary
                out_path = OUT / f"{name}_seed{seed}" / "evolutionary.json"
                would_skip = False
                if out_path.exists():
                    try:
                        d = json.loads(out_path.read_text(encoding="utf-8"))
                        if len(d.get("trajectory", [])) >= n_gens and d.get("config", {}).get("schema_version", 1) >= 3:
                            would_skip = True
                    except Exception:
                        pass
                res = run_one(name, seed, n_gens)
                if would_skip:
                    n_skipped += 1
                else:
                    n_ran += 1
                summary[name].append({
                    "seed": seed,
                    "elapsed_sec": res["elapsed_sec"],
                    "trajectory": res["trajectory"],
                })
            except Exception as e:
                print(f"  ERROR {name} seed{seed}: {e}")
                summary[name].append({"seed": seed, "error": str(e)})
    out_summary = OUT / "summary_pd_baselines.json"
    out_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSummary saved to {out_summary}")
    print(f"Total trials: {len(BASELINES) * len(seeds)}, ran: {n_ran}, skipped: {n_skipped}")
    # Print per-baseline final coop across seeds
    print("\n=== Final cooperation_rate_mean per baseline (across seeds) ===")
    for name in BASELINES:
        finals = []
        for entry in summary.get(name, []):
            tr = entry.get("trajectory", [])
            if tr:
                finals.append(tr[-1]["cooperation_rate_mean"])
        if finals:
            mean = sum(finals) / len(finals)
            print(f"  {name:6s}  mean={mean:.3f}  per-seed={[f'{x:.3f}' for x in finals]}")


if __name__ == "__main__":
    main()
