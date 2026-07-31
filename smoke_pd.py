"""Smoke test for the new PD game + 3-arg evaluate baselines.

Runs each of the 10 baselines (ALLC, ALLD, plus 8 leading-eight) for
1 seed x 30 generations to verify the new game logic works end-to-end
and the leading-eight maintain cooperation as expected.
"""
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
sys.path.insert(0, str(ROOT))

from experiments.v2_quantitative.population import V2EvolutionaryPopulation
from experiments.v2_quantitative.baselines import BASELINES


# Expectations (only checked with a flag, not blocking)
EXPECTATIONS = {
    "ALLC": ("== 1.0", lambda x: x >= 0.99),
    "ALLD": ("== 0.0", lambda x: x <= 0.01),
    # The successful leading-eight maintain cooperation
    "IS":   (">= 0.9", lambda x: x >= 0.9),
    "SS":   (">= 0.9", lambda x: x >= 0.9),
    "SJ":   (">= 0.9", lambda x: x >= 0.9),
    "SC":   (">= 0.9", lambda x: x >= 0.9),
    "SH":   (">= 0.9", lambda x: x >= 0.9),
    # The + variants can collapse (depending on initial conditions)
    "IS+":  (">= 0.0", lambda x: x >= 0.0),
    "SS+":  (">= 0.0", lambda x: x >= 0.0),
    "SJ+":  (">= 0.0", lambda x: x >= 0.0),
}


def run_one(name: str, seed: int, num_gens: int):
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
        api_key="dummy-not-used",
        api_base_url="http://localhost",
        mutation_temperature=0.8,
        seed=seed,
        results_dir=str(ROOT / "results" / "smoke_pd"),
        use_baseline=name,
    )
    print(f"\n[{name} seed{seed}] Starting baseline ({num_gens} gens)...")
    t0 = time.time()
    res = pop.run_evolution(num_generations=num_gens)
    elapsed = time.time() - t0
    final = res["trajectory"][-1]["cooperation_rate_mean"] if res["trajectory"] else None
    return final, elapsed, res


def main():
    print("=" * 70)
    print("M1 smoke test: 2-player PD game + 10 baselines (incl. 8 leading-eight)")
    print("=" * 70)
    n_gens = 30
    results = {}
    for name in BASELINES:
        try:
            final, elapsed, res = run_one(name, seed=0, num_gens=n_gens)
            results[name] = (final, elapsed)
            print(f"  [{name}] final coop = {final:.3f}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name] = (None, None)
    print("\n" + "=" * 70)
    print("Summary (1 seed x 30 gens, 2-player PD game):")
    print("=" * 70)
    for name, (final, elapsed) in results.items():
        flag = ""
        if name in EXPECTATIONS and final is not None:
            expected_str, check = EXPECTATIONS[name]
            if not check(final):
                flag = f"  <-- UNEXPECTED (expected {expected_str}, got {final:.3f})"
            else:
                flag = f"  OK (expected {expected_str})"
        print(f"  {name:<8s} final={final if final is None else f'{final:.3f}':<7s}  "
              f"{elapsed if elapsed is None else f'{elapsed:.1f}s':<7s} {flag}")


if __name__ == "__main__":
    main()
