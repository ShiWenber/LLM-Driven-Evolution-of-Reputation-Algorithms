"""Smoke test for the v3 (full LLMAgent class) interface.

Goal: verify that the type-2 infrastructure can run a population end-to-end
without the LLM. Uses the ALLC / ALLD baseline classes (always-cooperate
/ always-defect) plus a custom mock class that mimics a real LLM-emitted
strategy (records observations and decides based on its own internal
state, NOT on the framework's reputations matrix).

Runs 1 seed × 30 generations with population_size=5 (smaller for speed).
Prints final cooperation rate and fitness for each baseline / mock.

If this passes, the type-2 infrastructure is wired correctly and we can
proceed to M3 (real LLM probe).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
sys.path.insert(0, str(ROOT))

from experiments.v2_quantitative.population import V2EvolutionaryPopulation
from experiments.v2_quantitative.agent_full import (
    ALLC_CLASS_SOURCE, ALLD_CLASS_SOURCE,
)

OUT = ROOT / "results" / "smoke_v3"
OUT.mkdir(parents=True, exist_ok=True)

# A mock LLM class that uses ONLY its own state (not the framework's
# reputations matrix). Should illustrate that the type-2 interface gives
# the LLM genuine freedom over state structure.
MOCK_LLM_SOURCE = '''
class LLMAgent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self._ctx_opponent_id = None
        # LLM-owned state: per-opponent history of last action seen
        self.last_action_by_opponent = {}

    def decide(self) -> bool:
        # Tit-for-tat: cooperate unless the opponent has defected
        # against us in the most recent observation.
        opp = self._ctx_opponent_id
        if opp is None:
            return True
        return self.last_action_by_opponent.get(opp, "cooperate") == "cooperate"

    def observe(self, donor_id, donor_action, recipient_id, recipient_action) -> None:
        # Record what the opponent (relative to me) just did, if either
        # side of this joint action is me. Self-judgment: if donor_id or
        # recipient_id is me, the OTHER id is my opponent.
        if donor_id == self.agent_id:
            self.last_action_by_opponent[recipient_id] = recipient_action
        elif recipient_id == self.agent_id:
            self.last_action_by_opponent[donor_id] = donor_action
'''


def run_one(name: str, code: str, seed: int, n_agents: int = 5, n_gens: int = 30) -> dict:
    trial_dir = OUT / f"{name}_seed{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    pop = V2EvolutionaryPopulation(
        population_size=n_agents,
        num_rounds_per_gen=15,
        observability="full",
        observability_p=1.0,
        elite_count=2,
        num_eliminate=2,
        tournament_size=2,
        api_key="",
        api_base_url="",
        mutation_temperature=0.8,
        seed=seed,
        results_dir=str(trial_dir),
        use_baseline=None,
        agent_type="v3",
    )
    # Bypass the LLM: build agents from `code` and patch the two
    # LLM-related methods to no-ops (init + mutate). This exercises
    # the full selection / reproduction cycle without an API call.
    pop.agents = [pop._new_agent(code) for _ in range(n_agents)]
    pop._init_population_llm = lambda: None
    pop._mutate = lambda parent_code, parent_fitness: parent_code  # no change
    print(f"\n[{name} seed{seed}] Initialized {len(pop.agents)} agents with mock source")
    t0 = time.time()
    res = pop.run_evolution(num_generations=n_gens)
    elapsed = time.time() - t0
    res["elapsed_sec"] = elapsed
    out_path = trial_dir / "evolutionary.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    final = res["trajectory"][-1]["cooperation_rate_mean"]
    fit = res["trajectory"][-1]["fitness_mean"]
    print(f"[{name} seed{seed}] Done in {elapsed:.1f}s. Final coop = {final:.3f}, fitness = {fit:.2f}")
    return res


def main():
    print("=" * 60)
    print("M2 smoke test: v3 (full LLMAgent class) infrastructure")
    print("=" * 60)
    results = []
    # Test 1: ALLC baseline (always cooperate)
    r = run_one("ALLC", ALLC_CLASS_SOURCE, seed=0)
    results.append(("ALLC (always C)", r["trajectory"][-1]["cooperation_rate_mean"], 1.0))
    # Test 2: ALLD baseline (always defect)
    r = run_one("ALLD", ALLD_CLASS_SOURCE, seed=0)
    results.append(("ALLD (always D)", r["trajectory"][-1]["cooperation_rate_mean"], 0.0))
    # Test 3: mock tit-for-tat (uses its own per-opponent state)
    r = run_one("TFT_mock", MOCK_LLM_SOURCE, seed=0)
    final = r["trajectory"][-1]["cooperation_rate_mean"]
    results.append(("TFT_mock (own state)", final, "should be ~1.0 (TFT sustains coop)"))
    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"{'name':30s} {'final_coop':>12s}  expected")
    print("-" * 60)
    for name, got, expected in results:
        ok = "PASS" if (
            (expected == 1.0 and got > 0.9) or
            (expected == 0.0 and got < 0.1) or
            (isinstance(expected, str) and got > 0.5)
        ) else "FAIL"
        print(f"{name:30s} {got:>12.3f}  {str(expected):20s}  {ok}")
    # Dump one agent's brain to confirm class state survives
    pop_path = OUT / "TFT_mock_seed0" / "evolutionary.json"
    if pop_path.exists():
        d = json.loads(pop_path.read_text(encoding="utf-8"))
        # final_population is a list of dicts; the brain is gone (we
        # never serialized it), but the cooperation_rate is in the
        # trajectory. Just confirm the per-gen final coop looks right.
        gens = [g["cooperation_rate_mean"] for g in d["trajectory"]]
        print(f"\nTFT_mock per-gen coop: {[f'{c:.2f}' for c in gens]}")


if __name__ == "__main__":
    main()
