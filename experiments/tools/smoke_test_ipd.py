"""Smoke test for IPD game + IPD evolution runner (no API calls).

Verifies:
- IPDGame.play_match works with classical IPD strategies
- IPDGame.all_play_all returns sane statistics
- IPDTrialConfig and IPDTrialResult can be constructed
- RandomMutationOperator produces valid code that can be compiled
- IPDEvolutionaryRunner runs end-to-end with random mutation

This script is for developer testing only; not part of the experiment pipeline.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.game.ipd_game import (
    IPDGame, tit_for_tat, always_cooperate, always_defect,
    pavlov, grim_trigger, load_ipd_strategy_from_code,
    CLASSICAL_STRATEGIES,
)
from experiments.sandbox.validator import clean_code, validate_strategy_code, CodeValidationError
from experiments.evolution.ipd_evolution import (
    IPDTrialConfig,
    IPDTrialResult,
    IPDEvolutionaryRunner,
    _FALLBACK_TIT_FOR_TAT,
)


def test_ipd_match():
    print("\n[smoke] test_ipd_match")
    game = IPDGame(num_rounds=100, noise=0.0)

    # TFT vs TFT: mutual cooperation, payoff ≈ 300
    r = game.play_match(tit_for_tat, tit_for_tat, seed=42)
    print(f"  TFT vs TFT: coop_rate={r.cooperation_rate:.3f}, "
          f"payoff_a={r.payoff_a}, payoff_b={r.payoff_b}")
    assert r.cooperation_rate > 0.95
    assert abs(r.payoff_a - 300) < 1

    # ALLC vs ALLD: ALLC gets ~0, ALLD gets ~500
    r = game.play_match(always_cooperate, always_defect, seed=42)
    print(f"  ALLC vs ALLD: coop_rate={r.cooperation_rate:.3f}, "
          f"payoff_a={r.payoff_a} (should be ~0), "
          f"payoff_b={r.payoff_b} (should be ~500)")
    # coop_actions records decision_a (ALLC) which is always True
    assert r.cooperation_rate > 0.95
    assert r.payoff_a < 1
    assert r.payoff_b > 450

    # ALLD vs ALLD: mutual defection, payoff = 100
    r = game.play_match(always_defect, always_defect, seed=42)
    print(f"  ALLD vs ALLD: coop_rate={r.cooperation_rate:.3f}, "
          f"payoff_a={r.payoff_a}, payoff_b={r.payoff_b}")
    assert r.cooperation_rate < 0.1
    assert abs(r.payoff_a - 100) < 1

    # Pavlov vs ALLD: Pavlov starts cooperating, gets exploited, then alternates
    r = game.play_match(pavlov, always_defect, seed=42)
    print(f"  Pavlov vs ALLD: coop_rate={r.cooperation_rate:.3f}, "
          f"payoff_a={r.payoff_a}, payoff_b={r.payoff_b}")
    # Pavlov doesn't always lose — it can recover if ALLD ever cooperates (it doesn't)
    # So we expect payoff_a < payoff_b
    assert r.payoff_b > r.payoff_a

    # Grim vs ALLD: cooperates once, then defects forever
    r = game.play_match(grim_trigger, always_defect, seed=42)
    print(f"  Grim vs ALLD: coop_rate={r.cooperation_rate:.3f}, "
          f"payoff_a={r.payoff_a}, payoff_b={r.payoff_b}")

    print("  [OK] IPD match tests pass")


def test_all_play_all():
    print("\n[smoke] test_all_play_all")
    game = IPDGame(num_rounds=50, noise=0.0)
    strategies = [tit_for_tat, always_cooperate, always_defect, pavlov]
    result = game.all_play_all(strategies, seed=42)
    print(f"  num_agents={result['num_agents']}")
    print(f"  tournament_mean_cooperation={result['tournament_mean_cooperation']:.3f}")
    print(f"  tournament_mean_payoff={result['tournament_mean_payoff']:.3f}")
    for a in result['per_agent']:
        print(f"    agent {a['agent_id']}: coop={a['cooperation_rate']:.3f} "
              f"payoff={a['mean_payoff']:.1f}")
    assert result['num_agents'] == 4
    assert result['tournament_mean_cooperation'] > 0.2
    print("  [OK] all_play_all tests pass")


def test_load_ipd_strategy_from_code():
    print("\n[smoke] test_load_ipd_strategy_from_code")
    code = '''
def evaluate(current_reputation, observation, my_history, round_num):
    return 0.0

def decide(recipient_reputation, round_num, my_history):
    if not my_history:
        return True
    return my_history[-1].get("partner_action") == "donate"
'''
    strategy = load_ipd_strategy_from_code(code)
    assert strategy is not None
    # First round: cooperate
    assert strategy([], 0) is True
    # After opponent defected: defect
    assert strategy(
        [{"action": "donate", "partner_action": "not_donate"}], 1
    ) is False
    # After opponent cooperated: cooperate
    assert strategy(
        [{"action": "not_donate", "partner_action": "donate"}], 1
    ) is True
    print("  [OK] load_ipd_strategy_from_code works")


def test_random_mutation_produces_valid_code():
    print("\n[smoke] test_random_mutation_produces_valid_code")
    from experiments.evolution.mutation import RandomMutationOperator
    rm = RandomMutationOperator()
    for seed in range(5):
        code = rm.mutate(_FALLBACK_TIT_FOR_TAT, parent_fitness=0.0)
        if code is not None:
            try:
                validate_strategy_code(code)
                strategy = load_ipd_strategy_from_code(code)
                assert strategy is not None, f"seed={seed}: code validates but IPD load fails"
                print(f"  seed={seed}: [OK] valid")
            except CodeValidationError as e:
                print(f"  seed={seed}: [FAIL] {e}")
                raise
    print("  [OK] random mutation produces valid code")


def test_random_mutation_runner():
    print("\n[smoke] test_random_mutation_runner (no LLM)")
    from experiments.evolution.mutation import RandomMutationOperator

    config = IPDTrialConfig(
        population_size=6,
        num_generations=3,
        ipd_rounds_per_match=20,
        mutation_rate=0.5,
        seed=0,
        use_random_mutation=True,
        output_dir="results/_smoke",
    )
    mutator = RandomMutationOperator()
    runner = IPDEvolutionaryRunner(config, mutator)

    # Bypass LLM init by injecting fallback strategies directly
    runner._initial_strategies = [_FALLBACK_TIT_FOR_TAT] * config.population_size
    runner._agent_codes = list(runner._initial_strategies)
    runner._agents = runner._build_agents()
    assert len(runner._agents) == 6

    result = runner.run()
    print(f"  trial completed: {len(result.trajectory)} generations")
    print(f"  initial strategies: {len(result.initial_strategies)}")
    print(f"  final strategies: {len(result.final_strategies)}")
    assert len(result.trajectory) == 3
    assert len(result.final_strategies) == 6
    for gen_data in result.trajectory:
        print(f"    gen {gen_data['generation']}: "
              f"coop={gen_data['mean_cooperation']:.3f} "
              f"payoff={gen_data['mean_payoff']:.1f}")
    print("  [OK] random-mutation runner tests pass")


def test_classical_strategies_all_in_dict():
    print("\n[smoke] test_classical_strategies_all_in_dict")
    expected = {"tit_for_tat", "always_cooperate", "always_defect", "pavlov", "grim_trigger"}
    actual = set(CLASSICAL_STRATEGIES.keys())
    assert expected == actual, f"missing={expected-actual} extra={actual-expected}"
    print(f"  [OK] {len(CLASSICAL_STRATEGIES)} classical strategies present")


if __name__ == "__main__":
    print("=" * 60)
    print("IPD smoke tests (no API calls)")
    print("=" * 60)
    test_ipd_match()
    test_all_play_all()
    test_load_ipd_strategy_from_code()
    test_classical_strategies_all_in_dict()
    test_random_mutation_produces_valid_code()
    test_random_mutation_runner()
    print("\n[smoke] all tests passed")
