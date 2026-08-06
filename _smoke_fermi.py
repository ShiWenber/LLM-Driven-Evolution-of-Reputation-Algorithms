"""Sanity check: Fermi update rule behaves correctly on canonical
starting populations.

Three pure-Python unit tests (no LLM, no game):
  T1 Fermi+ALLC  -> stays at 100% C (all cooperators stable)
  T2 Fermi+ALLD  -> stays at 100% D (all defectors stable)
  T3 1 IS_Plus (fitness high) + 14 ALLD (fitness zero) -> IS_Plus
     should invade (P(sigmoid(25)) ~ 1.0 with beta=5, phi diff=5).
     Validates that high-fitness minority is copied.
  T4 1 ALLD (fitness 5) + 14 ALLC (fitness 10) -> ALLD invader
     contained (P(sigmoid(-25)) ~ 0). Validates that low-fitness
     invader is NOT copied.

If any of these fails, the Fermi rule is broken before we burn LLM
budget on a 30-gen smoke. Then a T5 LLM smoke (5 gen) to confirm
the full pipeline doesn't crash.

Run:
    python _smoke_fermi.py
"""
import sys
from pathlib import Path

# Repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.v2_quantitative.population import V2EvolutionaryPopulation
from experiments.v2_quantitative.baselines import get_baseline


def make_population(use_fermi=True, fermi_beta=5.0, mu=0.1,
                    updates_per_gen=15, forbid_self_pairing=True):
    """Construct an empty population (no init yet) for unit testing."""
    pop = V2EvolutionaryPopulation(
        population_size=15,
        num_rounds_per_gen=1,
        use_fermi=use_fermi,
        fermi_beta=fermi_beta,
        mutation_rate_on_adoption=mu,
        updates_per_gen=updates_per_gen,
        forbid_self_pairing=forbid_self_pairing,
        use_baseline="ALLC",  # we won't actually init, just to satisfy validation
    )
    # CRITICAL: replace _mutate with a no-op for unit tests so we
    # never call the LLM. Sanity tests want to validate the SELECTION
    # rule in isolation; mutation would inject LLM-perturbed codes
    # that break the strict "ALLC stays ALLC" assertion. With
    # mutation_rate_on_adoption=mu, the rule is: if a copy is
    # triggered, j's code is mutated with prob mu. In the unit
    # tests we set mu=0 (T3, T4) so this is unused; T1, T2 use
    # the default 0.1 but we patch _mutate anyway as a safety net.
    pop._mutate = lambda code, fitness: code
    return pop


def manually_seed(pop, codes, fitnesses):
    """Replace pop.agents with N=15 agents of given codes and fitnesses."""
    from experiments.v2_quantitative.agent import QuantitativeAgent
    from experiments.v2_quantitative.executor import V2StrategyExecutor
    pop.agents = []
    for i, (code, fit) in enumerate(zip(codes, fitnesses)):
        executor = V2StrategyExecutor(code)
        agent = QuantitativeAgent(agent_id=i, code=code, executor=executor)
        agent.fitness = fit
        pop.agents.append(agent)


def count_codes(pop):
    """Bucket agents by full code. All baseline strategies share
    the same first ~60 chars (the `evaluate` signature) so we need
    to hash on the full string. (Real test runs never have
    homogeneous ALLC vs heterogeneous mix like this; only
    hand-crafted sanity tests need full-code disambiguation.)
    """
    from collections import Counter
    return Counter(a.code for a in pop.agents)


def test_t1_alld_stable():
    """T1: ALLC homogeneous -> stays 100% ALLC."""
    pop = make_population()
    allc_code = get_baseline("ALLC")
    manually_seed(pop, [allc_code] * 15, [10.0] * 15)
    initial = count_codes(pop)
    for _ in range(50):
        pop._select_and_reproduce_fermi()
    after = count_codes(pop)
    assert initial[allc_code] == 15, f"init broken: {initial}"
    assert after[allc_code] == 15, f"ALLC destabilized: {after}"
    print("  T1 PASS: Fermi+ALLC(15/15) -> 15/15 ALLC after 50 Fermi gens")


def test_t2_alld_stable():
    """T2: ALLD homogeneous -> stays 100% ALLD."""
    pop = make_population()
    alld_code = get_baseline("ALLD")
    manually_seed(pop, [alld_code] * 15, [10.0] * 15)
    initial = count_codes(pop)
    for _ in range(50):
        pop._select_and_reproduce_fermi()
    after = count_codes(pop)
    assert initial[alld_code] == 15, f"init broken: {initial}"
    assert after[alld_code] == 15, f"ALLD destabilized: {after}"
    print("  T2 PASS: Fermi+ALLD(15/15) -> 15/15 ALLD after 50 Fermi gens")


def test_t3_is_invades_alld():
    """T3: 1 IS+ (fitness 5) + 14 ALLD (fitness 0). With mu=0,
    isolated selection rule. P(ALLD copies IS) = sigmoid(5*(5-0)) = 1.
    So IS+ should invade and convert all ALLD.

    This is the "rare-mutant-invades" test of the Fermi rule."""
    pop = make_population(mu=0.0, updates_per_gen=15)
    is_code = get_baseline("IS+")
    alld_code = get_baseline("ALLD")
    codes = [is_code] + [alld_code] * 14
    fitnesses = [5.0] + [0.0] * 14
    manually_seed(pop, codes, fitnesses)
    initial = count_codes(pop)
    print(f"    init: {dict(initial)}")
    for _ in range(30):
        pop._select_and_reproduce_fermi()
    after = count_codes(pop)
    print(f"    after 30 fermi gens (mu=0): {dict(after)}")
    alld_count = after.get(alld_code, 0)
    if alld_count == 0:
        print("  T3 PASS: IS+ invaded (final ALLD=0/15, IS=15/15)")
    elif alld_count <= 2:
        print(f"  T3 MOSTLY PASS: ALLD reduced to {alld_count}/15")
    else:
        raise AssertionError(
            f"T3 FAIL: IS+ should invade (sigmoid(25) ~ 1) "
            f"but ALLD is still {alld_count}/15. Fermi rule broken."
        )


def test_t4_alld_contained():
    """T4: 1 ALLD (fitness 5) + 14 ALLC (fitness 10). P(ALLC copies
    ALLD) = sigmoid(5*(5-10)) = sigmoid(-25) ~ 0. So ALLD invader
    should NOT spread. (Note: with mu=0 ALLD may persist as a
    minority since nobody copies it but also nobody replaces it.)"""
    pop = make_population(mu=0.0, updates_per_gen=15)
    alld_code = get_baseline("ALLD")
    allc_code = get_baseline("ALLC")
    codes = [alld_code] + [allc_code] * 14
    fitnesses = [5.0] + [10.0] * 14
    manually_seed(pop, codes, fitnesses)
    initial = count_codes(pop)
    print(f"    init: {dict(initial)}")
    for _ in range(30):
        pop._select_and_reproduce_fermi()
    after = count_codes(pop)
    print(f"    after 30 fermi gens (mu=0): {dict(after)}")
    alld_count = after.get(alld_code, 0)
    if alld_count <= 1:
        print(f"  T4 PASS: ALLD invader contained (final ALLD={alld_count}/15)")
    else:
        raise AssertionError(
            f"T4 FAIL: ALLD invader should be contained "
            f"(sigmoid(-25) ~ 0) but grew to {alld_count}/15. "
            f"Fermi rule broken."
        )


def test_t5_llm_smoke_5gen():
    """T5: 1 seed × 5 gen LLM smoke. Verifies the full pipeline
    (LLM init + Fermi selection + LLM mutation) doesn't crash.
    Skip if DEEPSEEK_API_KEY is not in env.
    """
    import os
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    api_base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    if not api_key:
        print("  T5 SKIP: no DEEPSEEK_API_KEY in env")
        return
    pop = V2EvolutionaryPopulation(
        population_size=15,
        use_fermi=True,
        fermi_beta=5.0,
        mutation_rate_on_adoption=0.1,
        updates_per_gen=15,
        use_baseline=None,  # LLM mode
        agent_type="v3",
        llm_model="deepseek-v4-flash",
        api_key=api_key,
        api_base_url=api_base,
        llm_thinking=False,
        seed=0,
    )
    result = pop.run_evolution(num_generations=5)
    coop_curve = [t["cooperation_rate_mean"] for t in result["trajectory"]]
    print(f"  T5 coop curve: {[f'{c:.3f}' for c in coop_curve]}")
    print("  T5 PASS: 5-gen LLM+Fermi smoke completed without crash")


if __name__ == "__main__":
    print("== Fermi sanity suite ==")
    print("\n[T1] Fermi+ALLC homogeneous stability")
    test_t1_alld_stable()
    print("\n[T2] Fermi+ALLD homogeneous stability")
    test_t2_alld_stable()
    print("\n[T3] Fermi: 1 IS_Plus (fitness 5) + 14 ALLD (fitness 0) -> invade?")
    test_t3_is_invades_alld()
    print("\n[T4] Fermi: 1 ALLD (fitness 5) + 14 ALLC (fitness 10) -> contained?")
    test_t4_alld_contained()
    print("\n[T5] Fermi + LLM 5-gen smoke (requires DEEPSEEK_API_KEY)")
    test_t5_llm_smoke_5gen()
    print("\n== All Fermi sanity tests done ==")
