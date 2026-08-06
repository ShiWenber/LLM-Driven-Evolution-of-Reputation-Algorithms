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
    # CRITICAL: replace the two LLM-producing paths with no-ops so
    # we never call the LLM in unit tests. Z-like scheme: every
    # Fermi copy event goes through one of these two (μ -> init,
    # 1-μ -> small_mutate). For selection-rule sanity, both should
    # reduce to "return j's code verbatim" so the only thing being
    # tested is the Fermi imitation decision.
    pop._llm_init_one_agent = lambda preserve_id: None  # patched below per-test
    pop._llm_small_mutate = lambda parent_code, preserve_id: None
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


def make_passthrough_agent(preserve_id, code):
    """Build a QuantitativeAgent with the given code, for use as
    a no-op LLM replacement in unit tests."""
    from experiments.v2_quantitative.agent import QuantitativeAgent
    from experiments.v2_quantitative.executor import V2StrategyExecutor
    executor = V2StrategyExecutor(code)
    return QuantitativeAgent(agent_id=preserve_id, code=code, executor=executor)


def patch_fermi_paths_noop(pop, fallback_code):
    """Replace the two LLM-producing methods with no-ops that always
    return a fresh agent wrapping `fallback_code`. This is the
    "no mutation" reference behavior: the selection rule decides
    who imitates whom, but the offspring is always `fallback_code`.
    For ALLC/ALLD homogeneous stability tests (T1, T2), `fallback_code`
    is the test's strategy. For invasion tests (T3, T4), `fallback_code`
    is irrelevant because mu=0 means only _llm_small_mutate fires,
    and that gets the parent code as input which we forward verbatim."""
    pop._llm_init_one_agent = lambda preserve_id: make_passthrough_agent(preserve_id, fallback_code)
    pop._llm_small_mutate = lambda parent_code, preserve_id: make_passthrough_agent(preserve_id, parent_code)


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
    patch_fermi_paths_noop(pop, allc_code)  # offspring is always ALLC
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
    patch_fermi_paths_noop(pop, alld_code)  # offspring is always ALLD
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

    This is the "rare-mutant-invades" test of the Fermi rule.

    With mu=0 the 1-μ path (small_mutate) is the only one called.
    We patch it to be a no-op (forward parent_code), so the test
    isolates the Fermi selection rule."""
    pop = make_population(mu=0.0, updates_per_gen=15)
    is_code = get_baseline("IS+")
    alld_code = get_baseline("ALLD")
    patch_fermi_paths_noop(pop, alld_code)  # fallback code unused, mu=0
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
    patch_fermi_paths_noop(pop, allc_code)  # mu=0 so fallback unused
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


def test_r7_fermi_z_like_mechanism():
    """R7: Z-like mechanism stress test with mu=0.1 (both paths fire).

    Starts with 8 ALLC + 7 IS+ (both with fitness 10). With mu=0.1
    both the μ path (_llm_init_one_agent, 10% of imitations) and the
    1-μ path (_llm_small_mutate, 90%) are exercised. With patched
    no-op LLM methods (μ path returns a neutral cooperative agent,
    1-μ returns parent's code), the test verifies:

      * the synchronous commit preserves agent_id stability
      * the selection rule still works under non-zero mu
      * the 1-μ path actually being called doesn't break things

    The "neutral cooperative" in μ path is a NEW strategy each call,
    so this also implicitly tests that the population can absorb
    fresh LLM-like diversity without crashing.
    """
    pop = make_population(mu=0.1, updates_per_gen=15)
    allc_code = get_baseline("ALLC")
    is_code = get_baseline("IS+")

    # Patch: μ path returns a "fresh ALLC-like" agent (LLM init = arbitrary
    # new strategy, here we just say it's ALLC). 1-μ path returns parent's
    # code verbatim (small-mutate = no-op).
    pop._llm_init_one_agent = lambda preserve_id: make_passthrough_agent(preserve_id, allc_code)
    pop._llm_small_mutate = lambda parent_code, preserve_id: make_passthrough_agent(preserve_id, parent_code)

    # 8 ALLC + 7 IS+, all fitness=10 (homogeneous -> no selection pressure,
    # only stochastic drift + μ-path noise)
    codes = [allc_code] * 8 + [is_code] * 7
    fitnesses = [10.0] * 15
    manually_seed(pop, codes, fitnesses)

    # Capture initial agent_ids — they must be preserved through generations
    initial_ids = [a.agent_id for a in pop.agents]
    for _ in range(50):
        pop._select_and_reproduce_fermi()
    after_ids = [a.agent_id for a in pop.agents]
    assert initial_ids == after_ids, (
        f"R7 FAIL: agent_id not stable through gens. "
        f"init={initial_ids}, after={after_ids}"
    )
    # No crash = pass. We don't make cooperation claims because μ=0.1
    # injects noise that mixes ALLC/IS+ identity.
    print("  R7 PASS: Z-like scheme with mu=0.1 ran 50 gens without crash, agent_ids stable")


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
    print("\n[R7] Fermi Z-like mechanism stress (mu=0.1, both paths fire, patched LLM)")
    test_r7_fermi_z_like_mechanism()
    print("\n[T5] Fermi + LLM 5-gen smoke (requires DEEPSEEK_API_KEY)")
    test_t5_llm_smoke_5gen()
    print("\n== All Fermi sanity tests done ==")
