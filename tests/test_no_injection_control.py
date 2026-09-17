"""Offline controls for the experiment driver; never written into results."""
from experiments.run_no_injection_control import AuditedPopulation, checkpoint, evaluate
from experiments.v2_quantitative.baselines import get_baseline


def make_population():
    pop = AuditedPopulation(population_size=4, target_interactions_per_gen=80,
        agent_type='agent-type1', seed=17, num_generations=4, llm_concurrency=1,
        mutation_rate_on_adoption=0, fermi_init_source='baseline',
        action_error_probability=.01, observation_error_probability=.01)
    def initialize():
        pop.agents = [pop._make_agent(get_baseline(name), i)
                      for i, name in enumerate(['ALLC', 'ALLD', 'L1', 'L2'])]
        pop._next_agent_id = 4
    pop._init_population_llm = initialize
    # Deterministic operator solely for verifying runner equivalence offline.
    pop._llm_small_mutate_code = lambda code, fitness, preserve_id: get_baseline('ALLC' if preserve_id % 2 else 'L1')
    return pop


def test_control_loop_and_checkpoint_resume_match_existing_engine():
    standard = make_population().run_evolution(4)
    pop = make_population()
    pop._init_population_llm()
    pop._init_lineage()
    trajectory = [evaluate(pop, 0)]
    for gen in range(1, 4):
        pop._select_and_reproduce_by_method(next_gen=gen)
        trajectory.append(evaluate(pop, gen))
        partial = checkpoint(pop, trajectory, {'test_only': True})
        # Reconstruct every generation to test actual on-disk checkpoint semantics.
        replacement = make_population()
        replacement._restore_from_evolution_log(partial)
        replacement.rng.setstate(replacement._tuple_tree(partial['config']['rng_state']))
        pop = replacement
    assert trajectory == standard['trajectory']
    assert partial['lineage_events'] == standard['lineage_events']
    assert partial['config']['rng_state'] == standard['config']['rng_state']
    births = [e for e in partial['lineage_events'] if e['birth_gen'] > 0]
    assert births and all(e['origin'] == 'imitate' and e['parent_lineage_id'] is not None for e in births)
