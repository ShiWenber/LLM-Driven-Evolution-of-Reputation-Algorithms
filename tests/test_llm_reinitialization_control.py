"""Offline validation of the mu=.1 LLM-source control, never experimental data."""
from experiments.run_llm_reinitialization_control import AuditedPopulation, checkpoint, evaluate
from experiments.v2_quantitative.baselines import get_baseline


def make_population():
    pop = AuditedPopulation(population_size=4, target_interactions_per_gen=80,
        agent_type='agent-type1', seed=17, num_generations=40, llm_concurrency=1,
        mutation_rate_on_adoption=.1, fermi_init_source='llm',
        action_error_probability=.01, observation_error_probability=.01)
    def initialize():
        pop.agents = [pop._make_agent(get_baseline(name), i)
                      for i, name in enumerate(['ALLC', 'ALLD', 'L1', 'L2'])]
        pop._next_agent_id = 4
    pop._init_population_llm = initialize
    # Deterministic stand-ins only for comparing driver and engine semantics.
    pop._llm_small_mutate_code = lambda code, fitness, preserve_id: get_baseline('ALLC' if preserve_id % 2 else 'L1')
    pop._llm_init_code = lambda preserve_id: get_baseline('ALLD')
    return pop


def test_reinitialization_loop_and_resume_match_engine():
    standard = make_population().run_evolution(40)
    pop = make_population()
    pop._init_population_llm()
    pop._init_lineage()
    trajectory = [evaluate(pop, 0)]
    for gen in range(1, 40):
        pop._select_and_reproduce_by_method(next_gen=gen)
        trajectory.append(evaluate(pop, gen))
        partial = checkpoint(pop, trajectory, {'test_only': True})
        replacement = make_population()
        replacement._restore_from_evolution_log(partial)
        replacement.rng.setstate(replacement._tuple_tree(partial['config']['rng_state']))
        pop = replacement
    assert trajectory == standard['trajectory']
    assert partial['lineage_events'] == standard['lineage_events']
    assert partial['config']['rng_state'] == standard['config']['rng_state']
    roots = [e for e in partial['lineage_events'] if e['origin'] == 'independent_init']
    assert roots and all(e['parent_lineage_id'] is None and e['parent_id'] is None for e in roots)
    assert any(e['origin'] == 'imitate' for e in partial['lineage_events'])


def test_independent_initialization_uses_fresh_prompt():
    pop = AuditedPopulation(population_size=4, agent_type='agent-type1',
                            mutation_rate_on_adoption=.1, fermi_init_source='llm')
    captured = []
    def request(prompt, label):
        captured.append((prompt, label))
        return 'test return value'
    pop._request_valid_code = request
    assert pop._llm_init_code(3) == 'test return value'
    assert captured[0][0] == pop._init_prompt()
    assert 'slot 3' in captured[0][1]
