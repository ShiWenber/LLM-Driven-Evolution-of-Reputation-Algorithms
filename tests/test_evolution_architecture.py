"""Contract tests for pluggable scenarios and evolution rules.

These tests intentionally describe the target architecture before the
implementation.  They protect the generation barrier: planners are pure with
respect to population state, LLM work is batched, and commits are ordered.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from experiments.v2_quantitative.evolution_architecture import (
    AgentSnapshot,
    EvaluationResult,
    GenerationPlan,
    OffspringJob,
    OffspringResult,
    RetainedAgent,
)
from experiments.v2_quantitative.population import V2EvolutionaryPopulation


def _selection_population(*, llm_concurrency=2):
    pop = V2EvolutionaryPopulation(
        population_size=4,
        elite_count=1,
        num_eliminate=2,
        tournament_size=2,
        learning_method="tournament",
        llm_concurrency=llm_concurrency,
        seed=7,
    )
    pop.agents = [
        SimpleNamespace(agent_id=i, code=f"parent-{i}", fitness=float(4 - i))
        for i in range(4)
    ]
    pop._next_agent_id = 4
    pop._slot_lineage = {i: 100 + i for i in range(4)}
    pop._slot_birth = {}
    pop._lineage_events = []
    pop._next_lineage_id = 200
    pop._fallback_mutation_count = 0
    # The worker callback is stubbed below, so avoid constructing a real
    # provider client during the batch executor's eager initialization.
    pop._llm_client = object()
    pop._make_agent = lambda code, agent_id: SimpleNamespace(
        agent_id=agent_id,
        code=code,
        fitness=0.0,
        reputations={},
    )
    return pop


def test_tournament_llm_work_is_parallel_and_commit_order_is_stable():
    pop = _selection_population(llm_concurrency=2)
    lock = threading.Lock()
    state = {"active": 0, "maximum": 0}

    def mutate_code(parent_code, parent_fitness):
        del parent_fitness
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        # Different completion times exercise the ordered commit boundary.
        time.sleep(0.02 if parent_code.endswith("0") else 0.01)
        with lock:
            state["active"] -= 1
        return f"child-of-{parent_code}"

    pop._mutate_code = mutate_code
    pop._select_and_reproduce(next_gen=1)

    assert state["maximum"] == 2
    assert [agent.agent_id for agent in pop.agents[-2:]] == [4, 5]
    assert all(agent.code.startswith("child-of-parent-") for agent in pop.agents[-2:])
    assert [event["lineage_id"] for event in pop._lineage_events] == [200, 201]


def test_tournament_request_failure_is_counted_and_falls_back_to_its_parent():
    pop = _selection_population(llm_concurrency=2)
    attempted_parents = []

    def fail(parent_code, parent_fitness):
        del parent_fitness
        attempted_parents.append(parent_code)
        return None

    pop._mutate_code = fail
    pop._select_and_reproduce(next_gen=3)

    assert pop._fallback_mutation_count == 2
    assert [agent.code for agent in pop.agents[-2:]] == attempted_parents
    assert all(event["origin"] == "mutate" for event in pop._lineage_events)
    assert all(event["birth_gen"] == 3 for event in pop._lineage_events)


def test_custom_game_scenario_is_injected_without_changing_population():
    class Scenario:
        def __init__(self):
            self.calls = []

        def evaluate(self, agents, *, generation_seed, num_rounds):
            self.calls.append((tuple(a.agent_id for a in agents), generation_seed, num_rounds))
            return EvaluationResult(
                cooperation_rate_mean=0.25,
                n_interactions=12,
                round_num=num_rounds,
                payoffs=(3.0, 1.0),
            )

        def simulation_parameters(self, *, num_generations, initial_reputation):
            return {
                "population_size": 2,
                "num_rounds_per_gen": 6,
                "num_generations": num_generations,
                "num_pairs": 1,
                "benefit": 0.0,
                "cost": 0.0,
                "cc_payoff": 0.0,
                "initial_reputation": initial_reputation,
            }

        def overall_rules_prompt(self, *, num_generations, initial_reputation):
            del num_generations, initial_reputation
            return "CUSTOM GAME RULES"

    scenario = Scenario()
    pop = V2EvolutionaryPopulation(
        population_size=2,
        num_rounds_per_gen=6,
        game_scenario=scenario,
        seed=11,
    )
    pop.agents = [SimpleNamespace(agent_id=0), SimpleNamespace(agent_id=1)]

    stats = pop._run_one_generation()

    assert stats == {
        "cooperation_rate_mean": 0.25,
        "n_interactions": 12,
        "round_num": 6,
        "payoffs": [3.0, 1.0],
    }
    assert scenario.calls[0][0] == (0, 1)
    assert pop._overall_game_rules_prompt() == "CUSTOM GAME RULES"


def test_custom_evolution_rule_is_injected_without_dispatch_changes():
    class KeepPopulationRule:
        name = "keep"

        def __init__(self):
            self.calls = []

        def plan(self, population, *, rng, next_gen, lineage_by_agent_id):
            del rng
            self.calls.append((population, next_gen, lineage_by_agent_id))
            return GenerationPlan(
                population_size=len(population),
                retained=tuple(
                    RetainedAgent(output_index=i, snapshot=agent)
                    for i, agent in enumerate(population)
                ),
                jobs=(),
            )

    rule = KeepPopulationRule()
    pop = V2EvolutionaryPopulation(
        population_size=2,
        evolution_rule=rule,
        seed=5,
    )
    pop.agents = [
        SimpleNamespace(agent_id=8, code="a", fitness=2.0),
        SimpleNamespace(agent_id=9, code="b", fitness=1.0),
    ]
    pop._slot_lineage = {8: 80, 9: 90}
    pop._make_agent = lambda code, agent_id: SimpleNamespace(
        agent_id=agent_id, code=code, fitness=0.0, reputations={}
    )

    pop._select_and_reproduce_by_method(next_gen=4)

    assert [agent.agent_id for agent in pop.agents] == [8, 9]
    assert [agent.code for agent in pop.agents] == ["a", "b"]
    assert rule.calls[0][1] == 4
    assert rule.calls[0][0] == (
        AgentSnapshot(8, "a", 2.0, 80),
        AgentSnapshot(9, "b", 1.0, 90),
    )
    config = pop._result_config(num_generations=1)
    assert config["game_scenario"] == "reputation_prisoners_dilemma"
    assert config["evolution_rule"] == "keep"


def test_offspring_generator_is_an_injectable_execution_boundary():
    job = OffspringJob(
        ordinal=0,
        output_index=0,
        operator="llm_mutate",
        mutation_kind="full",
        preserve_agent_id=4,
        parent_id=4,
        parent_lineage_id=40,
        parent_code="parent",
        parent_fitness=1.0,
        origin="mutate",
        birth_gen=2,
    )

    class Rule:
        name = "replace-one"

        def plan(self, population, **_kwargs):
            return GenerationPlan(len(population), (), (job,))

    class Generator:
        def __init__(self):
            self.batches = []

        def execute_batch(self, jobs):
            self.batches.append(tuple(jobs))
            return [OffspringResult(job=job, code="generated")]

    generator = Generator()
    pop = V2EvolutionaryPopulation(
        population_size=1,
        evolution_rule=Rule(),
        offspring_generator=generator,
    )
    pop.agents = [SimpleNamespace(agent_id=4, code="parent", fitness=1.0)]
    pop._slot_lineage = {4: 40}
    pop._make_agent = lambda code, agent_id: SimpleNamespace(
        agent_id=agent_id, code=code, fitness=0.0, reputations={}
    )

    pop._select_and_reproduce_by_method(next_gen=2)

    assert generator.batches == [(job,)]
    assert pop.agents[0].code == "generated"
