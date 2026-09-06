"""Generation-boundary resume tests for Fermi evolution logs."""
import json

from experiments.evolution_log import SCHEMA_VERSION
from experiments.v2_quantitative.population import (
    FALLBACK_STRATEGIES,
    V2EvolutionaryPopulation,
)


def _agent_record(agent_id, lineage_id, fitness):
    return {
        "agent_id": agent_id,
        "code": FALLBACK_STRATEGIES[agent_id % len(FALLBACK_STRATEGIES)],
        "fitness": fitness,
        "cooperation_rate": 0.0,
        "self_reputation": 0.0,
        "lineage_id": lineage_id,
        "parent_id": None,
        "parent_lineage_id": None,
        "origin": "initial",
        "birth_gen": 0,
    }


def _old_log():
    population = [_agent_record(0, 10, 1.0), _agent_record(1, 11, 2.0)]
    return {
        "trajectory": [
            {
                "generation": generation,
                "cooperation_rate_mean": 0.5,
                "n_interactions": 2,
                "fitness_mean": 1.5,
                "fitness_max": 2.0,
                "population": population,
            }
            for generation in (0, 1)
        ],
        "final_population": population,
        "lineage_events": [
            {
                "lineage_id": lineage_id,
                "parent_lineage_id": None,
                "parent_id": None,
                "origin": "initial",
                "birth_gen": 0,
            }
            for lineage_id in (10, 11)
        ],
        "config": {
            "schema_version": SCHEMA_VERSION,
            "agent_type": "agent-type1",
            "seed": 7,
            "population_size": 2,
            "use_fermi": True,
            "fallback_init_count": 0,
            "fallback_mutation_count": 0,
        },
    }


def _population():
    return V2EvolutionaryPopulation(
        population_size=2,
        target_interactions_per_gen=2,
        seed=7,
        use_fermi=True,
        updates_per_gen=2,
        llm_concurrency=1,
        agent_type="agent-type1",
    )


def test_resume_transitions_before_first_new_generation_and_preserves_history(monkeypatch):
    pop = _population()
    events = []

    def select(next_gen):
        events.append(("select", next_gen, [agent.fitness for agent in pop.agents]))

    def evaluate():
        events.append(("evaluate", pop.round_num_offset))
        return {
            "cooperation_rate_mean": 0.75,
            "n_interactions": 2,
            "payoffs": [3.0, 4.0],
        }

    monkeypatch.setattr(pop, "_select_and_reproduce_fermi", select)
    monkeypatch.setattr(pop, "_run_one_generation", evaluate)

    result = pop.resume_evolution(_old_log(), 1, source_path="old.json")

    assert events == [("select", 2, [1.0, 2.0]), ("evaluate", 2)]
    assert [row["generation"] for row in result["trajectory"]] == [0, 1, 2]
    assert len(result["lineage_events"]) == 2
    assert result["config"]["num_generations"] == 3
    assert result["config"]["resume"]["rng_mode"] == "derived_branch"
    assert result["config"]["rng_state_format"] == "python_random_v1"


def test_new_resume_log_uses_saved_rng_checkpoint(monkeypatch):
    first_pop = _population()
    monkeypatch.setattr(first_pop, "_select_and_reproduce_fermi", lambda next_gen: None)
    monkeypatch.setattr(first_pop, "_run_one_generation", lambda: {
        "cooperation_rate_mean": 0.5,
        "n_interactions": 2,
        "payoffs": [1.0, 2.0],
    })
    first = first_pop.resume_evolution(_old_log(), 1)
    # Exercise the real tuple -> JSON array -> tuple checkpoint round trip.
    first = json.loads(json.dumps(first))

    second_pop = _population()
    selected = []
    monkeypatch.setattr(
        second_pop,
        "_select_and_reproduce_fermi",
        lambda next_gen: selected.append(next_gen),
    )
    monkeypatch.setattr(second_pop, "_run_one_generation", lambda: {
        "cooperation_rate_mean": 0.5,
        "n_interactions": 2,
        "payoffs": [1.0, 2.0],
    })
    second = second_pop.resume_evolution(first, 1)

    assert selected == [3]
    assert [row["generation"] for row in second["trajectory"]] == [0, 1, 2, 3]
    assert second["config"]["resume"]["rng_mode"] == "checkpoint"
