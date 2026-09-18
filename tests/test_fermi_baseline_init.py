"""Weighted baseline introductions during Fermi updates."""
import random
from collections import Counter

import pytest

from experiments.run_fermi_v3 import build_parser
from experiments.v2_quantitative.baselines import BASELINE_VERSION, LEADING_EIGHT, get_baseline
from experiments.v2_quantitative.evolution_architecture import (
    AgentSnapshot, FermiEvolutionRule,
)
from experiments.v2_quantitative.population import V2EvolutionaryPopulation


SNAPSHOT = tuple(AgentSnapshot(i, "parent", 1.0, i) for i in range(2))


def test_independent_initialization_uses_fresh_prompt(monkeypatch):
    pop = V2EvolutionaryPopulation(
        population_size=4, agent_type="agent-type1",
        mutation_rate_on_adoption=.1, fermi_init_source="llm",
    )
    captured = []

    def request(prompt, label):
        captured.append((prompt, label))
        return "test return value"

    monkeypatch.setattr(pop, "_request_valid_code", request)
    assert pop._llm_init_code(3) == "test return value"
    assert captured[0][0] == pop._init_prompt()
    assert "slot 3" in captured[0][1]


def test_baseline_pool_probabilities_and_lineage():
    rule = FermiEvolutionRule(
        beta=0, mutation_rate=1, updates_per_gen=2, init_source="baseline",
    )
    rng = random.Random(17)
    counts = Counter()
    for _ in range(5000):
        plan = rule.plan(SNAPSHOT, rng=rng, next_gen=1, lineage_by_agent_id={})
        for job in plan.jobs:
            assert job.operator == "baseline_init"
            assert job.origin == "independent_init"
            assert job.parent_id is None
            assert job.parent_code is None
            counts[job.baseline_name] += 1
    total = sum(counts.values())
    assert set(counts) == {"ALLD", *LEADING_EIGHT}
    assert 0.46 < counts["ALLD"] / total < 0.54
    for name in LEADING_EIGHT:
        assert 0.045 < counts[name] / total < 0.08


@pytest.mark.parametrize("source,mu,operator", [
    ("llm", 1, "llm_init"),
    ("baseline", 0, "llm_mutate"),
])
def test_source_only_changes_independent_initialization(source, mu, operator):
    rule = FermiEvolutionRule(
        beta=0, mutation_rate=mu, updates_per_gen=2, init_source=source,
    )
    rng = random.Random(7)
    jobs = [job for _ in range(20) for job in rule.plan(
        SNAPSHOT, rng=rng, next_gen=1, lineage_by_agent_id={}
    ).jobs]
    assert jobs
    assert all(job.operator == operator and job.baseline_name is None for job in jobs)


def test_baseline_updates_use_no_llm_and_are_reproducible_across_concurrency(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("baseline introduction called the LLM")

    def run(concurrency):
        pop = V2EvolutionaryPopulation(
            population_size=32, agent_type="agent-type1", seed=17,
            fermi_init_source="baseline", mutation_rate_on_adoption=1,
            llm_concurrency=concurrency,
        )
        pop.agents = [pop._make_agent(get_baseline("ALLC"), i) for i in range(32)]
        pop._init_lineage()
        monkeypatch.setattr(pop, "_call_llm", forbidden)
        pop._select_and_reproduce_fermi(next_gen=1)
        introduced = [
            agent for agent in pop.agents
            if pop._slot_birth[agent.agent_id]["birth_gen"] == 1
        ]
        assert introduced
        allowed = {get_baseline(name) for name in ("ALLD", *LEADING_EIGHT)}
        assert all(agent.code in allowed for agent in introduced)
        assert pop._fallback_mutation_count == 0
        assert pop._result_config(2)["fermi_init_source"] == "baseline"
        assert pop._result_config(2)["baseline_version"] == BASELINE_VERSION
        return [agent.code for agent in pop.agents], pop._lineage_events

    assert run(1) == run(4)


def test_initial_population_still_uses_llm(monkeypatch):
    pop = V2EvolutionaryPopulation(
        population_size=2, agent_type="agent-type1", fermi_init_source="baseline",
        llm_concurrency=1,
    )
    calls = []

    def generate(*args, **kwargs):
        calls.append(1)
        return get_baseline("ALLC")

    monkeypatch.setattr(pop, "_call_llm", generate)
    pop._init_population_llm()
    assert len(calls) == 2
    assert all(agent.code.strip() == get_baseline("ALLC").strip() for agent in pop.agents)


def test_cli_source_and_unsupported_agent_type():
    parser = build_parser()
    assert parser.parse_args([]).fermi_init_source == "llm"
    assert parser.parse_args(["--fermi-init-source", "baseline"]).fermi_init_source == "baseline"
    with pytest.raises(ValueError, match="Fermi and agent-type1"):
        V2EvolutionaryPopulation(agent_type="agent-type2", fermi_init_source="baseline")
