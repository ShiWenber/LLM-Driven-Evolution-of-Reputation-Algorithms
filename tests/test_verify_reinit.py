"""Per-generation full re-instantiation tests (no LLM calls).

Converted from the root-level ``_verify_reinit.py`` script to pytest.
Covers BOTH ``agent_type='agent-type1'`` / ``'agent-type2'`` and BOTH
selection paths (tournament / fermi):

  1. every agent object is rebuilt (new instance, same slot)
  2. agent_id set is preserved
  3. reputations are RESET to the initial matrix ``{agent_id: INITIAL_REPUTATION}``
  4. no lineage events are recorded for untouched slots
"""
import pytest

from experiments.v2_quantitative.agent import INITIAL_REPUTATION
from experiments.v2_quantitative.agent_full import (
    INITIAL_REPUTATION as FULL_INITIAL_REPUTATION,
)
from experiments.v2_quantitative.population import (
    FALLBACK_CLASS_V3,
    FALLBACK_STRATEGIES,
    V2EvolutionaryPopulation,
)


def _make_pop(agent_type: str, seed: int) -> V2EvolutionaryPopulation:
    pop = V2EvolutionaryPopulation(population_size=8, agent_type=agent_type, seed=seed)
    code = FALLBACK_CLASS_V3 if agent_type == "agent-type2" else FALLBACK_STRATEGIES[0]
    pop.agents = [pop._make_agent(code, i) for i in range(8)]
    for i, a in enumerate(pop.agents):
        a.fitness = float(i)
        a.reputations = {j: 0.25 * (j % 4) for j in range(8)}
    return pop


def test_initial_reputation_is_neutral_for_both_agent_types():
    assert INITIAL_REPUTATION == 0.0
    assert FULL_INITIAL_REPUTATION == 0.0

    for agent_type in ("agent-type1", "agent-type2"):
        pop = _make_pop(agent_type, seed=0)
        fresh = pop._make_agent(pop.agents[0].code, agent_id=99)
        assert fresh.reputations == {99: 0.0}
        assert fresh.get_reputation(12345) == 0.0


def _reproduce(pop: V2EvolutionaryPopulation, path: str) -> None:
    if path == "tournament":
        # Keep default num_eliminate (5) to avoid the tournament
        # while-loop edge case (n_needed close to N with monotone
        # fitness can stall); mock _mutate to avoid LLM calls.
        pop._mutate = lambda parent_code, parent_fitness: parent_code
        pop._select_and_reproduce(next_gen=1)
    else:
        pop.updates_per_gen = 0  # avoid LLM calls
        pop._select_and_reproduce_fermi(next_gen=1)


@pytest.mark.parametrize("agent_type", ["agent-type1", "agent-type2"])
@pytest.mark.parametrize("path", ["tournament", "fermi"])
def test_full_reinstantiation_with_reset(agent_type, path):
    pop = _make_pop(agent_type, seed=42 if path == "tournament" else 7)
    old = list(pop.agents)
    old_ids = {a.agent_id for a in old}
    old_by_id = {a.agent_id: a for a in old}
    n_lineage_before = len(pop._lineage_events)

    _reproduce(pop, path)

    new = pop.agents
    new_by_id = {a.agent_id: a for a in new}
    # 1. population size is preserved
    assert len(new) == 8
    # 2. agent_id set is preserved
    assert set(new_by_id) == old_ids
    # 3. every new object differs from its old counterpart (re-instantiated)
    assert all(new_by_id[aid] is not old_by_id[aid] for aid in old_ids)
    # 4. reputations reset to the initial matrix (only self, at
    #    INITIAL_REPUTATION) — no cross-gen memory
    assert all(
        new_by_id[aid].reputations == {aid: INITIAL_REPUTATION} for aid in old_ids
    )
    # 5. no new lineage events for the untouched-slot rebuilds. The
    #    tournament path DOES add one "mutate" event per eliminated slot
    #    (a genuine bloodline branch) — only the rebuilt survivors must
    #    not add any.
    if path == "tournament":
        assert len(pop._lineage_events) == n_lineage_before + pop.num_eliminate
    else:
        assert len(pop._lineage_events) == n_lineage_before
