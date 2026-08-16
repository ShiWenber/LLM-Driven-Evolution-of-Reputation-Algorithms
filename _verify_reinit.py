"""Quick verification of per-generation full re-instantiation (no LLM calls).

Checks, for BOTH agent_type='v2' and 'v3' and BOTH selection paths:
  1. every agent object is rebuilt (new instance, same slot)
  2. agent_id set is preserved
  3. reputations are RESET to the initial matrix {agent_id: INITIAL_REPUTATION}
  4. no lineage events are recorded for untouched slots
"""
from experiments.v2_quantitative.population import (
    V2EvolutionaryPopulation,
    FALLBACK_STRATEGIES,
    FALLBACK_CLASS_V3,
)
from experiments.v2_quantitative.agent import INITIAL_REPUTATION


def make_pop(agent_type, seed):
    pop = V2EvolutionaryPopulation(population_size=8, agent_type=agent_type, seed=seed)
    code = FALLBACK_CLASS_V3 if agent_type == "v3" else FALLBACK_STRATEGIES[0]
    pop.agents = [pop._make_agent(code, i) for i in range(8)]
    for i, a in enumerate(pop.agents):
        a.fitness = float(i)
        a.reputations = {j: 0.25 * (j % 4) for j in range(8)}
    return pop


def check(agent_type, path):
    pop = make_pop(agent_type, seed=42 if path == "tournament" else 7)
    old = list(pop.agents)
    old_ids = {a.agent_id for a in old}
    n_lineage_before = len(pop._lineage_events)
    if path == "tournament":
        # Keep default num_eliminate (5) to avoid the tournament
        # while-loop edge case (n_needed close to N with monotone
        # fitness can stall); mock _mutate to avoid LLM calls.
        pop._mutate = lambda parent_code, parent_fitness: parent_code
        pop._select_and_reproduce(next_gen=1)
    else:
        pop.updates_per_gen = 0  # avoid LLM calls
        pop._select_and_reproduce_fermi(next_gen=1)
    new = pop.agents
    assert len(new) == 8, f"{agent_type}/{path}: population size changed"
    new_by_id = {a.agent_id: a for a in new}
    assert set(new_by_id) == old_ids, f"{agent_type}/{path}: ids not preserved"
    # Re-instantiated: every new object must differ from its old counterpart.
    old_by_id = {a.agent_id: a for a in old}
    assert all(new_by_id[aid] is not old_by_id[aid] for aid in old_ids), f"{agent_type}/{path}: slot not re-instantiated"
    # Reputations must be reset to the initial matrix (only self,
    # at INITIAL_REPUTATION) — no cross-gen memory.
    assert all(new_by_id[aid].reputations == {aid: INITIAL_REPUTATION} for aid in old_ids), f"{agent_type}/{path}: reputations not reset"
    # No new lineage events for the untouched-slot rebuilds. The
    # tournament path DOES add one "mutate" event per eliminated slot
    # (a genuine bloodline branch) — only the rebuilt survivors must
    # not add any.
    if path == "tournament":
        assert len(pop._lineage_events) == n_lineage_before + pop.num_eliminate, f"{agent_type}/{path}: unexpected lineage events"
    else:
        assert len(pop._lineage_events) == n_lineage_before, f"{agent_type}/{path}: unexpected lineage events"
    print(f"  {agent_type}/{path}: OK (8/8 rebuilt, ids kept, reps reset, no rebuild lineage events)")


if __name__ == "__main__":
    for at in ("v2", "v3"):
        for path in ("tournament", "fermi"):
            check(at, path)
    print("ALL PASS")
