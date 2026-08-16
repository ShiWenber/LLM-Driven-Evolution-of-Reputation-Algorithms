"""IO helpers: extract per-generation agent populations from evolutionary.json.

Only pure parsing lives here (no matplotlib / sklearn imports), so importing
this module is cheap and side-effect free.

The on-disk format is the shared evolution-log contract
(``experiments.evolution_log``); field names below are the schema constants.
"""
from __future__ import annotations

from experiments.evolution_log import (
    F_AGENT_ID, F_CODE, F_GENERATION, F_LINEAGE_ID, F_ORIGIN,
    F_PARENT_ID, F_PARENT_LINEAGE_ID, F_POPULATION, K_TRAJECTORY,
)


def load_generations(data: dict) -> list[dict]:
    """Parse ``data[K_TRAJECTORY]`` into a list of generation records.

    Each returned record has the shape::

        {F_GENERATION: int, F_POPULATION: [{F_AGENT_ID: int, F_GENERATION: int,
                                            F_CODE: str}, ...]}

    Empty generations are skipped; raises ``ValueError`` if no population at all.
    """
    generations = []
    for gen in sorted(data.get(K_TRAJECTORY, []), key=lambda g: g.get(F_GENERATION, 0)):
        population = []
        for agent in gen.get(F_POPULATION, []):
            code = agent.get(F_CODE, "") or ""
            population.append({
                F_AGENT_ID: int(agent.get(F_AGENT_ID, -1)),
                F_GENERATION: int(gen.get(F_GENERATION, 0)),
                F_CODE: code,
                F_ORIGIN: agent.get(F_ORIGIN),
                F_PARENT_ID: agent.get(F_PARENT_ID),
                F_LINEAGE_ID: agent.get(F_LINEAGE_ID),
                F_PARENT_LINEAGE_ID: agent.get(F_PARENT_LINEAGE_ID),
            })
        if population:
            generations.append({
                F_GENERATION: int(gen.get(F_GENERATION, 0)),
                F_POPULATION: population,
            })
    if not generations:
        raise ValueError("No trajectory population found in evolutionary.json")
    return generations


def merge_generations(
    seed_generations: list[list[dict]],
    *,
    agent_id_offset: int = 10000,
) -> list[dict]:
    """Merge several seeds' generation lists into one combined record.

    Same-index generations across seeds are concatenated into a single
    generation (so gen 0 holds every seed's gen-0 agents), and agent_ids
    are offset per seed to stay unique across seeds. This lets a single
    global clustering run over all seeds' strategies at once.
    """
    if not seed_generations:
        return []
    n_gens = max(len(g) for g in seed_generations)
    merged = []
    for gi in range(n_gens):
        population = []
        for si, gens in enumerate(seed_generations):
            if gi < len(gens):
                offset = si * agent_id_offset
                for agent in gens[gi][F_POPULATION]:
                    a = dict(agent)
                    a[F_AGENT_ID] = int(a.get(F_AGENT_ID, -1)) + offset
                    population.append(a)
        if population:
            merged.append({
                F_GENERATION: gi,
                F_POPULATION: population,
            })
    return merged
