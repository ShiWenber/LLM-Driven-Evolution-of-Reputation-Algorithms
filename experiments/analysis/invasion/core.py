"""Shared fixed-strategy invasion primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.v2_quantitative.agent import QuantitativeAgent
from experiments.v2_quantitative.agent_full import FullAgent, V3StrategyExecutor
from experiments.v2_quantitative.baselines import BASELINES, LEADING_EIGHT
from experiments.v2_quantitative.executor import V2StrategyExecutor


AGENT_TYPES = ("agent-type1", "agent-type2")
NORMS = (*LEADING_EIGHT, "ALLC", "ALLD")

# The invasion experiment measures a single frequency axis: the candidate
# strategy's share ``x``. A count of ``k`` places the candidate at ``x = k/100``,
# so sweeping ``k = 1..99`` already covers every composition. Whether the
# candidate is itself invadable is a separate question, measured directly by the
# fixation benchmark rather than inferred from this sweep.
#
# Archived sweeps were written before that simplification, when results were
# nested under a direction level (``<label>/<direction>/<norm>/...``). Only this
# one label was ever produced, so it is retained purely so those archived
# results stay readable; nothing new is written under it.
ARCHIVED_DIRECTION_LABEL = "evolved_invades_norm"

INTERACTIONS_PER_GENERATION = 1_000
FITNESS_INTERACTIONS = 200
NUM_GENERATIONS = 50

# The two kinds in a mixed population. The candidate is the strategy under test;
# the norm is the fixed baseline resident it is pitted against.
KIND_CANDIDATE = "candidate"
KIND_NORM = "norm"


@dataclass(frozen=True)
class EvolvedSource:
    agent_type: str
    path: Path
    agent_id: int
    lineage_id: int
    root_lineage_id: int
    root_family_size: int
    fitness: float
    code: str
    code_sha256: str


def root_lineage(lineage_id: int, parent_by_lineage: dict[int, int | None]) -> int:
    root = lineage_id
    seen: set[int] = set()
    while parent_by_lineage.get(root) is not None:
        if root in seen:
            raise ValueError(f"Cycle in lineage graph at {root}")
        seen.add(root)
        root = int(parent_by_lineage[root])
    return root


@dataclass
class Competitor:
    agent_id: int
    kind: str
    norm: str
    source: EvolvedSource
    agent: Any

    @classmethod
    def create(
        cls, agent_id: int, kind: str, norm: str, source: EvolvedSource,
    ) -> "Competitor":
        if kind == KIND_NORM:
            code = BASELINES[norm]
            agent = QuantitativeAgent(agent_id, code, executor=V2StrategyExecutor(code))
        elif source.agent_type == "agent-type1":
            agent = QuantitativeAgent(
                agent_id, source.code, executor=V2StrategyExecutor(source.code)
            )
        elif source.agent_type == "agent-type2":
            agent = FullAgent(
                agent_id, V3StrategyExecutor(source.code), code=source.code
            )
        else:
            raise ValueError(f"Unknown source agent type: {source.agent_type}")
        return cls(agent_id, kind, norm, source, agent)

    @property
    def fitness(self) -> float:
        return float(self.agent.fitness)

    @fitness.setter
    def fitness(self, value: float) -> None:
        self.agent.fitness = float(value)

    def reset_generation_tracking(self) -> None:
        self.agent.reset_for_generation()
        self.fitness = 0.0

    def choose(self, opponent_id: int) -> bool:
        return bool(self.agent.choose(opponent_id))

    def observe(
        self, actor_id: int, actor_action: str,
        recipient_id: int, recipient_action: str,
    ) -> None:
        self.agent.observe_and_judge(
            actor_id, actor_action, recipient_id, recipient_action
        )


def payoff_imitation_update(
    population: list[Competitor], rng: Any, updates: int,
) -> list[Competitor]:
    """Synchronously copy a sampled role model only when strictly fitter."""
    old = population
    next_population = list(old)
    size = len(old)
    for _ in range(updates):
        learner_pos = rng.randrange(size)
        model_pos = rng.randrange(size - 1)
        if model_pos >= learner_pos:
            model_pos += 1
        learner, model = old[learner_pos], old[model_pos]
        if model.fitness <= learner.fitness or learner.kind == model.kind:
            continue
        next_population[learner_pos] = Competitor.create(
            learner.agent_id, model.kind, learner.norm, learner.source
        )
    return [
        member if member is not old[pos] else Competitor.create(
            member.agent_id, member.kind, member.norm, member.source
        )
        for pos, member in enumerate(next_population)
    ]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
