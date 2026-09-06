"""Shared fixed-strategy invasion primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.v2_quantitative.agent import QuantitativeAgent
from experiments.v2_quantitative.agent_full import FullAgent, V3StrategyExecutor
from experiments.v2_quantitative.baselines import BASELINES
from experiments.v2_quantitative.executor import V2StrategyExecutor


AGENT_TYPES = ("agent-type1", "agent-type2")
NORMS = ("IS", "SS", "SJ", "SC", "SH", "IS+", "SS+", "SJ+")
DIRECTIONS = ("evolved_invades_norm", "norm_invades_evolved")
INTERACTIONS_PER_GENERATION = 1_000
FITNESS_INTERACTIONS = 200
NUM_GENERATIONS = 50


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


class LegacyEvaluateAgent:
    """Adapter retained solely for historical evaluate/decide strategies."""

    def __init__(self, agent_id: int, code: str):
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        self._evaluate = namespace.get("evaluate")
        self._decide = namespace.get("decide")
        if not callable(self._evaluate) or not callable(self._decide):
            raise ValueError("Legacy strategy must define evaluate and decide")
        self.agent_id = agent_id
        self.code = code
        self.reputations = {agent_id: 0.0}
        self.fitness = 0.0
        self.total_decisions = 0
        self.cooperations = 0

    def reset_for_generation(self) -> None:
        self.total_decisions = 0
        self.cooperations = 0

    def _reputation(self, agent_id: int) -> float:
        return self.reputations.get(agent_id, 0.0)

    def choose(self, opponent_id: int, round_num: int = 0) -> bool:
        try:
            action = bool(self._decide(
                self._reputation(self.agent_id), self._reputation(opponent_id)
            ))
        except Exception:
            action = False
        self.total_decisions += 1
        self.cooperations += int(action)
        return action

    def observe_and_judge(
        self, donor_id: int, donor_action: str,
        recipient_id: int, recipient_action: str,
    ) -> None:
        my_reputation = self._reputation(self.agent_id)
        for target_id, action in (
            (donor_id, donor_action), (recipient_id, recipient_action)
        ):
            old = self._reputation(target_id)
            try:
                new = float(self._evaluate(old, action, my_reputation))
            except Exception:
                new = old
            self.reputations[target_id] = max(-1.0, min(1.0, new))


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
        if kind == "norm":
            code = BASELINES[norm]
            agent = QuantitativeAgent(agent_id, code, executor=V2StrategyExecutor(code))
        elif source.agent_type == "agent-type1":
            namespace: dict[str, Any] = {}
            exec(source.code, namespace)
            agent = (
                QuantitativeAgent(
                    agent_id, source.code, executor=V2StrategyExecutor(source.code)
                )
                if callable(namespace.get("observe"))
                else LegacyEvaluateAgent(agent_id, source.code)
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
