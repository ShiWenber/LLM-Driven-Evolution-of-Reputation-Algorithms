"""Shared fixed-strategy invasion primitives.

Everything in the invasion experiment is a two-type mixture: one CANDIDATE
strategy against one RESIDENT. A resident is either a canonical norm or another
arbitrary strategy, and both are represented the same way -- as an
``EvolvedSource``. A norm is simply the source whose code is ``BASELINES[norm]``
(see ``norm_source``), so competitor construction and imitation have exactly one
code path regardless of what the two sides are.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.v2_quantitative.agent import QuantitativeAgent
from experiments.v2_quantitative.agent_full import FullAgent, V3StrategyExecutor
from experiments.v2_quantitative.baselines import BASELINES, LEADING_EIGHT
from experiments.v2_quantitative.executor import V2StrategyExecutor
from experiments.v2_quantitative.game import (
    prisoners_dilemma_payoff,
    resolve_fitness_window,
)


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

# Default number of joint actions (pairs) played per generation by the sweep.
# CHANGED 2026-09-13: 1_000 -> 10_000. This is only the DEFAULT for
# ``run_invasion --interactions``. Every result records the value it was
# produced with under ``config.interactions_per_generation``, and
# ``cache_matches`` compares against that recorded value, so raising the
# default does not reinterpret archived runs -- it only (a) changes what a
# new run does when the flag is omitted, and (b) makes the cache correctly
# treat the archived 1000-interaction runs as stale.
#
# With N=100 each generation pairs all 100 agents into 50 pairs per pass, so
# 10_000 interactions = 200 passes = 200 pairings per agent -- 10x the 1000
# interactions the 2026-09 archived sweeps used.
INTERACTIONS_PER_GENERATION = 10_000
NUM_GENERATIONS = 50

# The payoff matrix and the fitness window are both taken from the main
# evolutionary engine (``experiments.v2_quantitative.game``) rather than
# redefined here. The three values below are the protocol constants every
# archived invasion result was produced with; they are recorded facts about
# that data, not live parameters, so changing them would break comparability
# with the archive.
BENEFIT = 2.0
COST = 1.0
FITNESS_WINDOW_FRACTION = 0.2


def resolve_window(
    interactions: int,
    population_size: int,
    fraction: float = FITNESS_WINDOW_FRACTION,
) -> int:
    """Trailing interactions of a generation that count toward fitness.

    Delegates to the main engine's ``resolve_fitness_window`` so the burn-in
    semantics are defined in exactly one place; returns ``interactions`` when
    the window is disabled or covers the whole generation.
    """
    window = resolve_fitness_window(
        fraction, interactions, max(1, population_size // 2)
    )
    return interactions if window is None else window


# The two roles in a mixture. The candidate is the strategy under test; the
# resident is what it is pitted against. KIND_NORM / KIND_RESIDENT distinguish a
# canonical-norm resident from an arbitrary-strategy resident for reporting only;
# neither affects behaviour.
KIND_CANDIDATE = "candidate"
KIND_NORM = "norm"
KIND_RESIDENT = "resident"


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


def norm_source(norm: str) -> EvolvedSource:
    """A canonical norm expressed as an ordinary strategy source.

    Norms used to be a separate competitor kind, which forced competitor
    construction and the imitation copier to branch. Representing a norm as a
    source removes that branch and lets one implementation serve norm residents
    and strategy residents alike.
    """
    if norm not in NORMS:
        raise ValueError(f"Unknown norm: {norm}")
    code = BASELINES[norm]
    return EvolvedSource(
        agent_type="agent-type1",
        path=Path(f"<baseline:{norm}>"),
        agent_id=-1,
        lineage_id=-1,
        root_lineage_id=-1,
        root_family_size=0,
        fitness=0.0,
        code=code,
        code_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
    )


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
    label: str
    source: EvolvedSource
    agent: Any

    @classmethod
    def create(
        cls, agent_id: int, kind: str, label: str, source: EvolvedSource,
    ) -> "Competitor":
        if source.agent_type == "agent-type1":
            agent = QuantitativeAgent(
                agent_id, source.code, executor=V2StrategyExecutor(source.code)
            )
        elif source.agent_type == "agent-type2":
            agent = FullAgent(
                agent_id, V3StrategyExecutor(source.code), code=source.code
            )
        else:
            raise ValueError(f"Unknown source agent type: {source.agent_type}")
        return cls(agent_id, kind, label, source, agent)

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
    """Synchronously copy a sampled role model only when strictly fitter.

    The learner adopts the MODEL's whole identity -- kind, label and source.
    Copying only the kind is correct when every member shares one source, but it
    silently mismatches a slot's declared type with the code it runs as soon as
    the population holds two different strategies.
    """
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
            learner.agent_id, model.kind, model.label, model.source
        )
    return [
        member if member is not old[pos] else Competitor.create(
            member.agent_id, member.kind, member.label, member.source
        )
        for pos, member in enumerate(next_population)
    ]


def _flip_action(action: str, rng: Any, probability: float) -> str:
    if probability <= 0.0:
        return action
    if rng.random() >= probability:
        return action
    return "defect" if action == "cooperate" else "cooperate"


def play_generation_noisy(
    population: list[Competitor],
    rng: Any,
    interactions: int,
    fitness_window_fraction: float = FITNESS_WINDOW_FRACTION,
    action_error: float = 0.0,
    observation_error: float = 0.0,
) -> dict[str, Any]:
    """Play one generation with independent action and perception flips.

    ``fitness_window_fraction`` selects the trailing share of interactions that
    counts toward fitness (see ``resolve_window``); earlier ones are burn-in
    that still updates reputations.
    """
    for agent in population:
        agent.reset_generation_tracking()
    window_start = interactions - resolve_window(
        interactions, len(population), fitness_window_fraction
    )
    payoffs = [0.0] * len(population)
    counted = [0.0] * len(population)
    completed = 0
    cooperation_count = 0
    while completed < interactions:
        order = list(range(len(population)))
        rng.shuffle(order)
        for offset in range(0, len(order) - 1, 2):
            if completed >= interactions:
                break
            first_pos, second_pos = order[offset], order[offset + 1]
            first, second = population[first_pos], population[second_pos]
            first_intended = "cooperate" if first.choose(second.agent_id) else "defect"
            second_intended = "cooperate" if second.choose(first.agent_id) else "defect"
            first_action = _flip_action(first_intended, rng, action_error)
            second_action = _flip_action(second_intended, rng, action_error)
            first_cooperates = first_action == "cooperate"
            second_cooperates = second_action == "cooperate"
            # Payoffs come from the shared matrix definition in the main
            # engine, evaluated at this protocol's (benefit, cost) values.
            first_payoff = prisoners_dilemma_payoff(
                my_cooperates=first_cooperates,
                other_cooperates=second_cooperates,
                benefit=BENEFIT,
                cost=COST,
            )
            second_payoff = prisoners_dilemma_payoff(
                my_cooperates=second_cooperates,
                other_cooperates=first_cooperates,
                benefit=BENEFIT,
                cost=COST,
            )
            payoffs[first_pos] += first_payoff
            payoffs[second_pos] += second_payoff
            if completed >= window_start:
                counted[first_pos] += first_payoff
                counted[second_pos] += second_payoff

            for observer in population:
                seen_first = _flip_action(first_action, rng, observation_error)
                seen_second = _flip_action(second_action, rng, observation_error)
                if observer.agent_id == second.agent_id:
                    observer.observe(
                        second.agent_id, seen_second, first.agent_id, seen_first
                    )
                else:
                    observer.observe(
                        first.agent_id, seen_first, second.agent_id, seen_second
                    )
            cooperation_count += int(first_cooperates) + int(second_cooperates)
            completed += 1
    for pos, agent in enumerate(population):
        agent.fitness = counted[pos]
    return {
        "cooperation_rate": cooperation_count / (2 * completed),
        "fitness": counted,
        "all_interaction_payoffs": payoffs,
    }


# Historical private name, still imported by the legacy runners.
_play_generation_noisy = play_generation_noisy


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
