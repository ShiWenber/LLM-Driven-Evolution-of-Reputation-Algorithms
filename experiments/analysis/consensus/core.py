"""Replay support and numeric-agreement metrics for the reputation heatmap.

Scope
-----
This module exists to serve ``plot_private_reputation_matrix.py``.  It does two
things and nothing else:

1. **Replay one generation.**  ``load_run`` reads an evolution run's final
   population, and ``play_generation`` replays one generation of the N-player
   pairwise Prisoner's Dilemma with the same semantics as
   ``experiments/analysis/invasion/run_invasion`` (private
   reputation matrix, per-generation reset, selectable third-party
   observability).  The result is the raw ``[observer, target]`` matrix.
2. **Score that matrix's numeric agreement.**  ``disagreement`` answers
   "do observers put the *same number* on a target?" - the input to the
   heatmap's per-column captions.

Only ``agent-type1`` is supported: its ``observe``/``decide`` functions are the
whole strategy and the engine owns a private scalar reputation matrix.
``agent-type2`` agents keep arbitrary internal state instead, so "the private
reputation matrix" is not well defined for them.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.evolution_log import (
    F_AGENT_ID,
    F_CODE,
    F_FITNESS,
    F_LINEAGE_ID,
    F_ORIGIN,
    F_SELF_REPUTATION,
    K_FINAL_POPULATION,
    load_evolution_json,
)
from experiments.v2_quantitative.agent import QuantitativeAgent
from experiments.v2_quantitative.executor import V2StrategyExecutor

OBSERVABILITY_MODES = ("full", "partial", "private")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinalMember:
    """One member of the recorded final population."""

    agent_id: int
    code: str
    fitness: float
    lineage_id: int
    origin: str
    self_reputation: float

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> FinalMember:
        return cls(
            agent_id=int(record[F_AGENT_ID]),
            code=str(record[F_CODE]),
            fitness=float(record.get(F_FITNESS, 0.0)),
            lineage_id=int(record.get(F_LINEAGE_ID, -1)),
            origin=str(record.get(F_ORIGIN, "")),
            self_reputation=float(record.get(F_SELF_REPUTATION, 0.0)),
        )


@dataclass(frozen=True)
class EvolvedRun:
    """A recorded evolution run, reduced to what a replay needs."""

    path: Path
    label: str
    agent_type: str
    config: dict[str, Any]
    members: list[FinalMember]

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def codes(self) -> list[str]:
        return [member.code for member in self.members]


def load_run(path: Path, label: str | None = None) -> EvolvedRun:
    """Load a run and verify it exposes a private reputation matrix.

    Only ``agent-type1`` is analysable: its ``observe``/``decide`` functions
    are the whole strategy and the engine owns a private scalar reputation
    matrix.  ``agent-type2`` agents keep arbitrary internal state instead, so
    "the private reputation matrix" is not well defined for them.
    """
    path = Path(path)
    data = load_evolution_json(path)
    config = dict(data.get("config") or {})
    agent_type = str(config.get("agent_type", ""))
    if agent_type != "agent-type1":
        raise ValueError(
            f"{path} has agent_type={agent_type!r}; the private-reputation "
            "heatmap is only defined for 'agent-type1'"
        )
    records = data.get(K_FINAL_POPULATION) or []
    if not records:
        raise ValueError(f"No final_population members in {path}")
    members = sorted(
        (FinalMember.from_record(record) for record in records),
        key=lambda member: member.agent_id,
    )
    return EvolvedRun(
        path=path,
        label=label or path.parent.name,
        agent_type=agent_type,
        config=config,
        members=members,
    )


def build_from_codes(codes: Sequence[str], size: int) -> list[QuantitativeAgent]:
    """Instantiate ``size`` agents by cycling through ``codes``."""
    if not codes:
        raise ValueError("codes must be non-empty")
    if size < 2:
        raise ValueError("population size must be at least 2")
    return [
        QuantitativeAgent(position, codes[position % len(codes)],
                          executor=V2StrategyExecutor(codes[position % len(codes)]))
        for position in range(size)
    ]


# ---------------------------------------------------------------------------
# One generation of play
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    """One cooperation decision, with the chooser's pre-decision belief."""

    round_index: int
    chooser_id: int
    target_id: int
    target_reputation: float
    action: str

    @property
    def cooperated(self) -> bool:
        return self.action == "cooperate"


@dataclass
class GenerationOutcome:
    """Everything a matrix analysis needs from one generation."""

    agent_ids: list[int]
    reputation: np.ndarray          # [observer_pos, target_pos], NaN on the diagonal
    observed: np.ndarray            # bool, True where the observer holds an entry
    decisions: list[Decision] = field(default_factory=list)
    cooperation_rate: float = 0.0
    interactions: int = 0
    snapshots: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


def _flip(action: str, rng: random.Random, probability: float) -> str:
    if probability <= 0.0 or rng.random() >= probability:
        return action
    return "defect" if action == "cooperate" else "cooperate"


def _snapshot(
    population: Sequence[QuantitativeAgent], agent_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(population)
    matrix = np.full((n, n), np.nan)
    observed = np.zeros((n, n), dtype=bool)
    for position, agent in enumerate(population):
        for target_position, target_id in enumerate(agent_ids):
            if position == target_position:
                continue
            # Framework semantics: an unseen target reads as the neutral prior.
            matrix[position, target_position] = agent.get_reputation(target_id)
            observed[position, target_position] = target_id in agent.reputations
    return matrix, observed


def play_generation(
    population: Sequence[QuantitativeAgent],
    interactions: int = 1_000,
    rng: random.Random | None = None,
    action_error: float = 0.0,
    observation_error: float = 0.0,
    observability: str = "full",
    observability_p: float = 1.0,
    snapshot_at: Iterable[int] | None = None,
) -> GenerationOutcome:
    """Replay one generation and return the private reputation matrix.

    Mirrors ``play_generation_noisy`` in the invasion core with two
    additions: third-party observability is selectable (``full`` /
    ``partial`` / ``private``) and the private reputation matrix can be
    snapshotted after a given number of interactions.
    """
    if observability not in OBSERVABILITY_MODES:
        raise ValueError(f"observability must be one of {OBSERVABILITY_MODES}")
    if not 0.0 <= observability_p <= 1.0:
        raise ValueError("observability_p must be in [0, 1]")
    if interactions < 1:
        raise ValueError("interactions must be positive")
    rng = rng or random.Random(0)
    snapshot_points = set(snapshot_at or ())
    for agent in population:
        agent.reset_for_generation()

    agent_ids = [agent.agent_id for agent in population]
    decisions: list[Decision] = []
    cooperation_count = 0
    completed = 0
    snapshots: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    while completed < interactions:
        order = list(range(len(population)))
        rng.shuffle(order)
        for offset in range(0, len(order) - 1, 2):
            if completed >= interactions:
                break
            first, second = population[order[offset]], population[order[offset + 1]]

            first_belief = first.get_reputation(second.agent_id)
            second_belief = second.get_reputation(first.agent_id)
            first_intended = "cooperate" if first.choose(second.agent_id) else "defect"
            second_intended = "cooperate" if second.choose(first.agent_id) else "defect"
            decisions.append(Decision(completed, first.agent_id, second.agent_id, first_belief, first_intended))
            decisions.append(Decision(completed, second.agent_id, first.agent_id, second_belief, second_intended))

            first_action = _flip(first_intended, rng, action_error)
            second_action = _flip(second_intended, rng, action_error)
            cooperation_count += (first_action == "cooperate") + (second_action == "cooperate")

            if observability == "private":
                observers = (first, second)
            elif observability == "full":
                observers = population
            else:
                observers = [
                    agent
                    for agent in population
                    if agent is first
                    or agent is second
                    or rng.random() < observability_p
                ]

            for observer in observers:
                seen_first = _flip(first_action, rng, observation_error)
                seen_second = _flip(second_action, rng, observation_error)
                if observer.agent_id == second.agent_id:
                    observer.observe_and_judge(
                        second.agent_id, seen_second, first.agent_id, seen_first
                    )
                else:
                    observer.observe_and_judge(
                        first.agent_id, seen_first, second.agent_id, seen_second
                    )

            completed += 1
            if completed in snapshot_points:
                snapshots[completed] = _snapshot(population, agent_ids)

    if snapshot_points and completed not in snapshots:
        snapshots[completed] = _snapshot(population, agent_ids)

    matrix, observed = _snapshot(population, agent_ids)
    return GenerationOutcome(
        agent_ids=agent_ids,
        reputation=matrix,
        observed=observed,
        decisions=decisions,
        cooperation_rate=cooperation_count / (2 * completed),
        interactions=completed,
        snapshots=snapshots,
    )


# ---------------------------------------------------------------------------
# Numeric agreement:  same value, not merely the same sign
# ---------------------------------------------------------------------------
#
# Two observers can agree perfectly on WHO is good while disagreeing wildly on
# HOW good.  A sign-level judgement is blind to that; this is not.
#
# The metric is the residual norm below, not a pairwise average.  Averaging
# |v_it - v_jt| over pairs looks natural but is a dead end: it is not a norm,
# it is not bounded by construction, and it weights a target by how many
# observers happened to rate it.  Projecting the matrix onto "everyone agrees"
# and taking the RMS residual gives the same intuition with none of that, and
# needs only a per-target mean instead of an O(n^2) pair sweep.


def disagreement(
    reputation: np.ndarray, observed: np.ndarray | None = None,
) -> float:
    """RMS distance from the observed matrix to the nearest consistent one.

    "Everyone agrees" means every observer gives a target the *same* number,
    i.e. each target's column is constant.  The nearest such matrix is the
    least-squares projection: replace each column with its mean over the
    observers who actually rated that target.  ``disagreement`` is the RMS of
    what is left over, which is exactly the within-target standard deviation.

    In reputation units on the same [-1, 1] scale the game uses, so it reads
    directly: ``0.0`` = every observer holds the same number for every target,
    ``1.0`` = the widest split the scale allows.  NaN when nothing was rated.

    One number, not a pair of them, and it deliberately does not say *why*
    observers differ - see the module README for what the metric can and
    cannot separate.
    """
    if reputation.ndim != 2 or reputation.shape[0] != reputation.shape[1]:
        raise ValueError("reputation must be a square matrix")

    live = ~np.isnan(reputation)
    if observed is not None:
        live &= observed
    for i in range(reputation.shape[0]):
        live[i, i] = False
    if not live.any():
        return float("nan")

    # Column means over the live cells only, taken from sums and counts rather
    # than ``np.nanmean`` so that a target nobody rated cannot raise a "mean of
    # empty slice" warning - and its placeholder mean is never read, because
    # every cell in that column is dead too.
    values = np.where(live, reputation, 0.0)
    counts = live.sum(axis=0)
    column_mean = np.divide(
        values.sum(axis=0), counts,
        out=np.zeros(reputation.shape[0], dtype=float),
        where=counts > 0,
    )
    squared = np.where(live, (values - column_mean[None, :]) ** 2, 0.0)
    return float(np.sqrt(squared.sum() / live.sum()))
