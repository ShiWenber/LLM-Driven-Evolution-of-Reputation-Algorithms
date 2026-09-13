"""Extension contracts for game scenarios and evolutionary rules.

The module deliberately contains no population-manager state.  Evolution rules
consume immutable snapshots and produce immutable plans; LLM execution and the
stateful commit are owned by ``V2EvolutionaryPopulation``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, Sequence

from ..evolution_log import (
    ORIGIN_IMITATE,
    ORIGIN_INDEPENDENT_INIT,
    ORIGIN_MUTATE,
)
from .game import DonorGame, resolve_fitness_window
from .prompts import OVERALL_GAME_RULES_PROMPT


@dataclass(frozen=True)
class EvaluationResult:
    cooperation_rate_mean: float
    n_interactions: int
    round_num: int
    payoffs: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "cooperation_rate_mean": self.cooperation_rate_mean,
            "n_interactions": self.n_interactions,
            "round_num": self.round_num,
            "payoffs": list(self.payoffs),
        }


@dataclass(frozen=True)
class AgentSnapshot:
    agent_id: int
    code: str
    fitness: float
    lineage_id: int | None


@dataclass(frozen=True)
class RetainedAgent:
    output_index: int
    snapshot: AgentSnapshot


@dataclass(frozen=True)
class OffspringJob:
    ordinal: int
    output_index: int
    operator: Literal["llm_init", "llm_mutate"]
    mutation_kind: Literal["full", "small"] | None
    preserve_agent_id: int | None
    parent_id: int | None
    parent_lineage_id: int | None
    parent_code: str | None
    parent_fitness: float | None
    origin: str
    birth_gen: int


@dataclass(frozen=True)
class OffspringResult:
    job: OffspringJob
    code: str | None
    attempts: int = 1
    latency_s: float = 0.0
    error_kind: str | None = None


@dataclass(frozen=True)
class GenerationPlan:
    population_size: int
    retained: tuple[RetainedAgent, ...]
    jobs: tuple[OffspringJob, ...]


class GameScenario(Protocol):
    """A replaceable game that evaluates one frozen generation."""

    def evaluate(
        self,
        agents: Sequence[object],
        *,
        generation_seed: int,
        num_rounds: int,
    ) -> EvaluationResult: ...

    def simulation_parameters(
        self, *, num_generations: int, initial_reputation: float
    ) -> Mapping[str, object]: ...

    def overall_rules_prompt(
        self, *, num_generations: int, initial_reputation: float
    ) -> str: ...


class EvolutionRule(Protocol):
    """A pure planner: it must not call the LLM or mutate population state."""

    name: str

    def plan(
        self,
        population: tuple[AgentSnapshot, ...],
        *,
        rng,
        next_gen: int,
        lineage_by_agent_id: Mapping[int, int],
    ) -> GenerationPlan: ...


class OffspringGenerator(Protocol):
    """Execution boundary for LLM, local, cached, or remote generators."""

    def execute_batch(
        self, jobs: Sequence[OffspringJob]
    ) -> Sequence[OffspringResult]: ...


class ReputationPrisonersDilemmaScenario:
    """Default adapter around the ``DonorGame`` engine."""

    name = "reputation_prisoners_dilemma"

    def __init__(
        self,
        *,
        population_size: int,
        benefit: float,
        cost: float,
        observability: str,
        observability_p: float,
        fitness_window_fraction: float | None,
        num_rounds_per_gen: int,
    ) -> None:
        self.population_size = population_size
        self.benefit = benefit
        self.cost = cost
        self.observability = observability
        self.observability_p = observability_p
        self.fitness_window_fraction = fitness_window_fraction
        self.num_rounds_per_gen = num_rounds_per_gen

    def evaluate(
        self,
        agents: Sequence[object],
        *,
        generation_seed: int,
        num_rounds: int,
    ) -> EvaluationResult:
        game = DonorGame(
            population_size=self.population_size,
            benefit=self.benefit,
            cost=self.cost,
            observability=self.observability,
            observability_p=self.observability_p,
            seed=generation_seed,
            fitness_window_fraction=self.fitness_window_fraction,
        )
        game.setup_population(list(agents))
        for agent in agents:
            agent.reset_for_generation()
        game.round_num = 0
        game.payoffs = [0.0] * self.population_size
        game._global_log = []
        game._interaction_deltas = []
        for _ in range(num_rounds):
            game.distribute_observations_and_self_judgments(
                game.play_interaction()["interactions"]
            )
        coop_count = sum(
            1
            for interaction in game._global_log
            for role in ("donor_action", "recipient_action")
            if interaction[role] == "cooperate"
        )
        return EvaluationResult(
            cooperation_rate_mean=coop_count / max(1, 2 * len(game._global_log)),
            n_interactions=len(game._global_log),
            round_num=num_rounds,
            payoffs=tuple(game.get_windowed_fitness()),
        )

    def simulation_parameters(
        self, *, num_generations: int, initial_reputation: float
    ) -> Mapping[str, object]:
        return {
            "population_size": self.population_size,
            "num_rounds_per_gen": self.num_rounds_per_gen,
            "num_generations": num_generations,
            "benefit": self.benefit,
            "cost": self.cost,
            "cc_payoff": self.benefit - self.cost,
            "initial_reputation": initial_reputation,
        }

    def overall_rules_prompt(
        self, *, num_generations: int, initial_reputation: float,
    ) -> str:
        observability = self.observability
        if observability == "private":
            observability_description = (
                "No third-party agent observes another pair's interaction."
            )
        elif observability.startswith("partial"):
            observability_description = (
                "Each third-party agent independently observes each pair's "
                f"interaction with probability {self.observability_p:g}."
            )
        else:
            observability_description = (
                "Every third-party agent observes every pair's interaction."
            )

        rounds = self.num_rounds_per_gen
        total_interactions = rounds
        fraction = self.fitness_window_fraction
        window = resolve_fitness_window(fraction, total_interactions, 1)
        if window is None:
            fitness_window_description = (
                "summed over all joint interactions in the generation"
            )
        else:
            burn_in = total_interactions - window
            fitness_window_description = (
                f"summed over only the final {window} of the generation's "
                f"{total_interactions} joint interactions (the final "
                f"{float(fraction):.0%} of the generation); the first {burn_in} "
                "interactions are burn-in (they still affect observations and "
                "state, but their payoffs do not count)"
            )
        params = dict(
            self.simulation_parameters(
                num_generations=num_generations,
                initial_reputation=initial_reputation,
            )
        )
        params["num_rounds_per_gen"] = rounds
        return OVERALL_GAME_RULES_PROMPT.format(
            **params,
            observability_description=observability_description,
            fitness_window_description=fitness_window_description,
        ).strip()


class TournamentEvolutionRule:
    name = "tournament"

    def __init__(self, *, elite_count: int, num_eliminate: int, tournament_size: int):
        self.elite_count = elite_count
        self.num_eliminate = num_eliminate
        self.tournament_size = tournament_size

    def plan(
        self,
        population: tuple[AgentSnapshot, ...],
        *,
        rng,
        next_gen: int,
        lineage_by_agent_id: Mapping[int, int],
    ) -> GenerationPlan:
        size = len(population)
        ranked = sorted(population, key=lambda agent: (agent.fitness, -agent.agent_id), reverse=True)
        survivors = list(ranked[: self.elite_count])
        survivor_ids = {agent.agent_id for agent in survivors}
        needed = size - self.num_eliminate
        while len(survivors) < needed:
            candidates = rng.sample(ranked, min(self.tournament_size, len(ranked)))
            winner = max(candidates, key=lambda agent: agent.fitness)
            if winner.agent_id not in survivor_ids:
                survivors.append(winner)
                survivor_ids.add(winner.agent_id)
        survivors = sorted(
            survivors, key=lambda agent: (agent.fitness, -agent.agent_id), reverse=True
        )[:needed]
        retained = tuple(
            RetainedAgent(output_index=index, snapshot=agent)
            for index, agent in enumerate(survivors)
        )
        jobs = []
        for ordinal in range(size - needed):
            parent = rng.choice(survivors)
            jobs.append(
                OffspringJob(
                    ordinal=ordinal,
                    output_index=needed + ordinal,
                    operator="llm_mutate",
                    mutation_kind="full",
                    preserve_agent_id=None,
                    parent_id=parent.agent_id,
                    parent_lineage_id=lineage_by_agent_id.get(parent.agent_id),
                    parent_code=parent.code,
                    parent_fitness=parent.fitness,
                    origin=ORIGIN_MUTATE,
                    birth_gen=next_gen,
                )
            )
        return GenerationPlan(size, retained, tuple(jobs))


class FermiEvolutionRule:
    name = "fermi"

    def __init__(self, *, beta: float, mutation_rate: float, updates_per_gen: int):
        self.beta = beta
        self.mutation_rate = mutation_rate
        self.updates_per_gen = updates_per_gen

    def plan(
        self,
        population: tuple[AgentSnapshot, ...],
        *,
        rng,
        next_gen: int,
        lineage_by_agent_id: Mapping[int, int],
    ) -> GenerationPlan:
        size = len(population)
        if size < 2:
            retained = tuple(
                RetainedAgent(index, agent) for index, agent in enumerate(population)
            )
            return GenerationPlan(size, retained, ())
        jobs = []
        updated_indices = set()
        learner_indices = rng.sample(range(size), self.updates_per_gen)
        for learner_index in learner_indices:
            role_model_index = rng.randrange(size - 1)
            if role_model_index >= learner_index:
                role_model_index += 1
            learner = population[learner_index]
            role_model = population[role_model_index]
            difference = role_model.fitness - learner.fitness
            try:
                probability = 1.0 / (1.0 + math.exp(-self.beta * difference))
            except OverflowError:
                probability = 0.0 if difference < 0 else 1.0
            if rng.random() >= probability:
                continue
            if rng.random() < self.mutation_rate:
                operator = "llm_init"
                mutation_kind = None
                parent_id = None
                parent_lineage_id = None
                parent_code = None
                parent_fitness = None
                origin = ORIGIN_INDEPENDENT_INIT
            else:
                operator = "llm_mutate"
                mutation_kind = "small"
                parent_id = role_model.agent_id
                parent_lineage_id = lineage_by_agent_id.get(role_model.agent_id)
                parent_code = role_model.code
                parent_fitness = role_model.fitness
                origin = ORIGIN_IMITATE
            jobs.append(
                OffspringJob(
                    ordinal=len(jobs),
                    output_index=learner_index,
                    operator=operator,
                    mutation_kind=mutation_kind,
                    preserve_agent_id=learner.agent_id,
                    parent_id=parent_id,
                    parent_lineage_id=parent_lineage_id,
                    parent_code=parent_code,
                    parent_fitness=parent_fitness,
                    origin=origin,
                    birth_gen=next_gen,
                )
            )
            updated_indices.add(learner_index)
        retained = tuple(
            RetainedAgent(index, agent)
            for index, agent in enumerate(population)
            if index not in updated_indices
        )
        return GenerationPlan(size, retained, tuple(jobs))
