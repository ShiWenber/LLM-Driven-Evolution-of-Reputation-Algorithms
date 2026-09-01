"""Bidirectional invasion: selected evolved strategies vs Leading Eight.

The runner reuses the project's quantitative agents, strategy executors,
Leading Eight definitions, and interaction mechanics. Selection uses
deterministic imitation of a strictly fitter sampled role model. The
representative evolved strategy for each agent type is selected reproducibly:
choose the largest root-lineage family in the final population, then the
highest-fitness member of that family (agent id breaks ties).

Default formal design:
  2 agent types x 8 norms x 2 directions x 14 initial counts x 3 seeds
  = 1,344 cached/resumable runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments.evolution_log import load_evolution_json
from experiments.v2_quantitative.agent import QuantitativeAgent
from experiments.v2_quantitative.agent_full import FullAgent, V3StrategyExecutor
from experiments.v2_quantitative.baselines import BASELINES
from experiments.v2_quantitative.executor import V2StrategyExecutor

from ..paths import project_root, quantitative_results_dir
from .run_agent2_schmid_invasion import (
    BENEFIT,
    COST,
    FITNESS_INTERACTIONS,
    FIXATION_THRESHOLD,
    INITIAL_REPUTATION,
    INTERACTIONS_PER_GENERATION,
    NUM_GENERATIONS,
    POPULATION_SIZE,
    UPDATES_PER_GENERATION,
    _play_generation,
    _write_json_atomic,
)


AGENT_TYPES = ("agent-type1", "agent-type2")
NORMS = ("IS", "SS", "SJ", "SC", "SH", "IS+", "SS+", "SJ+")
DIRECTIONS = ("evolved_invades_norm", "norm_invades_evolved")
SOURCE_LABELS = {
    "agent-type1": "LLM_agent-type1_fermi_z_v3_g100_1000inter_N16_genreset_seed4",
    "agent-type2": "LLM_v3_fermi_z_v3_g100_1000inter_N16_genreset_seed0",
}


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


def _root_lineage(lineage_id: int, parent_by_lineage: dict[int, int | None]) -> int:
    root = lineage_id
    seen: set[int] = set()
    while parent_by_lineage.get(root) is not None:
        if root in seen:
            raise ValueError(f"Cycle in lineage graph at {root}")
        seen.add(root)
        root = int(parent_by_lineage[root])
    return root


def load_representative(agent_type: str) -> EvolvedSource:
    """Select the representative final survivor using the documented rule."""
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"Unknown agent type: {agent_type}")
    path = quantitative_results_dir() / SOURCE_LABELS[agent_type] / "evolutionary.json"
    data = load_evolution_json(path)
    parents = {
        int(event["lineage_id"]): (
            None
            if event.get("parent_lineage_id") is None
            else int(event["parent_lineage_id"])
        )
        for event in data["lineage_events"]
    }
    members: list[tuple[dict[str, Any], int]] = []
    family_counts: dict[int, int] = {}
    for member in data["final_population"]:
        lineage_id = int(member["lineage_id"])
        root = _root_lineage(lineage_id, parents)
        members.append((member, root))
        family_counts[root] = family_counts.get(root, 0) + 1
    dominant_root = min(family_counts, key=lambda root: (-family_counts[root], root))
    candidates = [member for member, root in members if root == dominant_root]
    winner = min(
        candidates,
        key=lambda member: (-float(member["fitness"]), int(member["agent_id"])),
    )
    code = str(winner["code"])
    return EvolvedSource(
        agent_type=agent_type,
        path=path,
        agent_id=int(winner["agent_id"]),
        lineage_id=int(winner["lineage_id"]),
        root_lineage_id=dominant_root,
        root_family_size=family_counts[dominant_root],
        fitness=float(winner["fitness"]),
        code=code,
        code_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
    )


class LegacyEvaluateAgent:
    """Adapter for the historical agent-type1 evaluate/decide interface."""

    def __init__(self, agent_id: int, code: str):
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        self._evaluate = namespace.get("evaluate")
        self._decide = namespace.get("decide")
        if not callable(self._evaluate) or not callable(self._decide):
            raise ValueError("agent-type1 representative must define evaluate and decide")
        self.agent_id = agent_id
        self.code = code
        self.reputations = {agent_id: INITIAL_REPUTATION}
        self.fitness = 0.0
        self.total_decisions = 0
        self.cooperations = 0

    def reset_for_generation(self) -> None:
        self.total_decisions = 0
        self.cooperations = 0

    def _reputation(self, agent_id: int) -> float:
        return self.reputations.get(agent_id, INITIAL_REPUTATION)

    def choose(self, opponent_id: int, round_num: int = 0) -> bool:
        try:
            action = bool(
                self._decide(
                    self._reputation(self.agent_id),
                    self._reputation(opponent_id),
                )
            )
        except Exception:
            action = False
        self.total_decisions += 1
        self.cooperations += int(action)
        return action

    def observe_and_judge(
        self,
        donor_id: int,
        donor_action: str,
        recipient_id: int,
        recipient_action: str,
    ) -> None:
        my_reputation = self._reputation(self.agent_id)
        for target_id, action in (
            (donor_id, donor_action),
            (recipient_id, recipient_action),
        ):
            old = self._reputation(target_id)
            try:
                new = float(self._evaluate(old, action, my_reputation))
            except Exception:
                new = old
            self.reputations[target_id] = max(-1.0, min(1.0, new))


@dataclass
class Competitor:
    """One fixed-strategy population slot backed by project agent modules."""

    agent_id: int
    kind: str
    norm: str
    source: EvolvedSource
    agent: Any

    @classmethod
    def create(
        cls,
        agent_id: int,
        kind: str,
        norm: str,
        source: EvolvedSource,
        reputations: dict[int, float] | None = None,
    ) -> "Competitor":
        if kind == "norm":
            code = BASELINES[norm]
            agent = QuantitativeAgent(
                agent_id,
                code,
                executor=V2StrategyExecutor(code),
            )
        elif source.agent_type == "agent-type1":
            namespace: dict[str, Any] = {}
            exec(source.code, namespace)
            if callable(namespace.get("observe")):
                agent = QuantitativeAgent(
                    agent_id,
                    source.code,
                    executor=V2StrategyExecutor(source.code),
                )
            else:
                agent = LegacyEvaluateAgent(agent_id, source.code)
        elif source.agent_type == "agent-type2":
            agent = FullAgent(
                agent_id,
                V3StrategyExecutor(source.code),
                code=source.code,
            )
        else:
            raise ValueError(f"Unknown competitor kind/source: {kind}/{source.agent_type}")
        if reputations is not None:
            agent.reputations = dict(reputations)
        return cls(agent_id, kind, norm, source, agent)

    @property
    def fitness(self) -> float:
        return float(self.agent.fitness)

    @fitness.setter
    def fitness(self, value: float) -> None:
        self.agent.fitness = float(value)

    @property
    def reputations(self) -> dict[int, float]:
        return self.agent.reputations

    def reset_generation_tracking(self) -> None:
        self.agent.reset_for_generation()
        self.fitness = 0.0

    def choose(self, opponent_id: int) -> bool:
        return bool(self.agent.choose(opponent_id))

    def observe(
        self,
        actor_id: int,
        actor_action: str,
        recipient_id: int,
        recipient_action: str,
    ) -> None:
        self.agent.observe_and_judge(
            actor_id,
            actor_action,
            recipient_id,
            recipient_action,
        )


def _payoff_imitation_update(
    population: list[Competitor],
    rng: random.Random,
    updates: int,
) -> list[Competitor]:
    """Synchronously and deterministically imitate strictly fitter role models."""
    old = population
    next_population = list(old)
    size = len(old)
    for _ in range(updates):
        learner_pos = rng.randrange(size)
        model_pos = rng.randrange(size - 1)
        if model_pos >= learner_pos:
            model_pos += 1
        learner = old[learner_pos]
        model = old[model_pos]
        difference = model.fitness - learner.fitness
        if difference <= 0 or learner.kind == model.kind:
            continue
        next_population[learner_pos] = Competitor.create(
            agent_id=learner.agent_id,
            kind=model.kind,
            norm=learner.norm,
            source=learner.source,
        )

    # Match V2EvolutionaryPopulation's generation lifecycle exactly: every
    # slot starts the next generation as a fresh agent instance.  Updated
    # slots keep the adopted strategy kind; untouched slots keep their own
    # strategy kind.  Stable slot IDs survive, while private reputations and
    # all strategy-internal state are reset.
    return [
        member
        if member is not old[pos]
        else Competitor.create(
            agent_id=member.agent_id,
            kind=member.kind,
            norm=member.norm,
            source=member.source,
        )
        for pos, member in enumerate(next_population)
    ]


def run_one(
    source: EvolvedSource,
    norm: str,
    direction: str,
    invader_count: int,
    seed: int,
    generations: int = NUM_GENERATIONS,
    interactions: int = INTERACTIONS_PER_GENERATION,
    fitness_interactions: int = FITNESS_INTERACTIONS,
) -> dict[str, Any]:
    if norm not in NORMS or direction not in DIRECTIONS:
        raise ValueError(f"Invalid norm/direction: {norm}/{direction}")
    if not 1 <= invader_count < POPULATION_SIZE:
        raise ValueError("invader_count must be in 1..14")
    rng = random.Random(seed)
    random.seed(1_000_003 + seed)
    invader_kind = "evolved" if direction == "evolved_invades_norm" else "norm"
    resident_kind = "norm" if invader_kind == "evolved" else "evolved"
    invader_slots = set(rng.sample(range(POPULATION_SIZE), invader_count))
    population = [
        Competitor.create(
            slot,
            invader_kind if slot in invader_slots else resident_kind,
            norm,
            source,
        )
        for slot in range(POPULATION_SIZE)
    ]

    trajectory = []
    started = time.perf_counter()
    for generation in range(generations):
        stats = _play_generation(
            population,
            rng,
            interactions=interactions,
            fitness_interactions=fitness_interactions,
        )
        evolved_count = sum(member.kind == "evolved" for member in population)
        norm_count = POPULATION_SIZE - evolved_count
        invader_population = [m for m in population if m.kind == invader_kind]
        resident_population = [m for m in population if m.kind == resident_kind]
        invader_frequency = len(invader_population) / POPULATION_SIZE
        trajectory.append(
            {
                "generation": generation,
                "invader_count": len(invader_population),
                "invader_frequency": invader_frequency,
                "evolved_count": evolved_count,
                "norm_count": norm_count,
                "cooperation_rate": stats["cooperation_rate"],
                "fitness_mean": sum(stats["fitness"]) / POPULATION_SIZE,
                "invader_fitness_mean": (
                    sum(m.fitness for m in invader_population) / len(invader_population)
                    if invader_population else None
                ),
                "resident_fitness_mean": (
                    sum(m.fitness for m in resident_population) / len(resident_population)
                    if resident_population else None
                ),
                "n_interactions": stats["n_interactions"],
                "burn_in_interactions": stats["burn_in_interactions"],
                "fitness_interactions": stats["fitness_interactions"],
            }
        )
        if generation < generations - 1:
            population = _payoff_imitation_update(
                population,
                rng,
                updates=UPDATES_PER_GENERATION,
            )

    final_frequency = trajectory[-1]["invader_frequency"]
    return {
        "schema_version": 1,
        "experiment": "best_evolved_vs_leading_eight_bidirectional_invasion",
        "agent_type": source.agent_type,
        "norm": norm,
        "direction": direction,
        "invader_kind": invader_kind,
        "resident_kind": resident_kind,
        "initial_invader_count": invader_count,
        "seed": seed,
        "config": {
            "population_size": POPULATION_SIZE,
            "num_generations": generations,
            "interactions_per_generation": interactions,
            "burn_in_interactions_per_generation": interactions - fitness_interactions,
            "fitness_interactions_per_generation": fitness_interactions,
            "benefit": BENEFIT,
            "cost": COST,
            "observability": "full",
            "reputation_ownership": "observer-private",
            "selection": "synchronous_deterministic_payoff_imitation",
            "imitation_eligibility": "strictly_higher_fitness_always_copied",
            "generation_lifecycle": "fresh_agent_and_reputation_reset",
            "updates_per_generation": UPDATES_PER_GENERATION,
            "mutation_rate": 0.0,
            "fixation_threshold": FIXATION_THRESHOLD,
        },
        "evolved_source": {
            "path": str(source.path.relative_to(project_root())),
            "agent_id": source.agent_id,
            "lineage_id": source.lineage_id,
            "root_lineage_id": source.root_lineage_id,
            "root_family_size": source.root_family_size,
            "fitness": source.fitness,
            "code_sha256": source.code_sha256,
        },
        "norm_source": {
            "module": "experiments.v2_quantitative.baselines",
            "leading_eight": list(NORMS),
        },
        "trajectory": trajectory,
        "final_invader_frequency": final_frequency,
        "invader_fixed": final_frequency >= FIXATION_THRESHOLD,
        "invader_extinct": final_frequency == 0.0,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _result_path(
    output: Path,
    agent_type: str,
    norm: str,
    direction: str,
    invader_count: int,
    seed: int,
) -> Path:
    return output / agent_type / direction / norm / f"n{invader_count}_seed{seed}" / "invasion.json"


def _write_summary(output: Path, rows: list[dict[str, Any]], sources: dict[str, EvolvedSource]) -> None:
    groups: dict[str, Any] = {}
    for row in rows:
        group = (
            groups.setdefault(row["agent_type"], {})
            .setdefault(row["direction"], {})
            .setdefault(row["norm"], {"runs": 0, "fixations": 0, "extinctions": 0, "final_frequencies": []})
        )
        group["runs"] += 1
        group["fixations"] += int(row["invader_fixed"])
        group["extinctions"] += int(row["invader_extinct"])
        group["final_frequencies"].append(row["final_invader_frequency"])
    for type_group in groups.values():
        for direction_group in type_group.values():
            for group in direction_group.values():
                values = group.pop("final_frequencies")
                group["mean_final_invader_frequency"] = sum(values) / len(values)
    payload = {
        "experiment": "best_evolved_vs_leading_eight_bidirectional_invasion",
        "completed_or_cached_runs": len(rows),
        "selection": "synchronous_deterministic_payoff_imitation",
        "generation_lifecycle": "fresh_agent_and_reputation_reset",
        "sources": {
            agent_type: {
                "agent_id": source.agent_id,
                "root_lineage_id": source.root_lineage_id,
                "root_family_size": source.root_family_size,
                "fitness": source.fitness,
                "code_sha256": source.code_sha256,
            }
            for agent_type, source in sources.items()
        },
        "groups": groups,
        "runs": rows,
    }
    _write_json_atomic(output / "summary.json", payload)


def _values(values: list[int] | None, default: Iterable[int]) -> list[int]:
    result = list(default if values is None else values)
    if not result:
        raise ValueError("At least one value is required")
    return result


def _execute_task(payload: tuple[Any, ...]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Process-pool entry point for one independent invasion run."""
    source, norm, direction, count, seed, generations, interactions, fitness_interactions = payload
    result = run_one(
        source,
        norm,
        direction,
        count,
        seed,
        generations=generations,
        interactions=interactions,
        fitness_interactions=fitness_interactions,
    )
    return (source.agent_type, norm, direction, count, seed), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=quantitative_results_dir() / "invasion" / "best_vs_leading_eight",
    )
    parser.add_argument("--agent-types", nargs="+", choices=AGENT_TYPES, default=list(AGENT_TYPES))
    parser.add_argument("--norms", nargs="+", choices=NORMS, default=list(NORMS))
    parser.add_argument("--directions", nargs="+", choices=DIRECTIONS, default=list(DIRECTIONS))
    parser.add_argument("--invader-counts", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--interactions", type=int, default=INTERACTIONS_PER_GENERATION)
    parser.add_argument("--fitness-interactions", type=int, default=FITNESS_INTERACTIONS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.agent_types = list(AGENT_TYPES)
        args.norms = ["IS"]
        args.directions = list(DIRECTIONS)
        args.invader_counts = [1]
        args.seeds = [0]
        args.generations = 2
        args.interactions = 20
        args.fitness_interactions = 5
        args.output = args.output / "_smoke"

    counts = _values(args.invader_counts, range(1, POPULATION_SIZE))
    seeds = _values(args.seeds, (0, 1, 2))
    if any(count not in range(1, POPULATION_SIZE) for count in counts):
        parser.error("--invader-counts must be in 1..14")
    if not 0 < args.fitness_interactions <= args.interactions:
        parser.error("--fitness-interactions must be in 1..--interactions")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    sources = {agent_type: load_representative(agent_type) for agent_type in args.agent_types}
    tasks = [
        (agent_type, norm, direction, count, seed)
        for agent_type in args.agent_types
        for norm in args.norms
        for direction in args.directions
        for count in counts
        for seed in seeds
    ]
    print("=== Best evolved strategies vs Leading Eight ===", flush=True)
    for source in sources.values():
        print(
            f"{source.agent_type}: agent={source.agent_id}, fitness={source.fitness:g}, "
            f"root={source.root_lineage_id}, family={source.root_family_size}/16, "
            f"sha256={source.code_sha256}",
            flush=True,
        )
    print(
        f"runs={len(tasks)}, N={POPULATION_SIZE}, G={args.generations}, "
        f"interactions={args.interactions}, fitness-window={args.fitness_interactions}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    pending: list[tuple[Any, ...]] = []
    completed = 0

    def record_result(task: tuple[Any, ...], result: dict[str, Any], status: str) -> None:
        nonlocal completed
        agent_type, norm, direction, count, seed = task
        path = _result_path(args.output, agent_type, norm, direction, count, seed)
        if status == "new":
            _write_json_atomic(path, result)
        row = {
            "agent_type": agent_type,
            "norm": norm,
            "direction": direction,
            "initial_invader_count": count,
            "seed": seed,
            "final_invader_frequency": result["final_invader_frequency"],
            "invader_fixed": result["invader_fixed"],
            "invader_extinct": result["invader_extinct"],
            "status": status,
            "path": str(path.relative_to(args.output)),
        }
        rows.append(row)
        completed += 1
        print(
            f"[{completed:04d}/{len(tasks):04d}] {status:6s} {agent_type} "
            f"{direction} {norm} n={count} seed={seed}: "
            f"final={result['final_invader_frequency']:.3f}",
            flush=True,
        )
        _write_summary(args.output, rows, sources)

    for task in tasks:
        agent_type, norm, direction, count, seed = task
        path = _result_path(args.output, agent_type, norm, direction, count, seed)
        if path.exists() and not args.force:
            cached = json.loads(path.read_text(encoding="utf-8"))
            cached_sha = cached.get("evolved_source", {}).get("code_sha256")
            cached_config = cached.get("config", {})
            expected_config = {
                "population_size": POPULATION_SIZE,
                "num_generations": args.generations,
                "interactions_per_generation": args.interactions,
                "burn_in_interactions_per_generation": (
                    args.interactions - args.fitness_interactions
                ),
                "fitness_interactions_per_generation": args.fitness_interactions,
                "updates_per_generation": UPDATES_PER_GENERATION,
                "mutation_rate": 0.0,
                "selection": "synchronous_deterministic_payoff_imitation",
                "imitation_eligibility": "strictly_higher_fitness_always_copied",
                "generation_lifecycle": "fresh_agent_and_reputation_reset",
            }
            cache_matches = cached_sha == sources[agent_type].code_sha256 and all(
                cached_config.get(key) == value
                for key, value in expected_config.items()
            )
            if cache_matches:
                record_result(task, cached, "cached")
                continue
            print(
                f"stale cache {agent_type} {direction} {norm} n={count} seed={seed}: "
                "source, configuration, or generation lifecycle changed",
                flush=True,
            )
            pending.append(
                (
                    sources[agent_type], norm, direction, count, seed,
                    args.generations, args.interactions, args.fitness_interactions,
                )
            )
        else:
            pending.append(
                (
                    sources[agent_type], norm, direction, count, seed,
                    args.generations, args.interactions, args.fitness_interactions,
                )
            )

    if args.workers == 1:
        for payload in pending:
            task, result = _execute_task(payload)
            record_result(task, result, "new")
    elif pending:
        print(f"running {len(pending)} pending tasks with {args.workers} workers", flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_execute_task, payload) for payload in pending]
            for future in concurrent.futures.as_completed(futures):
                task, result = future.result()
                record_result(task, result, "new")
    print(f"=== completed {len(rows)} runs in {time.perf_counter() - started:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
