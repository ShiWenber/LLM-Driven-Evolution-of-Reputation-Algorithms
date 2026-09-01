"""N=100 bidirectional invasion sweep over initial invader counts.

The experiment uses deterministic payoff imitation: for each sampled
learner/model pair, the learner copies the model iff the model has strictly
higher realized fitness. No Fermi/logistic acceptance probability is used.
Optional action error flips the executed action after a strategy chooses it;
optional observation error independently flips each action seen by each observer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import time
from pathlib import Path
from typing import Any

from ..paths import quantitative_results_dir
from .run_best_leading_eight_invasion import (
    AGENT_TYPES,
    DIRECTIONS,
    FITNESS_INTERACTIONS,
    INTERACTIONS_PER_GENERATION,
    NORMS,
    NUM_GENERATIONS,
    UPDATES_PER_GENERATION,
    Competitor,
    EvolvedSource,
    _payoff_imitation_update,
    _play_generation,
    _write_json_atomic,
    load_representative,
)


POPULATION_SIZE = 100
DEFAULT_COUNTS = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)
DEFAULT_SEEDS = (0, 1, 2)


def default_output() -> Path:
    return quantitative_results_dir() / "invasion" / "n100_invasion_count_sweep"


def noisy_output() -> Path:
    return quantitative_results_dir() / "invasion" / "n100_noisy_invasion_count_sweep"


def experiment_name(action_error: float, observation_error: float) -> str:
    return (
        "n100_noisy_bidirectional_invasion_count_sweep"
        if action_error or observation_error
        else "n100_bidirectional_invasion_count_sweep"
    )


def _flip_action(action: str, rng: random.Random, probability: float) -> str:
    if probability <= 0.0:
        return action
    if rng.random() >= probability:
        return action
    return "defect" if action == "cooperate" else "cooperate"


def _play_generation_noisy(
    population: list[Competitor],
    rng: random.Random,
    interactions: int,
    fitness_interactions: int,
    action_error: float,
    observation_error: float,
) -> dict[str, Any]:
    """Play one generation with independent action and perception flips."""
    for agent in population:
        agent.reset_generation_tracking()
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
            first_payoff = 2 * int(second_cooperates) - int(first_cooperates)
            second_payoff = 2 * int(first_cooperates) - int(second_cooperates)
            payoffs[first_pos] += first_payoff
            payoffs[second_pos] += second_payoff
            if completed >= interactions - fitness_interactions:
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


def run_one(
    source: EvolvedSource,
    norm: str,
    direction: str,
    invader_count: int,
    seed: int,
    generations: int = NUM_GENERATIONS,
    interactions: int = INTERACTIONS_PER_GENERATION,
    fitness_interactions: int = FITNESS_INTERACTIONS,
    action_error: float = 0.0,
    observation_error: float = 0.0,
) -> dict[str, Any]:
    if norm not in NORMS or direction not in DIRECTIONS:
        raise ValueError(f"Invalid norm/direction: {norm}/{direction}")
    if not 1 <= invader_count < POPULATION_SIZE:
        raise ValueError("invader_count must be in 1..99")

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

    trajectory: list[dict[str, Any]] = []
    started = time.perf_counter()
    for generation in range(generations):
        if action_error or observation_error:
            stats = _play_generation_noisy(
                population, rng, interactions, fitness_interactions,
                action_error, observation_error,
            )
        else:
            stats = _play_generation(
                population, rng, interactions=interactions,
                fitness_interactions=fitness_interactions,
            )
        invaders = [member for member in population if member.kind == invader_kind]
        residents = [member for member in population if member.kind == resident_kind]
        frequency = len(invaders) / POPULATION_SIZE
        trajectory.append(
            {
                "generation": generation,
                "invader_count": len(invaders),
                "invader_frequency": frequency,
                "cooperation_rate": stats["cooperation_rate"],
                "invader_fitness_mean": (
                    sum(member.fitness for member in invaders) / len(invaders)
                    if invaders
                    else None
                ),
                "resident_fitness_mean": (
                    sum(member.fitness for member in residents) / len(residents)
                    if residents
                    else None
                ),
            }
        )
        if generation < generations - 1:
            population = _payoff_imitation_update(
                population, rng, updates=POPULATION_SIZE
            )
            post_update_count = sum(
                member.kind == invader_kind for member in population
            )
            if post_update_count in (0, POPULATION_SIZE):
                trajectory.append(
                    {
                        "generation": generation + 1,
                        "invader_count": post_update_count,
                        "invader_frequency": post_update_count / POPULATION_SIZE,
                        "cooperation_rate": None,
                        "invader_fitness_mean": None,
                        "resident_fitness_mean": None,
                        "absorbed_after_update": True,
                    }
                )
                break

    final_frequency = trajectory[-1]["invader_frequency"]
    return {
        "schema_version": 1,
        "experiment": experiment_name(action_error, observation_error),
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
            "selection": "synchronous_deterministic_payoff_imitation",
            "imitation_eligibility": "strictly_higher_fitness_always_copied",
            "updates_per_generation": POPULATION_SIZE,
            "mutation_rate": 0.0,
            "generation_lifecycle": "fresh_agent_and_reputation_reset",
            "fixation_threshold": 1.0,
            "absorbing_state_early_stop": True,
            "action_error_probability": action_error,
            "observation_error_probability": observation_error,
        },
        "evolved_source": {
            "path": str(source.path),
            "agent_id": source.agent_id,
            "root_lineage_id": source.root_lineage_id,
            "fitness": source.fitness,
            "code_sha256": source.code_sha256,
        },
        "trajectory": trajectory,
        "final_invader_frequency": final_frequency,
        "invader_fixed": final_frequency == 1.0,
        "invader_extinct": final_frequency == 0.0,
        "elapsed_seconds": time.perf_counter() - started,
    }


def result_path(
    output: Path,
    agent_type: str,
    direction: str,
    norm: str,
    count: int,
    seed: int,
) -> Path:
    return output / agent_type / direction / norm / f"n{count}_seed{seed}" / "invasion.json"


def cache_matches(
    result: dict[str, Any], source: EvolvedSource, generations: int,
    interactions: int, fitness_interactions: int,
    action_error: float, observation_error: float,
) -> bool:
    config = result.get("config", {})
    expected = {
        "population_size": POPULATION_SIZE,
        "num_generations": generations,
        "interactions_per_generation": interactions,
        "fitness_interactions_per_generation": fitness_interactions,
        "selection": "synchronous_deterministic_payoff_imitation",
        "updates_per_generation": POPULATION_SIZE,
        "generation_lifecycle": "fresh_agent_and_reputation_reset",
        "absorbing_state_early_stop": True,
        "action_error_probability": action_error,
        "observation_error_probability": observation_error,
    }
    return (
        result.get("evolved_source", {}).get("code_sha256") == source.code_sha256
        and all(config.get(key) == value for key, value in expected.items())
    )


def execute(payload: tuple[Any, ...]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    source, norm, direction, count, seed, generations, interactions, fitness, action_error, observation_error = payload
    result = run_one(
        source, norm, direction, count, seed,
        generations=generations,
        interactions=interactions,
        fitness_interactions=fitness,
        action_error=action_error,
        observation_error=observation_error,
    )
    return (source.agent_type, norm, direction, count, seed), result


def write_summary(
    output: Path, rows: list[dict[str, Any]], sources: dict[str, EvolvedSource],
    counts: list[int], seeds: list[int], action_error: float,
    observation_error: float,
) -> None:
    groups: dict[str, Any] = {}
    for row in rows:
        group = (
            groups.setdefault(row["agent_type"], {})
            .setdefault(row["direction"], {})
            .setdefault(row["norm"], {})
            .setdefault(
                str(row["initial_invader_count"]),
                {"runs": 0, "fixations": 0, "extinctions": 0, "final_frequencies": []},
            )
        )
        group["runs"] += 1
        group["fixations"] += int(row["invader_fixed"])
        group["extinctions"] += int(row["invader_extinct"])
        group["final_frequencies"].append(row["final_invader_frequency"])
    for type_group in groups.values():
        for direction_group in type_group.values():
            for norm_group in direction_group.values():
                for group in norm_group.values():
                    values = group.pop("final_frequencies")
                    group["mean_final_invader_frequency"] = sum(values) / len(values)
    _write_json_atomic(
        output / "summary.json",
        {
            "experiment": experiment_name(action_error, observation_error),
            "completed_or_cached_runs": len(rows),
            "population_size": POPULATION_SIZE,
            "initial_invader_counts": counts,
            "seeds": seeds,
            "action_error_probability": action_error,
            "observation_error_probability": observation_error,
            "selection": "synchronous_deterministic_payoff_imitation",
        "generation_lifecycle": "fresh_agent_and_reputation_reset",
        "absorbing_state_early_stop": True,
            "sources": {
                key: {"agent_id": value.agent_id, "code_sha256": value.code_sha256}
                for key, value in sources.items()
            },
            "groups": groups,
            "runs": rows,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--agent-types", nargs="+", choices=AGENT_TYPES, default=list(AGENT_TYPES))
    parser.add_argument("--norms", nargs="+", choices=NORMS, default=list(NORMS))
    parser.add_argument("--directions", nargs="+", choices=DIRECTIONS, default=list(DIRECTIONS))
    parser.add_argument("--invader-counts", nargs="+", type=int, default=list(DEFAULT_COUNTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--interactions", type=int, default=INTERACTIONS_PER_GENERATION)
    parser.add_argument("--fitness-interactions", type=int, default=FITNESS_INTERACTIONS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--action-error", type=float, default=0.0)
    parser.add_argument("--observation-error", type=float, default=0.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.action_error <= 1.0:
        parser.error("--action-error must be in [0, 1]")
    if not 0.0 <= args.observation_error <= 1.0:
        parser.error("--observation-error must be in [0, 1]")
    if args.output is None:
        args.output = (
            noisy_output()
            if args.action_error or args.observation_error
            else default_output()
        )
    if args.smoke:
        args.agent_types = list(AGENT_TYPES)
        args.norms = ["IS"]
        args.directions = list(DIRECTIONS)
        args.invader_counts = [1, 50]
        args.seeds = [0]
        args.generations = 2
        args.interactions = 20
        args.fitness_interactions = 5
        args.output = args.output / "_smoke"
    if any(not 1 <= count < POPULATION_SIZE for count in args.invader_counts):
        parser.error("--invader-counts must be in 1..99")
    if not 0 < args.fitness_interactions <= args.interactions:
        parser.error("--fitness-interactions must be in 1..--interactions")

    sources = {kind: load_representative(kind) for kind in args.agent_types}
    tasks = [
        (kind, norm, direction, count, seed)
        for kind in args.agent_types
        for norm in args.norms
        for direction in args.directions
        for count in args.invader_counts
        for seed in args.seeds
    ]
    rows: list[dict[str, Any]] = []
    pending: list[tuple[Any, ...]] = []
    task_by_key = {task: task for task in tasks}
    started = time.perf_counter()
    print(f"=== N=100 invasion-count sweep: {len(tasks)} runs ===", flush=True)

    def record(task: tuple[Any, ...], result: dict[str, Any], status: str) -> None:
        kind, norm, direction, count, seed = task
        path = result_path(args.output, kind, direction, norm, count, seed)
        if status in ("new", "normalized"):
            _write_json_atomic(path, result)
        rows.append(
            {
                "agent_type": kind,
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
        )
        if len(rows) % 25 == 0 or len(rows) == len(tasks):
            print(f"[{len(rows):04d}/{len(tasks):04d}] {status}", flush=True)

    for task in tasks:
        kind, norm, direction, count, seed = task
        path = result_path(args.output, kind, direction, norm, count, seed)
        if path.exists() and not args.force:
            result = json.loads(path.read_text(encoding="utf-8"))
            if cache_matches(
                result, sources[kind], args.generations,
                args.interactions, args.fitness_interactions,
                args.action_error, args.observation_error,
            ):
                needs_normalization = (
                    result.get("config", {}).get("fixation_threshold") != 1.0
                    or result.get("invader_fixed")
                    != (result["final_invader_frequency"] == 1.0)
                )
                if needs_normalization:
                    result["config"]["fixation_threshold"] = 1.0
                    result["invader_fixed"] = result["final_invader_frequency"] == 1.0
                record(task, result, "normalized" if needs_normalization else "cached")
                continue
        pending.append(
            (sources[kind], norm, direction, count, seed, args.generations,
             args.interactions, args.fitness_interactions,
             args.action_error, args.observation_error)
        )

    if pending:
        print(f"running {len(pending)} pending tasks with {args.workers} workers", flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(execute, payload): payload for payload in pending}
            for future in concurrent.futures.as_completed(futures):
                key, result = future.result()
                record(task_by_key[key], result, "new")
    write_summary(
        args.output, rows, sources, list(args.invader_counts), list(args.seeds),
        args.action_error, args.observation_error,
    )
    print(f"=== completed in {time.perf_counter() - started:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
