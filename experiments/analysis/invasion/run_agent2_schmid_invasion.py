"""Bidirectional invasion experiments: evolved agent 2 vs Schmid L1/L2/L7/L8.

This runner matches the production quantitative experiment where it matters:

* symmetric two-player PD, benefit=2 and cost=1;
* N=15 and full observation with observer-private reputation dictionaries;
* exactly 1,000 pair interactions per generation;
* the first 800 interactions update strategy state/reputations but do not
  contribute to selection fitness; only the final 200 interactions count;
* synchronous Fermi imitation, beta=5, 15 update opportunities/generation;
* no LLM calls and no mutation: the experiment compares two fixed strategies.

The four norms are the robust quantitative-assessment norms identified by
Schmid et al. (2023), Supporting Information section 2: L1, L2, L7, L8.
Their canonical third-order action/assessment rules are implemented directly,
rather than mapping them onto the repository's simplified IS/SS/... labels.

By default the full design runs both directions, n=1..14, seeds=0,1,2:
4 norms x 2 directions x 14 initial invader counts x 3 seeds = 336 runs.

Use ``--smoke`` for a tiny non-production validation run. Existing completed
JSON files are skipped, so the full experiment can be resumed safely.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..paths import project_root, quantitative_results_dir, invasion_results_dir, evolution_json_path
from experiments.evolution_log import (
    F_AGENT_ID, F_CODE, F_GENERATION, F_POPULATION, K_TRAJECTORY,
    load_evolution_json,
)

NORMS = ("L1", "L2", "L7", "L8")
DIRECTIONS = ("agent2_invades_norm", "norm_invades_agent2")

POPULATION_SIZE = 15
INTERACTIONS_PER_GENERATION = 1_000
FITNESS_INTERACTIONS = 200
BURN_IN_INTERACTIONS = INTERACTIONS_PER_GENERATION - FITNESS_INTERACTIONS
NUM_GENERATIONS = 50
BENEFIT = 2.0
COST = 1.0
FERMI_BETA = 5.0
UPDATES_PER_GENERATION = 15
REPUTATION_STEP = 1.0 / 3.0
INITIAL_REPUTATION = 0.1
FIXATION_THRESHOLD = 0.9


def default_source_json() -> Path:
    """Locate the production evolutionary run used as Agent 2's source."""
    return evolution_json_path(
        quantitative_results_dir(), "LLM_v3_fermi_z_v3_g100_1000inter", 2
    )


def default_output() -> Path:
    """Locate the default output directory for the invasion sweep."""
    return invasion_results_dir()


def _load_agent2_code(source_json: Path) -> tuple[str, int]:
    """Load agent_id=2 from the numerically last saved generation."""
    data = load_evolution_json(source_json)
    last = max(data[K_TRAJECTORY], key=lambda item: int(item[F_GENERATION]))
    matches = [
        member for member in last[F_POPULATION] if int(member[F_AGENT_ID]) == 2
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one agent_id=2 in generation {last[F_GENERATION]}, "
            f"found {len(matches)}"
        )
    return matches[0][F_CODE], int(last[F_GENERATION])


def _compile_agent2(code: str) -> type:
    namespace: dict[str, Any] = {}
    exec(code, namespace)
    cls = namespace.get("LLMAgent")
    if not isinstance(cls, type):
        raise RuntimeError("Agent 2 code did not define LLMAgent")
    smoke = cls(0)
    for method in ("decide", "observe"):
        if not callable(getattr(smoke, method, None)):
            raise RuntimeError(f"Agent 2 LLMAgent is missing {method}()")
    return cls


AGENT2_SOURCE_JSON = default_source_json()
AGENT2_CODE, AGENT2_SOURCE_GENERATION = _load_agent2_code(AGENT2_SOURCE_JSON)
AGENT2_CODE_SHA256 = hashlib.sha256(AGENT2_CODE.encode("utf-8")).hexdigest()
Agent2Brain = _compile_agent2(AGENT2_CODE)


def _is_good(value: float) -> bool:
    return value > 0.0


def _norm_action(norm: str, my_reputation: float, opponent_reputation: float) -> bool:
    """Canonical action rules for L1, L2, L7, and L8.

    All four cooperate with a good recipient and a good donor defects against a
    bad recipient. A bad donor cooperates with a bad recipient only in L1/L2;
    L7/L8 prescribe defection in that state.
    """
    if _is_good(opponent_reputation):
        return True
    if _is_good(my_reputation):
        return False
    return norm in ("L1", "L2")


def _norm_assesses_good(
    norm: str,
    actor_reputation: float,
    recipient_reputation: float,
    actor_action: str,
) -> bool:
    """Canonical eight-case assessment table for Schmid's robust norms."""
    actor_good = _is_good(actor_reputation)
    recipient_good = _is_good(recipient_reputation)
    cooperated = actor_action == "cooperate"

    if cooperated and recipient_good:
        return True
    if not cooperated and recipient_good:
        return False
    if not cooperated and not recipient_good:
        # For L1/L2/L7/L8, a bad-recipient defection is good only when the
        # actor was previously good. A bad actor continues to be assessed bad.
        return actor_good

    # Cooperation with a bad recipient distinguishes the four norms.
    if norm == "L1":
        return True
    if norm == "L2":
        return not actor_good
    if norm == "L7":
        return actor_good
    if norm == "L8":
        return False
    raise ValueError(f"Unknown norm: {norm}")


@dataclass
class Competitor:
    """One population slot containing either agent 2 or a fixed social norm."""

    agent_id: int
    kind: str
    norm: str
    reputations: dict[int, float]
    brain: Any | None = None
    fitness: float = 0.0
    decisions: int = 0
    cooperations: int = 0

    @classmethod
    def create(
        cls,
        agent_id: int,
        kind: str,
        norm: str,
        reputations: dict[int, float] | None = None,
    ) -> "Competitor":
        if kind not in ("agent2", "norm"):
            raise ValueError(f"Unknown competitor kind: {kind}")
        if norm not in NORMS:
            raise ValueError(f"Unknown norm: {norm}")
        return cls(
            agent_id=agent_id,
            kind=kind,
            norm=norm,
            reputations=(
                dict(reputations)
                if reputations is not None
                else {agent_id: INITIAL_REPUTATION}
            ),
            brain=Agent2Brain(agent_id) if kind == "agent2" else None,
        )

    def reset_generation_tracking(self) -> None:
        self.fitness = 0.0
        self.decisions = 0
        self.cooperations = 0

    def reputation_of(self, target_id: int) -> float:
        return self.reputations.get(target_id, INITIAL_REPUTATION)

    def choose(self, opponent_id: int) -> bool:
        if self.kind == "agent2":
            self.brain._ctx_opponent_id = opponent_id
            try:
                action = bool(self.brain.decide())
            except Exception:
                # Matches FullAgent's production fallback for decide failures.
                action = False
        else:
            action = _norm_action(
                self.norm,
                self.reputation_of(self.agent_id),
                self.reputation_of(opponent_id),
            )
        self.decisions += 1
        self.cooperations += int(action)
        return action

    def observe(
        self,
        actor_id: int,
        actor_action: str,
        recipient_id: int,
        recipient_action: str,
    ) -> None:
        if self.kind == "agent2":
            try:
                self.brain.observe(
                    actor_id,
                    actor_action,
                    recipient_id,
                    recipient_action,
                )
            except Exception:
                pass
            return

        # The norm independently assesses both active players. Each player's
        # counterpart is the recipient/context for that assessment.
        self._assess(actor_id, actor_action, recipient_id)
        self._assess(recipient_id, recipient_action, actor_id)

    def _assess(self, actor_id: int, actor_action: str, recipient_id: int) -> None:
        actor_rep = self.reputation_of(actor_id)
        recipient_rep = self.reputation_of(recipient_id)
        good = _norm_assesses_good(
            self.norm,
            actor_rep,
            recipient_rep,
            actor_action,
        )
        delta = REPUTATION_STEP if good else -REPUTATION_STEP
        self.reputations[actor_id] = max(-1.0, min(1.0, actor_rep + delta))


def _observe_pair(
    population: list[Competitor],
    first: Competitor,
    first_action: str,
    second: Competitor,
    second_action: str,
) -> None:
    """Use the same self/third-party orientation as V2DonorGame."""
    first.observe(
        first.agent_id,
        first_action,
        second.agent_id,
        second_action,
    )
    second.observe(
        second.agent_id,
        second_action,
        first.agent_id,
        first_action,
    )
    for observer in population:
        if observer.agent_id in (first.agent_id, second.agent_id):
            continue
        observer.observe(
            first.agent_id,
            first_action,
            second.agent_id,
            second_action,
        )


def _play_generation(
    population: list[Competitor],
    rng: random.Random,
    interactions: int,
    fitness_interactions: int,
) -> dict[str, Any]:
    for agent in population:
        agent.reset_generation_tracking()

    payoffs = [0.0] * len(population)
    counted_payoffs = [0.0] * len(population)
    action_count = 0
    cooperation_count = 0
    completed = 0

    while completed < interactions:
        order = list(range(len(population)))
        rng.shuffle(order)
        for offset in range(0, len(order) - 1, 2):
            if completed >= interactions:
                break
            first_pos = order[offset]
            second_pos = order[offset + 1]
            first = population[first_pos]
            second = population[second_pos]

            first_cooperates = first.choose(second.agent_id)
            second_cooperates = second.choose(first.agent_id)
            first_action = "cooperate" if first_cooperates else "defect"
            second_action = "cooperate" if second_cooperates else "defect"

            first_payoff = (
                BENEFIT * int(second_cooperates) - COST * int(first_cooperates)
            )
            second_payoff = (
                BENEFIT * int(first_cooperates) - COST * int(second_cooperates)
            )
            payoffs[first_pos] += first_payoff
            payoffs[second_pos] += second_payoff
            if completed >= interactions - fitness_interactions:
                counted_payoffs[first_pos] += first_payoff
                counted_payoffs[second_pos] += second_payoff

            _observe_pair(
                population,
                first,
                first_action,
                second,
                second_action,
            )
            action_count += 2
            cooperation_count += int(first_cooperates) + int(second_cooperates)
            completed += 1

    for pos, agent in enumerate(population):
        agent.fitness = counted_payoffs[pos]

    return {
        "n_interactions": completed,
        "burn_in_interactions": interactions - fitness_interactions,
        "fitness_interactions": fitness_interactions,
        "cooperation_rate": cooperation_count / action_count,
        "fitness": counted_payoffs,
        "all_interaction_payoffs": payoffs,
    }


def _fermi_update(
    population: list[Competitor],
    rng: random.Random,
    beta: float,
    updates: int,
) -> list[Competitor]:
    """Synchronous fixed-strategy Fermi imitation without mutation."""
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
        try:
            probability = 1.0 / (1.0 + math.exp(-beta * difference))
        except OverflowError:
            probability = 1.0 if difference > 0 else 0.0
        if rng.random() >= probability or learner.kind == model.kind:
            continue

        # Match the main engine's replacement semantics: the learner's stable
        # slot/id and private reputation view survive; the adopted strategy's
        # internal brain starts fresh. There are no LLM calls or mutations.
        next_population[learner_pos] = Competitor.create(
            agent_id=learner.agent_id,
            kind=model.kind,
            norm=learner.norm,
            reputations=learner.reputations,
        )

    return next_population


def _kind_counts(population: Iterable[Competitor]) -> dict[str, int]:
    counts = {"agent2": 0, "norm": 0}
    for agent in population:
        counts[agent.kind] += 1
    return counts


def run_one(
    norm: str,
    direction: str,
    invader_count: int,
    seed: int,
    generations: int = NUM_GENERATIONS,
    interactions: int = INTERACTIONS_PER_GENERATION,
    fitness_interactions: int = FITNESS_INTERACTIONS,
) -> dict[str, Any]:
    if norm not in NORMS:
        raise ValueError(f"Unknown norm: {norm}")
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction: {direction}")
    if not 1 <= invader_count < POPULATION_SIZE:
        raise ValueError("invader_count must be in 1..14")
    if not 0 < fitness_interactions <= interactions:
        raise ValueError("fitness_interactions must be in 1..interactions")

    # One RNG controls matching and Fermi sampling. Agent 2's evolved code uses
    # Python's module-global random generator, so seed it separately but
    # deterministically to make each run reproducible.
    rng = random.Random(seed)
    random.seed(1_000_003 + seed)

    invader_kind = "agent2" if direction == "agent2_invades_norm" else "norm"
    resident_kind = "norm" if invader_kind == "agent2" else "agent2"
    invader_slots = set(rng.sample(range(POPULATION_SIZE), invader_count))
    population = [
        Competitor.create(
            agent_id=slot,
            kind=invader_kind if slot in invader_slots else resident_kind,
            norm=norm,
        )
        for slot in range(POPULATION_SIZE)
    ]

    trajectory: list[dict[str, Any]] = []
    started = time.perf_counter()
    for generation in range(generations):
        stats = _play_generation(
            population,
            rng,
            interactions=interactions,
            fitness_interactions=fitness_interactions,
        )
        counts = _kind_counts(population)
        invader_fitness = [
            member.fitness for member in population if member.kind == invader_kind
        ]
        resident_fitness = [
            member.fitness for member in population if member.kind == resident_kind
        ]
        trajectory.append(
            {
                "generation": generation,
                "invader_count": counts[invader_kind],
                "invader_frequency": counts[invader_kind] / POPULATION_SIZE,
                "agent2_count": counts["agent2"],
                "norm_count": counts["norm"],
                "cooperation_rate": stats["cooperation_rate"],
                "fitness_mean": sum(stats["fitness"]) / POPULATION_SIZE,
                "invader_fitness_mean": (
                    sum(invader_fitness) / len(invader_fitness)
                    if invader_fitness
                    else None
                ),
                "resident_fitness_mean": (
                    sum(resident_fitness) / len(resident_fitness)
                    if resident_fitness
                    else None
                ),
                "n_interactions": stats["n_interactions"],
                "burn_in_interactions": stats["burn_in_interactions"],
                "fitness_interactions": stats["fitness_interactions"],
            }
        )
        if generation < generations - 1:
            population = _fermi_update(
                population,
                rng,
                beta=FERMI_BETA,
                updates=UPDATES_PER_GENERATION,
            )

    final_frequency = trajectory[-1]["invader_frequency"]
    return {
        "schema_version": 1,
        "experiment": "agent2_schmid_quantitative_bidirectional_invasion",
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
            "burn_in_interactions_per_generation": interactions
            - fitness_interactions,
            "fitness_interactions_per_generation": fitness_interactions,
            "benefit": BENEFIT,
            "cost": COST,
            "observability": "full",
            "reputation_ownership": "observer-private",
            "reputation_range": [-1.0, 1.0],
            "initial_reputation": INITIAL_REPUTATION,
            "reputation_step": REPUTATION_STEP,
            "selection": "synchronous_fixed-strategy_fermi_imitation",
            "fermi_beta": FERMI_BETA,
            "updates_per_generation": UPDATES_PER_GENERATION,
            "mutation_rate": 0.0,
            "fixation_threshold": FIXATION_THRESHOLD,
        },
        "agent2_source": {
            "path": str(AGENT2_SOURCE_JSON.relative_to(project_root())),
            "generation": AGENT2_SOURCE_GENERATION,
            "agent_id": 2,
            "code_length": len(AGENT2_CODE),
            "code_sha256": AGENT2_CODE_SHA256,
        },
        "norm_source": {
            "paper": "Schmid et al., Nature Communications 14, 2086 (2023)",
            "robust_norms": list(NORMS),
            "doi": "10.1038/s41467-023-37817-x",
        },
        "trajectory": trajectory,
        "final_invader_frequency": final_frequency,
        "invader_fixed": final_frequency >= FIXATION_THRESHOLD,
        "invader_extinct": final_frequency == 0.0,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _result_path(
    output: Path,
    norm: str,
    direction: str,
    invader_count: int,
    seed: int,
) -> Path:
    return (
        output
        / direction
        / norm
        / f"n{invader_count}_seed{seed}"
        / "invasion.json"
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_summary(output: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    per_n: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for row in rows:
        direction_group = grouped.setdefault(row["direction"], {})
        norm_group = direction_group.setdefault(
            row["norm"],
            {"runs": 0, "fixations": 0, "extinctions": 0, "final_frequencies": []},
        )
        norm_group["runs"] += 1
        norm_group["fixations"] += int(row["invader_fixed"])
        norm_group["extinctions"] += int(row["invader_extinct"])
        norm_group["final_frequencies"].append(row["final_invader_frequency"])

        n_group = (
            per_n.setdefault(row["direction"], {})
            .setdefault(row["norm"], {})
            .setdefault(
                str(row["initial_invader_count"]),
                {
                    "runs": 0,
                    "fixations": 0,
                    "extinctions": 0,
                    "final_frequencies_by_seed": {},
                },
            )
        )
        n_group["runs"] += 1
        n_group["fixations"] += int(row["invader_fixed"])
        n_group["extinctions"] += int(row["invader_extinct"])
        n_group["final_frequencies_by_seed"][str(row["seed"])] = row[
            "final_invader_frequency"
        ]

    for direction_group in grouped.values():
        for norm_group in direction_group.values():
            frequencies = norm_group.pop("final_frequencies")
            norm_group["mean_final_invader_frequency"] = sum(frequencies) / len(
                frequencies
            )

    payload = {
        "experiment": "agent2_schmid_quantitative_bidirectional_invasion",
        "completed_or_cached_runs": len(rows),
        "agent2_code_sha256": AGENT2_CODE_SHA256,
        "groups": grouped,
        "per_initial_invader_count": per_n,
        "runs": rows,
    }
    _write_json_atomic(output / "summary.json", payload)


def _parse_int_range(values: list[int] | None, default: Iterable[int]) -> list[int]:
    result = list(default if values is None else values)
    if not result:
        raise ValueError("At least one integer value is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--norms", nargs="+", choices=NORMS, default=list(NORMS))
    parser.add_argument(
        "--directions", nargs="+", choices=DIRECTIONS, default=list(DIRECTIONS)
    )
    parser.add_argument("--invader-counts", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument(
        "--interactions", type=int, default=INTERACTIONS_PER_GENERATION
    )
    parser.add_argument(
        "--fitness-interactions", type=int, default=FITNESS_INTERACTIONS
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run L1, both directions, n=1, seed=0, 2 generations, 20+5 interactions",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run and overwrite completed JSON files"
    )
    args = parser.parse_args()

    if args.smoke:
        args.norms = ["L1"]
        args.directions = list(DIRECTIONS)
        args.invader_counts = [1]
        args.seeds = [0]
        args.generations = 2
        args.interactions = 20
        args.fitness_interactions = 5
        args.output = args.output / "_smoke"

    invader_counts = _parse_int_range(args.invader_counts, range(1, 15))
    seeds = _parse_int_range(args.seeds, (0, 1, 2))
    for count in invader_counts:
        if count not in range(1, 15):
            parser.error(f"invader count must be in 1..14, got {count}")
    if not 0 < args.fitness_interactions <= args.interactions:
        parser.error("--fitness-interactions must be in 1..--interactions")

    tasks = [
        (norm, direction, count, seed)
        for norm in args.norms
        for direction in args.directions
        for count in invader_counts
        for seed in seeds
    ]
    print("=== Agent 2 vs Schmid L1/L2/L7/L8 invasion ===", flush=True)
    print(f"source: generation={AGENT2_SOURCE_GENERATION}, sha256={AGENT2_CODE_SHA256}", flush=True)
    print(
        f"runs={len(tasks)}, N={POPULATION_SIZE}, G={args.generations}, "
        f"interactions={args.interactions}, "
        f"burn-in={args.interactions - args.fitness_interactions}, "
        f"fitness-window={args.fitness_interactions}",
        flush=True,
    )
    print(f"output={args.output}", flush=True)

    rows: list[dict[str, Any]] = []
    overall_started = time.perf_counter()
    for index, (norm, direction, count, seed) in enumerate(tasks, start=1):
        path = _result_path(args.output, norm, direction, count, seed)
        if path.exists() and not args.force:
            result = json.loads(path.read_text(encoding="utf-8"))
            status = "cached"
        else:
            result = run_one(
                norm=norm,
                direction=direction,
                invader_count=count,
                seed=seed,
                generations=args.generations,
                interactions=args.interactions,
                fitness_interactions=args.fitness_interactions,
            )
            _write_json_atomic(path, result)
            status = "new"

        row = {
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
        print(
            f"[{index:03d}/{len(tasks):03d}] {status:6s} {direction:22s} "
            f"{norm} n={count:2d} seed={seed}: "
            f"final={result['final_invader_frequency']:.3f}",
            flush=True,
        )
        _write_summary(args.output, rows)

    print(
        f"=== completed {len(rows)} runs in "
        f"{time.perf_counter() - overall_started:.1f}s ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
