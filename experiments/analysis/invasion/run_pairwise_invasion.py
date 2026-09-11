"""Pairwise bidirectional invasion: any strategy vs any strategy.

The ``run_n100_invasion_count_sweep`` engine pits one evolved strategy against
the eight canonical Leading Eight norms. This module generalises that to an
ARBITRARY ordered pair of strategies, which is what a strict superiority test
requires: to claim "strategy A beats strategy B" you must show A invades B from
rare while B cannot invade A from rare.

Sources may be either
  * a hand-written strategy ``.py`` file (current one-directional interface), or
  * an ``evolutionary.json`` run (the dominant lineage representative is used).

Unlike the norm sweep, the resident is a second arbitrary strategy, so the
payoff-imitation copier must copy the MODEL's kind/norm/source (the original
``payoff_imitation_update`` assumes a single shared source and copies the
learner's). A corrected copier is defined here.

Output schema mirrors the norm sweep so existing plotting utilities can be
adapted: ``groups[label_a][label_b][count] = {runs, fixations, extinctions,
mean_final_invader_frequency}`` meaning "label_a invades label_b".

Usage::

    uv run python -m experiments.analysis.invasion.run_pairwise_invasion \
        --strategy A=agent-type1=path/a.py \
        --strategy B=agent-type1=path/b.py \
        --workers 48 --action-error 0.01 --observation-error 0.01 \
        --output results/quantitative_baseline/invasion/pairwise_A_vs_B
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from experiments.evolution_log import load_evolution_json
from experiments.v2_quantitative.executor import V2StrategyExecutor

from ..paths import quantitative_results_dir
from .core import (
    FITNESS_INTERACTIONS,
    INTERACTIONS_PER_GENERATION,
    NUM_GENERATIONS,
    Competitor,
    EvolvedSource,
    root_lineage,
    write_json_atomic,
)
from .run_n100_invasion_count_sweep import (
    POPULATION_SIZE,
    _play_generation_noisy,
    load_representative_from_path,
)


KIND_A, KIND_B = "A", "B"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_source(label: str, agent_type: str, raw_path: str) -> EvolvedSource:
    """Load a strategy from a .py file or an evolutionary.json run."""
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"strategy file not found for {label}: {path}")
    if path.suffix == ".json":
        return load_representative_from_path(label, agent_type, path)
    code = path.read_text(encoding="utf-8")
    V2StrategyExecutor(code)  # fail fast on interface mismatch
    return EvolvedSource(
        agent_type=agent_type,
        path=path,
        agent_id=-1,
        lineage_id=-1,
        root_lineage_id=-1,
        root_family_size=0,
        fitness=0.0,
        code=code,
        code_sha256=_sha(code),
    )


def parse_strategies(values: list[str]) -> dict[str, EvolvedSource]:
    out: dict[str, EvolvedSource] = {}
    for value in values:
        parts = value.split("=", 2)
        if len(parts) == 3:
            label, agent_type, raw = parts
        elif len(parts) == 2:
            label, raw = parts
            agent_type = "agent-type1"
        else:
            raise ValueError(
                f"Invalid --strategy {value!r}; expected LABEL=[AGENT_TYPE=]PATH"
            )
        if not label or label in out:
            raise ValueError(f"Strategy labels must be non-empty and unique: {label!r}")
        out[label] = load_source(label, agent_type, raw)
    return out


def pairwise_update(
    population: list[Competitor], rng: random.Random, updates: int,
) -> list[Competitor]:
    """Copy a sampled role model only when strictly fitter (copies MODEL identity)."""
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
        # NOTE: copy the MODEL's kind/norm/source, not the learner's.
        next_population[learner_pos] = Competitor.create(
            learner.agent_id, model.kind, model.norm, model.source
        )
    return [
        member if member is not old[pos] else Competitor.create(
            member.agent_id, member.kind, member.norm, member.source
        )
        for pos, member in enumerate(next_population)
    ]


def run_pair(
    source_a: EvolvedSource,
    label_a: str,
    source_b: EvolvedSource,
    label_b: str,
    invader_count: int,
    seed: int,
    generations: int = NUM_GENERATIONS,
    interactions: int = INTERACTIONS_PER_GENERATION,
    fitness_interactions: int = FITNESS_INTERACTIONS,
    action_error: float = 0.0,
    observation_error: float = 0.0,
) -> dict[str, Any]:
    """Run label_a (invader) against label_b (resident) from invader_count copies."""
    if not 1 <= invader_count < POPULATION_SIZE:
        raise ValueError("invader_count must be in 1..99")
    rng = random.Random(seed)
    random.seed(1_000_003 + seed)
    invader_slots = set(rng.sample(range(POPULATION_SIZE), invader_count))
    sources = {KIND_A: (source_a, label_a), KIND_B: (source_b, label_b)}
    population = []
    for slot in range(POPULATION_SIZE):
        kind = KIND_A if slot in invader_slots else KIND_B
        source, label = sources[kind]
        population.append(Competitor.create(slot, kind, label, source))

    trajectory: list[dict[str, Any]] = []
    started = time.perf_counter()
    for generation in range(generations):
        stats = _play_generation_noisy(
            population, rng, interactions, fitness_interactions,
            action_error, observation_error,
        )
        invaders = [m for m in population if m.kind == KIND_A]
        residents = [m for m in population if m.kind == KIND_B]
        frequency = len(invaders) / POPULATION_SIZE
        trajectory.append(
            {
                "generation": generation,
                "invader_count": len(invaders),
                "invader_frequency": frequency,
                "cooperation_rate": stats["cooperation_rate"],
                "invader_fitness_mean": (
                    sum(m.fitness for m in invaders) / len(invaders)
                    if invaders else None
                ),
                "resident_fitness_mean": (
                    sum(m.fitness for m in residents) / len(residents)
                    if residents else None
                ),
            }
        )
        if generation < generations - 1:
            population = pairwise_update(population, rng, updates=POPULATION_SIZE)
            post = sum(m.kind == KIND_A for m in population)
            if post in (0, POPULATION_SIZE):
                trajectory.append(
                    {
                        "generation": generation + 1,
                        "invader_count": post,
                        "invader_frequency": post / POPULATION_SIZE,
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
        "experiment": "n100_pairwise_bidirectional_invasion",
        "invader_label": label_a,
        "resident_label": label_b,
        "initial_invader_count": invader_count,
        "seed": seed,
        "config": {
            "population_size": POPULATION_SIZE,
            "num_generations": generations,
            "interactions_per_generation": interactions,
            "fitness_interactions_per_generation": fitness_interactions,
            "selection": "synchronous_deterministic_payoff_imitation",
            "imitation_eligibility": "strictly_higher_fitness_always_copied",
            "updates_per_generation": POPULATION_SIZE,
            "mutation_rate": 0.0,
            "generation_lifecycle": "fresh_agent_and_reputation_reset",
            "absorbing_state_early_stop": True,
            "action_error_probability": action_error,
            "observation_error_probability": observation_error,
        },
        "invader_source": {"code_sha256": source_a.code_sha256},
        "resident_source": {"code_sha256": source_b.code_sha256},
        "trajectory": trajectory,
        "final_invader_frequency": final_frequency,
        "invader_fixed": final_frequency == 1.0,
        "invader_extinct": final_frequency == 0.0,
        "elapsed_seconds": time.perf_counter() - started,
    }


def result_path(output: Path, label_a: str, label_b: str, count: int, seed: int) -> Path:
    return output / label_a / f"invades_{label_b}" / f"n{count}_seed{seed}" / "invasion.json"


def execute(payload: tuple[Any, ...]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    (la, sa, lb, sb, count, seed, gens, inter, fit, ae, oe) = payload
    result = run_pair(sa, la, sb, lb, count, seed, gens, inter, fit, ae, oe)
    return (la, lb, count, seed), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--strategy", action="append", default=[],
        metavar="LABEL=[AGENT_TYPE=]PATH",
        help="Strategy source (.py or evolutionary.json); repeat for >=2 strategies.",
    )
    parser.add_argument(
        "--invader-counts", nargs="+", type=int,
        default=[1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--interactions", type=int, default=INTERACTIONS_PER_GENERATION)
    parser.add_argument("--fitness-interactions", type=int, default=FITNESS_INTERACTIONS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--action-error", type=float, default=0.0)
    parser.add_argument("--observation-error", type=float, default=0.0)
    parser.add_argument(
        "--directions", action="append", default=[],
        metavar="A>B", help="Restrict to ordered pairs, e.g. s4>s3.",
    )
    args = parser.parse_args()

    for name, val in (("--action-error", args.action_error),
                      ("--observation-error", args.observation_error)):
        if not 0.0 <= val <= 1.0:
            parser.error(f"{name} must be in [0, 1]")
    if any(not 1 <= c < POPULATION_SIZE for c in args.invader_counts):
        parser.error("--invader-counts must be in 1..99")
    if not 0 < args.fitness_interactions <= args.interactions:
        parser.error("--fitness-interactions must be in 1..--interactions")
    try:
        strategies = parse_strategies(args.strategy)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if len(strategies) < 2:
        parser.error("at least two --strategy sources are required")

    labels = list(strategies)
    allowed = None
    if args.directions:
        allowed = set()
        for spec in args.directions:
            try:
                a, b = spec.split(">", 1)
            except ValueError:
                parser.error(f"Invalid --directions {spec!r}; expected A>B")
            if a not in strategies or b not in strategies:
                parser.error(f"Unknown label in --directions {spec!r}")
            allowed.add((a, b))

    # One direction per UNORDERED pair.
    #
    # "A invades B at count k" and "B invades A at count 100-k" are the same
    # composition (A occupies k slots) and differ only by the random draw that
    # placed the slots. Sweeping the counts in a single direction therefore
    # already realises every composition, so emitting both would double the
    # work for no extra information.
    pairs: list[tuple[str, str]] = []
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            pairs.append((la, lb))

    tasks = []
    for la, lb in pairs:
        if allowed is not None and (la, lb) not in allowed and (lb, la) not in allowed:
            continue
        for count in args.invader_counts:
            for seed in args.seeds:
                tasks.append((la, lb, count, seed))

    rows: list[dict[str, Any]] = []
    pending: list[tuple[Any, ...]] = []
    started = time.perf_counter()
    print(f"=== pairwise invasion: {len(tasks)} runs ===", flush=True)

    def record(task: tuple[Any, ...], result: dict[str, Any], status: str) -> None:
        la, lb, count, seed = task
        path = result_path(args.output, la, lb, count, seed)
        if status == "new":
            write_json_atomic(path, result)
        rows.append(
            {
                "invader_label": la,
                "resident_label": lb,
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

    for (la, lb, count, seed) in tasks:
        path = result_path(args.output, la, lb, count, seed)
        if path.exists() and not args.force:
            result = json.loads(path.read_text(encoding="utf-8"))
            cfg = result.get("config", {})
            if (cfg.get("action_error_probability") == args.action_error
                    and cfg.get("observation_error_probability") == args.observation_error
                    and cfg.get("num_generations") == args.generations
                    and result.get("invader_source", {}).get("code_sha256")
                    == strategies[la].code_sha256
                    and result.get("resident_source", {}).get("code_sha256")
                    == strategies[lb].code_sha256):
                record((la, lb, count, seed), result, "cached")
                continue
        pending.append(
            (la, strategies[la], lb, strategies[lb], count, seed,
             args.generations, args.interactions, args.fitness_interactions,
             args.action_error, args.observation_error)
        )

    if pending:
        print(f"running {len(pending)} pending with {args.workers} workers", flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(execute, p) for p in pending]
            for future in concurrent.futures.as_completed(futures):
                key, result = future.result()
                record(key, result, "new")

    groups: dict[str, Any] = {}
    for row in rows:
        cell = (
            groups.setdefault(row["invader_label"], {})
            .setdefault(row["resident_label"], {})
            .setdefault(
                str(row["initial_invader_count"]),
                {"runs": 0, "fixations": 0, "extinctions": 0, "final_frequencies": []},
            )
        )
        cell["runs"] += 1
        cell["fixations"] += int(row["invader_fixed"])
        cell["extinctions"] += int(row["invader_extinct"])
        cell["final_frequencies"].append(row["final_invader_frequency"])
    for by_res in groups.values():
        for cell in by_res.values():
            for entry in cell.values():
                vals = entry.pop("final_frequencies")
                entry["mean_final_invader_frequency"] = sum(vals) / len(vals)

    write_json_atomic(
        args.output / "summary.json",
        {
            "experiment": "n100_pairwise_bidirectional_invasion",
            "completed_or_cached_runs": len(rows),
            "population_size": POPULATION_SIZE,
            "initial_invader_counts": list(args.invader_counts),
            "seeds": list(args.seeds),
            "action_error_probability": args.action_error,
            "observation_error_probability": args.observation_error,
            "selection": "synchronous_deterministic_payoff_imitation",
            "strategies": {
                label: {
                    "agent_type": src.agent_type,
                    "path": str(src.path),
                    "code_sha256": src.code_sha256,
                }
                for label, src in strategies.items()
            },
            "groups": groups,
            "runs": rows,
        },
    )
    print(f"=== completed in {time.perf_counter() - started:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
