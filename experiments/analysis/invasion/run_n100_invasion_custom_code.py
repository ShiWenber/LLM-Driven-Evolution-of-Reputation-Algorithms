"""N=100 invasion sweep for hand-written strategy code.

This is a thin variant of ``run_n100_invasion_count_sweep`` for strategies that
are NOT products of an evolutionary run (e.g. a best strategy recorded in the
README from an older interface generation). Instead of parsing an
``evolutionary.json`` and picking the dominant lineage representative, it
loads the strategy source directly from a ``.py`` file that defines the
current one-directional interface::

    def observe(A_rep, A_action, B_rep, B_action, my_reputation) -> float
    def decide(my_reputation, opponent_reputation) -> bool

Everything else (N=100 resident population, deterministic payoff imitation,
noise, caching, summary schema) is identical to the evolved-strategy sweep so
results are directly comparable.

Usage::

    uv run python -m experiments.analysis.invasion.run_n100_invasion_custom_code \
        --source readme_best=experiments/analysis/invasion/custom_strategies/readme_best.py \
        --workers 48 --action-error 0.01 --observation-error 0.01 \
        --output results/quantitative_baseline/invasion/<dir>
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from experiments.v2_quantitative.executor import V2StrategyExecutor

from .core import (
    FITNESS_INTERACTIONS,
    INTERACTIONS_PER_GENERATION,
    NORMS,
    NUM_GENERATIONS,
    EvolvedSource,
)
from .run_n100_invasion_count_sweep import (
    DEFAULT_COUNTS,
    DEFAULT_SEEDS,
    POPULATION_SIZE,
    cache_matches,
    execute,
    existing_result_path,
    experiment_name,
    result_path,
    write_json_atomic,
    write_summary,
)

AGENT_TYPE = "agent-type1"


def load_custom_source(label: str, raw_path: str) -> EvolvedSource:
    """Build an EvolvedSource directly from a strategy ``.py`` file."""
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"custom strategy file not found for {label}: {path}")
    code = path.read_text(encoding="utf-8")
    # Fail fast if the code does not satisfy the current interface.
    V2StrategyExecutor(code)
    return EvolvedSource(
        agent_type=AGENT_TYPE,
        path=path,
        agent_id=-1,
        lineage_id=-1,
        root_lineage_id=-1,
        root_family_size=0,
        fitness=0.0,
        code=code,
        code_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
    )


def parse_sources(values: list[str]) -> dict[str, EvolvedSource]:
    """Parse repeatable ``LABEL=PATH_TO_PY`` specifications."""
    sources: dict[str, EvolvedSource] = {}
    for value in values:
        try:
            label, raw_path = value.split("=", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --source {value!r}; expected LABEL=PATH_TO_PY"
            ) from exc
        if not label or label in sources:
            raise ValueError(f"Source labels must be non-empty and unique: {label!r}")
        sources[label] = load_custom_source(label, raw_path)
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="LABEL=PATH_TO_PY",
        help="Hand-written strategy file; repeat for multiple strategies.",
    )
    parser.add_argument("--norms", nargs="+", choices=NORMS, default=list(NORMS))
    parser.add_argument(
        "--invader-counts", nargs="+", type=int, default=list(DEFAULT_COUNTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--interactions", type=int, default=INTERACTIONS_PER_GENERATION)
    parser.add_argument(
        "--fitness-interactions", type=int, default=FITNESS_INTERACTIONS
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--action-error", type=float, default=0.0)
    parser.add_argument("--observation-error", type=float, default=0.0)
    args = parser.parse_args()

    if not 0.0 <= args.action_error <= 1.0:
        parser.error("--action-error must be in [0, 1]")
    if not 0.0 <= args.observation_error <= 1.0:
        parser.error("--observation-error must be in [0, 1]")
    if any(not 1 <= count < POPULATION_SIZE for count in args.invader_counts):
        parser.error("--invader-counts must be in 1..99")
    if not 0 < args.fitness_interactions <= args.interactions:
        parser.error("--fitness-interactions must be in 1..--interactions")
    if not args.source:
        parser.error("at least one --source LABEL=PATH_TO_PY is required")
    try:
        sources = parse_sources(args.source)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    tasks = [
        (kind, norm, count, seed)
        for kind in sources
        for norm in args.norms
        for count in args.invader_counts
        for seed in args.seeds
    ]
    rows: list[dict[str, Any]] = []
    pending: list[tuple[Any, ...]] = []
    task_by_key = {task: task for task in tasks}
    started = time.perf_counter()
    print(f"=== N=100 custom-code invasion sweep: {len(tasks)} runs ===", flush=True)

    def record(task: tuple[Any, ...], result: dict[str, Any], status: str) -> None:
        kind, norm, count, seed = task
        path = result_path(args.output, kind, norm, count, seed)
        if status in ("new", "normalized"):
            write_json_atomic(path, result)
        rows.append(
            {
                "agent_type": kind,
                "norm": norm,
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
        kind, norm, count, seed = task
        path = existing_result_path(args.output, kind, norm, count, seed)
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
            (kind, sources[kind], norm, count, seed, args.generations,
             args.interactions, args.fitness_interactions,
             args.action_error, args.observation_error)
        )

    if pending:
        print(
            f"running {len(pending)} pending tasks with {args.workers} workers",
            flush=True,
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(execute, payload): payload for payload in pending}
            for future in concurrent.futures.as_completed(futures):
                key, result = future.result()
                record(task_by_key[key], result, "new")

    write_summary(
        args.output, rows, sources, list(args.invader_counts), list(args.seeds),
        args.action_error, args.observation_error,
    )
    print(
        f"=== {experiment_name(args.action_error, args.observation_error)} "
        f"completed in {time.perf_counter() - started:.1f}s ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
