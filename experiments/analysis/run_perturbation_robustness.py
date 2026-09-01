"""Run the homogeneous-population part of STRATEGY_SUPERIORITY_STANDARD.md."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .invasion.run_best_leading_eight_invasion import (
    BASELINES,
    Competitor,
    EvolvedSource,
    NORMS,
    _write_json_atomic,
    load_representative,
)
from .invasion.run_n100_invasion_count_sweep import _play_generation_noisy
from .paths import quantitative_results_dir


POPULATION_SIZE = 100
INTERACTIONS = 1_000
FITNESS_INTERACTIONS = 200
CONDITIONS = (
    (0.00, 0.00),
    (0.01, 0.00),
    (0.05, 0.00),
    (0.00, 0.01),
    (0.00, 0.05),
    (0.01, 0.01),
    (0.05, 0.05),
    (0.10, 0.10),
)


def _condition_name(action_error: float, observation_error: float) -> str:
    return f"a{int(action_error * 100):02d}_o{int(observation_error * 100):02d}"


def _strategy_source(candidate: EvolvedSource, strategy: str) -> EvolvedSource:
    if strategy == "candidate":
        return candidate
    code = BASELINES[strategy]
    return EvolvedSource(
        agent_type="agent-type1",
        path=Path(f"baseline:{strategy}"),
        agent_id=-1,
        lineage_id=-1,
        root_lineage_id=-1,
        root_family_size=POPULATION_SIZE,
        fitness=0.0,
        code=code,
        code_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
    )


def run_one(payload: tuple[str, EvolvedSource, float, float, int]) -> dict[str, Any]:
    strategy, source, action_error, observation_error, seed = payload
    rng = random.Random(seed)
    random.seed(1_000_003 + seed)
    kind = "evolved" if strategy == "candidate" else "norm"
    norm = "IS" if strategy == "candidate" else strategy
    population = [
        Competitor.create(agent_id, kind, norm, source)
        for agent_id in range(POPULATION_SIZE)
    ]
    stats = _play_generation_noisy(
        population,
        rng,
        INTERACTIONS,
        FITNESS_INTERACTIONS,
        action_error,
        observation_error,
    )
    mean_payoff = sum(stats["all_interaction_payoffs"]) / POPULATION_SIZE
    return {
        "strategy": strategy,
        "seed": seed,
        "action_error_probability": action_error,
        "observation_error_probability": observation_error,
        "cooperation_rate": stats["cooperation_rate"],
        "mean_payoff_per_agent": mean_payoff,
    }


def _paired_ci(values: list[float], rng: np.random.Generator) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(data), size=(20_000, len(data)))
    means = data[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize(rows: list[dict[str, Any]], candidate: EvolvedSource, seeds: list[int]) -> dict[str, Any]:
    strategies = ("candidate", *NORMS)
    by_key = {
        (row["strategy"], row["action_error_probability"], row["observation_error_probability"], row["seed"]): row
        for row in rows
    }
    means: dict[str, Any] = {}
    retention: dict[tuple[str, int, str], float] = {}
    for strategy in strategies:
        control = {
            seed: by_key[(strategy, 0.0, 0.0, seed)]["mean_payoff_per_agent"]
            for seed in seeds
        }
        means[strategy] = {}
        for action_error, observation_error in CONDITIONS:
            name = _condition_name(action_error, observation_error)
            cells = [by_key[(strategy, action_error, observation_error, seed)] for seed in seeds]
            means[strategy][name] = {
                "mean_cooperation_rate": float(np.mean([cell["cooperation_rate"] for cell in cells])),
                "mean_payoff_per_agent": float(np.mean([cell["mean_payoff_per_agent"] for cell in cells])),
            }
            for seed, cell in zip(seeds, cells, strict=True):
                retention[(strategy, seed, name)] = (
                    cell["mean_payoff_per_agent"] / control[seed] if control[seed] else 0.0
                )
    comparisons: dict[str, Any] = {}
    for norm in NORMS:
        comparisons[norm] = {}
        for action_error, observation_error in CONDITIONS[1:]:
            name = _condition_name(action_error, observation_error)
            deltas = [retention[("candidate", seed, name)] - retention[(norm, seed, name)] for seed in seeds]
            low, high = _paired_ci(deltas, np.random.default_rng(73_001 + sum(seeds)))
            comparisons[norm][name] = {
                "mean_retention_difference": float(np.mean(deltas)),
                "paired_bootstrap_95_ci": [low, high],
                "noninferior_margin_minus_0_02": low >= -0.02,
                "superior": low > 0.0,
            }
        noisy_names = [
            _condition_name(action_error, observation_error)
            for action_error, observation_error in CONDITIONS[1:]
        ]
        worst_deltas = [
            min(retention[("candidate", seed, name)] for name in noisy_names)
            - min(retention[(norm, seed, name)] for name in noisy_names)
            for seed in seeds
        ]
        low, high = _paired_ci(
            worst_deltas, np.random.default_rng(91_003 + sum(seeds))
        )
        comparisons[norm]["worst_case_across_suite"] = {
            "mean_retention_difference": float(np.mean(worst_deltas)),
            "paired_bootstrap_95_ci": [low, high],
            "noninferior_margin_minus_0_02": low >= -0.02,
            "superior": low > 0.0,
        }
    return {
        "experiment": "strategy_perturbation_robustness_screen",
        "standard": "experiments/analysis/STRATEGY_SUPERIORITY_STANDARD.md",
        "stage": (
            "screening"
            if seeds == list(range(10))
            else "confirmation"
            if seeds == list(range(100, 130))
            else "custom"
        ),
        "candidate": {
            "agent_type": candidate.agent_type,
            "source": str(candidate.path),
            "agent_id": candidate.agent_id,
            "code_sha256": candidate.code_sha256,
        },
        "config": {
            "population_size": POPULATION_SIZE,
            "interactions": INTERACTIONS,
            "fitness_interactions": FITNESS_INTERACTIONS,
            "seeds": seeds,
            "conditions": [list(condition) for condition in CONDITIONS],
            "generation_lifecycle": "fresh_agent_and_reputation_reset",
        },
        "means": means,
        "retention_comparisons": comparisons,
        "raw_runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=quantitative_results_dir() / "robustness" / "agent-type1_seed4_screen",
    )
    args = parser.parse_args()
    candidate = load_representative("agent-type1")
    strategies = ("candidate", *NORMS)
    tasks = [
        (strategy, _strategy_source(candidate, strategy), action_error, observation_error, seed)
        for strategy in strategies
        for action_error, observation_error in CONDITIONS
        for seed in args.seeds
    ]
    print(f"Running {len(tasks)} homogeneous robustness trials", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_one, tasks))
    summary = summarize(rows, candidate, list(args.seeds))
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(args.output / "summary.json", summary)
    print(f"Wrote {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
