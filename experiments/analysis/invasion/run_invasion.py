"""Unified N=100 invasion experiment: one candidate against one resident.

Every invasion run is the same two-type mixture, whatever the two sides happen
to be:

  * the CANDIDATE -- the strategy under test, from an ``evolutionary.json`` run
    (dominant-lineage representative) or a hand-written ``.py`` file;
  * the RESIDENT -- what it is pitted against: a canonical Leading Eight norm, or
    another arbitrary strategy.

A norm is just a strategy whose code is ``BASELINES[norm]`` (see
``core.norm_source``), so all three historical entry points collapse into this
one module:

  ===========================  ==================  ==========================
  resident is...               flag                historical entry point
  ===========================  ==================  ==========================
  a canonical norm             ``--norms``         ``run_n100_invasion_count_sweep``
  (candidate from .json)                           ``run_n100_invasion_custom_code``
                                                   (candidate from .py)
  another arbitrary strategy   ``--residents``     ``run_pairwise_invasion``
  ===========================  ==================  ==========================

Mechanics are identical to the historical sweeps: N=100, deterministic payoff
imitation (a learner copies a sampled model iff the model has strictly higher
realized fitness), no mutation, fresh agents and reputations at each generation
boundary, and absorbing fixation/extinction stopping early. ``--action-error``
and ``--observation-error`` add independent execution and perception noise.

The sweep covers a single frequency axis: with ``k`` candidate copies the
candidate holds ``x = k/100``, so ``k = 1..99`` already spans every composition.
Whether the candidate is itself invadable is a separate question, measured by the
fixation benchmark (``docs/fixation_benchmark.md``) rather than inferred here.

Usage::

    # candidate vs the Leading Eight and the unconditional norms
    uv run python -m experiments.analysis.invasion.run_invasion \
        --source ev=agent-type1=results/.../evolutionary.json \
        --norms L1 L2 L3 L4 L5 L6 L7 L8 ALLC ALLD \
        --action-error 0.01 --observation-error 0.01 \
        --output results/quantitative_baseline/invasion/<dir>

    # candidate vs other arbitrary strategies (ordered pairs)
    uv run python -m experiments.analysis.invasion.run_invasion \
        --source A=agent-type1=a.json --source B=path/b.py \
        --residents A>B --output results/quantitative_baseline/invasion/<dir>
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
from experiments.v2_quantitative.baselines import BASELINES, BASELINE_VERSION
from experiments.v2_quantitative.executor import V2StrategyExecutor
from experiments.v2_quantitative.agent_full import V3StrategyExecutor
from experiments.v2_quantitative.signal_executor import SignalStrategyExecutor, SIGNAL_INTERFACE_VERSION

from ..paths import quantitative_results_dir
from .core import (
    AGENT_TYPES,
    ARCHIVED_DIRECTION_LABEL,
    BENEFIT,
    COST,
    FITNESS_WINDOW_FRACTION,
    INTERACTIONS_PER_GENERATION,
    KIND_CANDIDATE,
    KIND_NORM,
    KIND_RESIDENT,
    NORMS,
    NUM_GENERATIONS,
    Competitor,
    EvolvedSource,
    norm_source,
    payoff_imitation_update,
    play_generation_noisy,
    resolve_payoff_matrix,
    resolve_window,
    root_lineage,
    write_json_atomic,
)


# Host population for the invasion sweep. The archived sweeps were produced at
# N=100; the live default is N=20, which is ~5x cheaper per generation and is
# what the current benchmark runs use. Override with --population-size, and keep
# in mind that every initial-invader count must satisfy 1 <= count < N.
DEFAULT_POPULATION_SIZE = 20
# Legacy alias, retained because the deprecated runner shims re-export it.
POPULATION_SIZE = DEFAULT_POPULATION_SIZE
# 13 counts spread over 5%..95% of the host population, mirroring the archived
# N=100 grid so the two sweeps stay visually comparable.
DEFAULT_COUNTS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 19)
DEFAULT_SEEDS = (0, 1, 2)


def noisy_output(
    action_error: float,
    observation_error: float,
    population_size: int = DEFAULT_POPULATION_SIZE,
) -> Path:
    """Canonical noisy output dir; matches the archived naming convention."""
    suffix = f"ae{action_error:g}_oe{observation_error:g}".replace(".", "p")
    return (
        quantitative_results_dir() / "invasion"
        / f"n{population_size}_noisy_invasion_count_sweep_{suffix}"
    )


def default_output(
    action_error: float = 0.0,
    observation_error: float = 0.0,
    population_size: int = DEFAULT_POPULATION_SIZE,
) -> Path:
    if action_error or observation_error:
        return noisy_output(action_error, observation_error, population_size)
    return (
        quantitative_results_dir() / "invasion"
        / f"n{population_size}_invasion_count_sweep"
    )


def experiment_name(
    action_error: float,
    observation_error: float,
    population_size: int = DEFAULT_POPULATION_SIZE,
) -> str:
    """Descriptive experiment tag, matching the output-directory naming."""
    noise = "noisy_" if action_error or observation_error else ""
    return f"n{population_size}_{noise}invasion_count_sweep"


# --------------------------------------------------------------------------
# Candidate loading
# --------------------------------------------------------------------------
def _logged_payoff_matrix(config: dict[str, Any]) -> tuple[float | None, float | None]:
    """Payoff matrix recorded in an evolution log, or ``(None, None)``.

    Returning ``None`` rather than a default is what lets
    ``resolve_payoff_matrix`` tell "this log says b=2" apart from "this log says
    nothing, so use the archived constant".
    """
    benefit = config.get("benefit")
    cost = config.get("cost")
    return (
        None if benefit is None else float(benefit),
        None if cost is None else float(cost),
    )


def _representative_without_lineage(
    agent_type: str, path: Path, data: dict[str, Any],
) -> EvolvedSource:
    """Pick a representative from a legacy run that has no lineage records.

    Schema-v3 runs predate lineage tracking, so the dominant-lineage rule is
    unavailable. We instead group the final population by identical source and
    take the largest group as the dominant family, breaking ties by the family's
    best fitness. Within that family we keep the highest-fitness member. Runs
    whose members all differ degenerate to the single highest-fitness agent.
    """
    members = list(data["final_population"])
    if not members:
        raise ValueError(f"No final_population members in {path}")

    def best_of(family: list[dict[str, Any]]) -> dict[str, Any]:
        return min(
            family,
            key=lambda member: (-float(member["fitness"]), int(member["agent_id"])),
        )

    families: dict[str, list[dict[str, Any]]] = {}
    for member in members:
        key = hashlib.sha256(str(member["code"]).encode("utf-8")).hexdigest()
        families.setdefault(key, []).append(member)

    family_size, winner = min(
        ((len(family), best_of(family)) for family in families.values()),
        key=lambda item: (
            -item[0], -float(item[1]["fitness"]), int(item[1]["agent_id"]),
        ),
    )
    code = str(winner["code"])
    payoff = _logged_payoff_matrix(data.get("config", {}))
    return EvolvedSource(
        agent_type=agent_type,
        path=path,
        agent_id=int(winner["agent_id"]),
        lineage_id=-1,
        root_lineage_id=-1,
        root_family_size=family_size,
        fitness=float(winner["fitness"]),
        code=code,
        code_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
        benefit=payoff[0],
        cost=payoff[1],
    )


def load_representative_from_path(
    label: str, agent_type: str, path: Path,
) -> EvolvedSource:
    """Load a representative evolver outcome without hard-coded labels."""
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"Unknown agent type for {label}: {agent_type}")
    data = load_evolution_json(path)
    if not data.get("lineage_events"):
        return _representative_without_lineage(agent_type, path, data)
    parents = {
        int(event["lineage_id"]): (
            None if event.get("parent_lineage_id") is None
            else int(event["parent_lineage_id"])
        )
        for event in data["lineage_events"]
    }
    members = []
    family_counts: dict[int, int] = {}
    for member in data["final_population"]:
        root = root_lineage(int(member["lineage_id"]), parents)
        members.append((member, root))
        family_counts[root] = family_counts.get(root, 0) + 1
    dominant_root = min(family_counts, key=lambda root: (-family_counts[root], root))
    winner = min(
        (member for member, root in members if root == dominant_root),
        key=lambda member: (-float(member["fitness"]), int(member["agent_id"])),
    )
    code = str(winner["code"])
    payoff = _logged_payoff_matrix(data.get("config", {}))
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
        benefit=payoff[0],
        cost=payoff[1],
    )


def load_custom_source(label: str, raw_path: str, agent_type: str = "agent-type1") -> EvolvedSource:
    """Build an EvolvedSource directly from a hand-written strategy ``.py``."""
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"custom strategy file not found for {label}: {path}")
    code = path.read_text(encoding="utf-8")
    executors = {"agent-type1": V2StrategyExecutor, "agent-type2": V3StrategyExecutor,
                 "agent-type2-signal": SignalStrategyExecutor}
    if agent_type not in executors:
        raise ValueError(f"Unknown agent type for {label}: {agent_type}")
    executors[agent_type](code)  # fail fast on interface mismatch
    return EvolvedSource(
        agent_type=agent_type,
        path=path,
        agent_id=-1,
        lineage_id=-1,
        root_lineage_id=-1,
        root_family_size=0,
        fitness=0.0,
        code=code,
        code_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
    )


def load_candidate(label: str, agent_type: str, raw_path: str) -> EvolvedSource:
    """Load a candidate from a ``.py`` file or an ``evolutionary.json`` run."""
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"strategy file not found for {label}: {path}")
    if path.suffix == ".json":
        return load_representative_from_path(label, agent_type, path)
    return load_custom_source(label, path, agent_type)


def parse_sources(values: list[str]) -> dict[str, EvolvedSource]:
    """Parse repeatable ``LABEL=[AGENT_TYPE=]PATH`` specifications.

    ``AGENT_TYPE`` is optional and defaults to ``agent-type1``, which is what a
    hand-written ``.py`` candidate always is.
    """
    sources: dict[str, EvolvedSource] = {}
    for value in values:
        parts = value.split("=", 2)
        if len(parts) == 3:
            label, agent_type, raw_path = parts
        elif len(parts) == 2:
            label, raw_path = parts
            agent_type = "agent-type1"
        else:
            raise ValueError(
                f"Invalid --source {value!r}; expected LABEL=[AGENT_TYPE=]PATH"
            )
        if not label or label in sources:
            raise ValueError(f"Source labels must be non-empty and unique: {label!r}")
        sources[label] = load_candidate(label, agent_type, raw_path)
    return sources


# --------------------------------------------------------------------------
# Resident selection
# --------------------------------------------------------------------------
def norm_residents() -> dict[str, EvolvedSource]:
    return {norm: norm_source(norm) for norm in NORMS}


def parse_pairs(
    specs: list[str], sources: dict[str, EvolvedSource],
) -> list[tuple[str, str]]:
    """Expand repeated ``A>B`` specs into candidate/resident label pairs."""
    pairs: list[tuple[str, str]] = []
    for spec in specs:
        candidate, sep, resident = spec.partition(">")
        if not sep or not candidate or not resident:
            raise ValueError(f"Invalid pair {spec!r}; expected CANDIDATE>RESIDENT")
        for role, label in (("candidate", candidate), ("resident", resident)):
            if label not in sources:
                raise ValueError(f"Unknown {role} label in pair {spec!r}: {label!r}")
        if candidate == resident:
            raise ValueError(f"A strategy cannot invade itself: {spec!r}")
        pair = (candidate, resident)
        if pair not in pairs:
            pairs.append(pair)
    if not pairs:
        raise ValueError("--residents must name at least one CANDIDATE>RESIDENT pair")
    return pairs


def default_pairs(sources: dict[str, EvolvedSource]) -> list[tuple[str, str]]:
    """Every unordered pair, one ordering each (the other is its complement)."""
    labels = list(sources)
    if len(labels) < 2:
        raise ValueError("strategy-vs-strategy mode needs at least two --source values")
    return [
        (labels[i], labels[j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    ]


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------
def run_one(
    candidate: EvolvedSource,
    candidate_label: str,
    resident: EvolvedSource,
    resident_label: str,
    resident_kind: str,
    invader_count: int,
    seed: int,
    generations: int = NUM_GENERATIONS,
    interactions: int = INTERACTIONS_PER_GENERATION,
    fitness_window_fraction: float = FITNESS_WINDOW_FRACTION,
    action_error: float = 0.0,
    observation_error: float = 0.0,
    population_size: int = DEFAULT_POPULATION_SIZE,
    benefit: float = BENEFIT,
    cost: float = COST,
) -> dict[str, Any]:
    """Run the candidate (invader) against the resident from `invader_count` copies."""
    if not 1 <= invader_count < population_size:
        raise ValueError(
            f"invader_count must be in 1..{population_size - 1}, got {invader_count}"
        )
    fitness_interactions = resolve_window(
        interactions, population_size, fitness_window_fraction
    )
    rng = random.Random(seed)
    random.seed(1_000_003 + seed)
    invader_slots = set(rng.sample(range(population_size), invader_count))
    population = [
        Competitor.create(
            slot,
            KIND_CANDIDATE if slot in invader_slots else resident_kind,
            candidate_label if slot in invader_slots else resident_label,
            candidate if slot in invader_slots else resident,
        )
        for slot in range(population_size)
    ]

    trajectory: list[dict[str, Any]] = []
    started = time.perf_counter()
    for generation in range(generations):
        stats = play_generation_noisy(
            population, rng, interactions, fitness_window_fraction,
            action_error, observation_error, benefit, cost,
        )
        invaders = [m for m in population if m.kind == KIND_CANDIDATE]
        residents = [m for m in population if m.kind != KIND_CANDIDATE]
        frequency = len(invaders) / population_size
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
            population = payoff_imitation_update(
                population, rng, updates=population_size
            )
            post = sum(m.kind == KIND_CANDIDATE for m in population)
            if post in (0, population_size):
                trajectory.append(
                    {
                        "generation": generation + 1,
                        "invader_count": post,
                        "invader_frequency": post / population_size,
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
        "experiment": experiment_name(
            action_error, observation_error, population_size
        ),
        "candidate_label": candidate_label,
        "resident_label": resident_label,
        # Legacy aliases, kept so archived summaries stay readable downstream.
        "invader_label": candidate_label,
        "initial_invader_count": invader_count,
        "seed": seed,
        "candidate_kind": KIND_CANDIDATE,
        "resident_kind": resident_kind,
        "config": {
            "baseline_version": BASELINE_VERSION,
            "population_size": population_size,
            "num_generations": generations,
            "interactions_per_generation": interactions,
            "fitness_window_fraction": fitness_window_fraction,
            "fitness_interactions_per_generation": fitness_interactions,
            "burn_in_interactions_per_generation": interactions - fitness_interactions,
            # The matrix this mixture was paid from. Recorded so a rerun under a
            # different candidate log cannot silently reuse this result.
            "benefit": benefit,
            "cost": cost,
            "selection": "synchronous_deterministic_payoff_imitation",
            "imitation_eligibility": "strictly_higher_fitness_always_copied",
            "updates_per_generation": population_size,
            "mutation_rate": 0.0,
            "generation_lifecycle": "fresh_agent_and_reputation_reset",
            "fixation_threshold": 1.0,
            "absorbing_state_early_stop": True,
            "action_error_probability": action_error,
            "observation_error_probability": observation_error,
            "resident_kind": resident_kind,
        },
        "candidate_source": _source_record(candidate),
        # Legacy aliases used by earlier readers of this file format.
        "evolved_source": _source_record(candidate),
        "resident_source": _source_record(resident),
        "trajectory": trajectory,
        "final_invader_frequency": final_frequency,
        "invader_fixed": final_frequency == 1.0,
        "invader_extinct": final_frequency == 0.0,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _source_record(source: EvolvedSource) -> dict[str, Any]:
    record = {
        "agent_type": source.agent_type,
        "path": str(source.path),
        "agent_id": source.agent_id,
        "lineage_id": source.lineage_id,
        "root_lineage_id": source.root_lineage_id,
        "root_family_size": source.root_family_size,
        "fitness": source.fitness,
        "code_sha256": source.code_sha256,
    }
    if source.agent_type == "agent-type2-signal":
        record["signal_interface_version"] = SIGNAL_INTERFACE_VERSION
    return record


# --------------------------------------------------------------------------
# Paths and caching
# --------------------------------------------------------------------------
def result_path(
    output: Path, candidate_label: str, resident_label: str, count: int, seed: int,
) -> Path:
    """Current flat layout, with the resident folded in so pairs cannot collide."""
    return (
        output / candidate_label / f"invades_{resident_label}"
        / f"n{count}_seed{seed}" / "invasion.json"
    )


def legacy_result_paths(
    output: Path, candidate_label: str, resident_label: str, count: int, seed: int,
) -> list[Path]:
    """Locations written by the pre-unification runners, newest first.

    Two older layouts exist: the flat ``<label>/<norm>/...`` sweep layout (the
    resident label is the norm name), and the pairwise
    ``<label>/invades_<label>/...`` layout. Reading them keeps archived results
    cache-resident instead of forcing a recompute.
    """
    return [
        output / candidate_label / f"invades_{resident_label}"
        / f"n{count}_seed{seed}" / "invasion.json",
        output / candidate_label / resident_label
        / f"n{count}_seed{seed}" / "invasion.json",
        output / candidate_label / ARCHIVED_DIRECTION_LABEL / resident_label
        / f"n{count}_seed{seed}" / "invasion.json",
    ]


def existing_result_path(
    output: Path, candidate_label: str, resident_label: str, count: int, seed: int,
) -> Path:
    """Current path if present, else the newest archived path, else current."""
    current = result_path(output, candidate_label, resident_label, count, seed)
    for path in legacy_result_paths(
        output, candidate_label, resident_label, count, seed
    ):
        if path.exists():
            return path
    return current


def cache_matches(
    result: dict[str, Any],
    candidate: EvolvedSource,
    resident: EvolvedSource,
    resident_label: str,
    generations: int,
    interactions: int,
    fitness_window_fraction: float,
    action_error: float,
    observation_error: float,
    population_size: int = DEFAULT_POPULATION_SIZE,
    benefit: float = BENEFIT,
    cost: float = COST,
) -> bool:
    config = result.get("config", {})
    expected = {
        "population_size": population_size,
        "num_generations": generations,
        "interactions_per_generation": interactions,
        "fitness_interactions_per_generation": resolve_window(
            interactions, population_size, fitness_window_fraction
        ),
        # Results written before the payoff matrix was read from the log carry no
        # benefit, so they mismatch here and are recomputed rather than reused.
        "benefit": benefit,
        "cost": cost,
        "selection": "synchronous_deterministic_payoff_imitation",
        "updates_per_generation": population_size,
        "generation_lifecycle": "fresh_agent_and_reputation_reset",
        "absorbing_state_early_stop": True,
        "action_error_probability": action_error,
        "observation_error_probability": observation_error,
    }
    if not all(config.get(key) == value for key, value in expected.items()):
        return False
    for key, strategy in (("candidate_source", candidate), ("resident_source", resident)):
        if strategy.agent_type == "agent-type2-signal":
            saved = result.get(key, {})
            if saved.get("agent_type") != strategy.agent_type or saved.get("signal_interface_version") != SIGNAL_INTERFACE_VERSION:
                return False
    if result.get("candidate_source", result.get("evolved_source", {})).get(
        "code_sha256"
    ) != candidate.code_sha256:
        return False
    # Archived norm-mode results record no resident hash; their resident is the
    # norm named in the `norm` field.
    recorded_resident = result.get("resident_source", {}).get("code_sha256")
    if recorded_resident is not None:
        return recorded_resident == resident.code_sha256
    return result.get("norm") == resident_label


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
def execute(payload: tuple[Any, ...]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    (
        c_label, c_source, r_label, r_source, r_kind, count, seed,
        generations, interactions, fitness_window_fraction,
        action_error, observation_error, population_size,
        benefit, cost,
    ) = payload
    result = run_one(
        c_source, c_label, r_source, r_label, r_kind, count, seed,
        generations=generations,
        interactions=interactions,
        fitness_window_fraction=fitness_window_fraction,
        action_error=action_error,
        observation_error=observation_error,
        population_size=population_size,
        benefit=benefit,
        cost=cost,
    )
    return (c_label, r_label, count, seed), result


def write_summary(
    output: Path,
    rows: list[dict[str, Any]],
    pairs: list[tuple[str, str]],
    candidates: dict[str, EvolvedSource],
    residents: dict[str, EvolvedSource],
    counts: list[int],
    seeds: list[int],
    action_error: float,
    observation_error: float,
    population_size: int = DEFAULT_POPULATION_SIZE,
) -> None:
    groups: dict[str, Any] = {}
    for row in rows:
        cell = (
            groups.setdefault(row["candidate_label"], {})
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
    for by_resident in groups.values():
        for by_count in by_resident.values():
            for entry in by_count.values():
                values = entry.pop("final_frequencies")
                entry["mean_final_invader_frequency"] = sum(values) / len(values)

    write_json_atomic(
        output / "summary.json",
        {
            "experiment": experiment_name(
                action_error, observation_error, population_size
            ),
            "completed_or_cached_runs": len(rows),
            "population_size": population_size,
            "initial_invader_counts": counts,
            "seeds": seeds,
            "action_error_probability": action_error,
            "observation_error_probability": observation_error,
            # One entry per distinct matrix actually played. Normally a single
            # entry; more than one only when candidates loaded from logs that
            # disagree are swept together, in which case each candidate's own
            # rows carry the matrix it was played under.
            "payoff_matrices": [
                {"benefit": benefit, "cost": cost}
                for benefit, cost in sorted({
                    (row["benefit"], row["cost"]) for row in rows
                })
            ],
            "selection": "synchronous_deterministic_payoff_imitation",
            "generation_lifecycle": "fresh_agent_and_reputation_reset",
            "absorbing_state_early_stop": True,
            "pairs": [f"{c}>{r}" for c, r in pairs],
            "sources": {
                label: _source_record(source) for label, source in candidates.items()
            },
            "residents": {
                label: {
                    "kind": (
                        KIND_NORM if label in NORMS else KIND_RESIDENT
                    ),
                    "code_sha256": source.code_sha256,
                }
                for label, source in residents.items()
            },
            "groups": groups,
            "runs": rows,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--source",
        "--strategy",  # alias kept for the retired pairwise CLI
        action="append",
        default=[],
        metavar="LABEL=[AGENT_TYPE=]PATH",
        help=(
            "Candidate strategy: an evolutionary.json run (dominant-lineage "
            "representative) or a hand-written .py; repeat for several."
        ),
    )
    parser.add_argument(
        "--norms", nargs="+", choices=NORMS, default=None,
        help="Canonical norms to use as residents for every candidate.",
    )
    parser.add_argument(
        "--residents",
        "--directions",  # alias kept for the retired pairwise CLI
        action="append",
        default=[],
        metavar="CANDIDATE>RESIDENT",
        help=(
            "Arbitrary strategy-vs-strategy pairs. Without this, all unordered "
            "source pairs are run."
        ),
    )
    parser.add_argument(
        "--invader-counts", nargs="+", type=int, default=list(DEFAULT_COUNTS)
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
        help=(
            "Host population size for the sweep. Every --invader-counts value "
            "must satisfy 1 <= count < this. Default N=20 (archived sweeps "
            "used N=100)."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--interactions", type=int, default=INTERACTIONS_PER_GENERATION)
    parser.add_argument(
        "--fitness-window-fraction", type=float, default=FITNESS_WINDOW_FRACTION
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--action-error", type=float, default=0.0)
    parser.add_argument("--observation-error", type=float, default=0.0)
    parser.add_argument(
        "--benefit",
        type=float,
        default=None,
        help=(
            "PD cooperation benefit. Default (unset) reads it from each "
            "candidate's evolution log, so a strategy is tested in the game it "
            "was selected under. Pass explicitly to force one matrix, which is "
            "what reproducing the archived benefit=2 results requires."
        ),
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=None,
        help="PD cooperation cost. Default (unset): read from the log, like --benefit.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    for name, value in (
        ("--action-error", args.action_error),
        ("--observation-error", args.observation_error),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be in [0, 1]")
    if args.population_size < 3:
        parser.error("--population-size must be at least 3")
    if any(not 1 <= count < args.population_size for count in args.invader_counts):
        parser.error(
            f"--invader-counts must be in 1..{args.population_size - 1} "
            f"for --population-size {args.population_size}"
        )
    if not 0.0 <= args.fitness_window_fraction < 1.0:
        parser.error("--fitness-window-fraction must be in [0, 1)")
    if not args.source:
        parser.error("at least one --source LABEL=[AGENT_TYPE=]PATH is required")
    try:
        candidates = parse_sources(args.source)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    if args.output is None:
        args.output = default_output(
            args.action_error, args.observation_error, args.population_size
        )
    if args.smoke:
        args.invader_counts = [1, max(1, args.population_size // 2)]
        args.seeds = [0]
        args.generations = 2
        args.interactions = 20
        args.fitness_window_fraction = 0.25
        args.output = args.output / "_smoke"

    # Resident mode: canonical norms unless explicit pairs are given.
    if args.norms is not None and args.residents:
        parser.error("use either --norms or --residents, not both")
    if args.residents:
        try:
            pairs = parse_pairs(args.residents, candidates)
        except ValueError as exc:
            parser.error(str(exc))
        residents: dict[str, EvolvedSource] = dict(candidates)
        resident_kinds = {
            label: KIND_NORM if label in NORMS else KIND_RESIDENT
            for label in residents
        }
        mode = "strategy"
    else:
        norms = args.norms if args.norms is not None else list(NORMS)
        residents = norm_residents()
        residents = {norm: residents[norm] for norm in norms}
        resident_kinds = {label: KIND_NORM for label in residents}
        pairs = [(label, norm) for label in candidates for norm in residents]
        mode = "norm"

    tasks = [
        (c_label, r_label, count, seed)
        for c_label, r_label in pairs
        for count in args.invader_counts
        for seed in args.seeds
    ]
    # One matrix per pair: the population is paid from a single table, so the
    # candidate's log and (in pairwise mode) the resident's log have to agree.
    try:
        payoff_by_pair = {
            (c_label, r_label): resolve_payoff_matrix(
                candidates[c_label], residents[r_label],
                benefit=args.benefit, cost=args.cost,
            )
            for c_label, r_label in pairs
        }
    except ValueError as exc:
        parser.error(str(exc))
    for pair, (benefit, cost) in payoff_by_pair.items():
        if cost <= 0.0:
            parser.error(f"cost must be positive for pair {pair[0]}>{pair[1]}")
        if benefit <= cost:
            parser.error(
                f"benefit must exceed cost for a social dilemma; "
                f"pair {pair[0]}>{pair[1]} resolved to benefit={benefit}, cost={cost}"
            )
    if args.benefit is not None or args.cost is not None:
        # Say so when the override contradicts what the log says: the run then
        # no longer describes the game the strategies were selected under. A
        # pair whose own logs disagree is reported the same way -- that is
        # exactly the case the override was needed to get past.
        contradicted = []
        for pair in pairs:
            try:
                logged = resolve_payoff_matrix(
                    candidates[pair[0]], residents[pair[1]]
                )
            except ValueError:
                contradicted.append(pair)
                continue
            if logged != payoff_by_pair[pair]:
                contradicted.append(pair)
        if contradicted:
            print(
                f"  [warn] --benefit/--cost override the matrix recorded in the "
                f"source log(s) for {len(contradicted)} pair(s); results will not "
                f"describe the game those strategies were selected under.",
                flush=True,
            )
    rows: list[dict[str, Any]] = []
    pending: list[tuple[Any, ...]] = []
    task_by_key = {task: task for task in tasks}
    started = time.perf_counter()
    print(
        f"=== N={args.population_size} invasion ({mode}-resident): "
        f"{len(tasks)} runs, {len(pairs)} pair(s) ===",
        flush=True,
    )
    matrices = sorted(set(payoff_by_pair.values()))
    print(
        f"  payoff matrix: "
        + ", ".join(f"benefit={b:g}/cost={c:g}" for b, c in matrices)
        + ("  (from the source log(s); pass --benefit/--cost to override)"
           if args.benefit is None and args.cost is None else "  (explicit override)"),
        flush=True,
    )

    def record(task: tuple[Any, ...], result: dict[str, Any], status: str) -> None:
        c_label, r_label, count, seed = task
        path = result_path(args.output, c_label, r_label, count, seed)
        if status in ("new", "normalized"):
            write_json_atomic(path, result)
        rows.append(
            {
                "candidate_label": c_label,
                "resident_label": r_label,
                "initial_invader_count": count,
                "seed": seed,
                "final_invader_frequency": result["final_invader_frequency"],
                "invader_fixed": result["invader_fixed"],
                "invader_extinct": result["invader_extinct"],
                "benefit": result["config"]["benefit"],
                "cost": result["config"]["cost"],
                "status": status,
                "path": str(path.relative_to(args.output)),
            }
        )
        if len(rows) % 25 == 0 or len(rows) == len(tasks):
            print(f"[{len(rows):04d}/{len(tasks):04d}] {status}", flush=True)

    for task in tasks:
        c_label, r_label, count, seed = task
        benefit, cost = payoff_by_pair[(c_label, r_label)]
        path = existing_result_path(args.output, c_label, r_label, count, seed)
        if path.exists() and not args.force:
            result = json.loads(path.read_text(encoding="utf-8"))
            if cache_matches(
                result, candidates[c_label], residents[r_label], r_label,
                args.generations, args.interactions,
                args.fitness_window_fraction,
                args.action_error, args.observation_error,
                args.population_size, benefit, cost,
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
            (
                c_label, candidates[c_label], r_label, residents[r_label],
                resident_kinds[r_label], count, seed, args.generations,
                args.interactions, args.fitness_window_fraction,
                args.action_error, args.observation_error, args.population_size,
                benefit, cost,
            )
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
        args.output, rows, pairs, candidates, residents,
        list(args.invader_counts), list(args.seeds),
        args.action_error, args.observation_error,
        args.population_size,
    )
    print(f"=== completed in {time.perf_counter() - started:.1f}s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
