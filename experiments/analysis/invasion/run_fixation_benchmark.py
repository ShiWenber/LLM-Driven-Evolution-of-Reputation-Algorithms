"""Fixation-probability benchmark for a candidate strategy.

WHY THIS EXISTS
---------------
The imitation sweep in ``run_n100_invasion_count_sweep`` reads the *frequency*
of a strategy after a fixed number of generations. That is a finite-time
transient of a deterministic imitation process, and the manuscript already
states the consequence: it "cannot measure fixation probability or establish
evolutionary stability".

This module implements the alternative used by Schmid et al. (2023),
*Quantitative assessment can stabilize indirect reciprocity under imperfect
information*, Nat. Commun. 14:2086 (doi:10.1038/s41467-023-37817-x). Their
stability claim rests on three steps, all reproduced here:

1. **Two-type mixtures only.** For every composition ``k`` mutants + ``N-k``
   residents, simulate the reputation dynamics. The paper is explicit that it
   "does not consider the coexistence of more than two strategies" for this
   analysis, which avoids the three-way reputation ambiguity of the norm set.

2. **Stationary payoffs.** Reputations are *not* reset between rounds. The
   process is run to stationarity first (burn-in), then payoffs are estimated
   from the stationary phase. Payoff of player ``i`` is

       pi_i = 1/(N-1) * sum_j ( b * x_ji - c * x_ij )

   with ``x_ij`` the stationary rate at which ``i`` cooperates towards ``j``.
   For a simultaneous Prisoner's Dilemma with ``b=2, c=1`` this equals the mean
   payoff per interaction, which is what this module accumulates directly.

3. **Fixation probability, not a fate after 50 generations.** With
   ``d_k = pi_M(k) - pi_R(k)`` the Traulsen-Hauert fixation probability of a
   mutant M in a resident population R is

       rho_MR = 1 / (1 + sum_{i=1}^{N-1} prod_{k=1}^{i} exp(-beta * d_k))

   ``rho > 1/N`` means the mutant is favoured, ``rho < 1/N`` means it is
   suppressed.

OUTPUT
------
A ``rho`` row between the candidate and every probe, measured on a standing
reputation distribution rather than a per-generation reset.

Usage::

    uv run python -m experiments.analysis.invasion.run_fixation_benchmark \
        --candidate mine=agent-type1=path/to/strategy.py \
        --probes ALLC ALLD L1 L2 L3 L4 L5 L6 L7 L8 \
        --population-size 50 --beta 1.0 \
        --workers 48 --output results/quantitative_baseline/fixation/mine
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from experiments.v2_quantitative.baselines import BASELINES

from ..paths import quantitative_results_dir
from .core import NORMS, Competitor, EvolvedSource, write_json_atomic
from .run_n100_invasion_count_sweep import load_representative_from_path


DEFAULT_BURN_IN = 10_000
DEFAULT_MEASURE = 10_000
DEFAULT_BETA = 1.0
DEFAULT_POPULATION = 50
PROBE_CHOICES = ("ALLC", "ALLD", *NORMS)

# Canonical config key names. The writer and the cache predicate must agree
# exactly; deriving both from these constants is what stops them drifting
# (an earlier version wrote "burn_in_interactions" but compared "burn_in",
# so the cache never hit).
KEY_POPULATION = "population_size"
KEY_BURN_IN = "burn_in_interactions"
KEY_MEASURE = "measure_interactions"
KEY_BETA = "beta"
KEY_ACTION_ERROR = "action_error_probability"
KEY_OBSERVATION_ERROR = "observation_error_probability"

# Stationarity is the load-bearing assumption, so every result carries the
# block-wise payoff series. Inflate --burn-in / --measure until the two series
# agree to within the precision you need; the fix is cheap (a probe costs a few
# seconds at N=20) whereas an unconverged payoff difference feeds straight into
# the exponential of the fixation formula.


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_output() -> Path:
    return quantitative_results_dir() / "fixation" / "fixation_benchmark"


def load_candidate(label: str, agent_type: str, raw_path: str) -> EvolvedSource:
    """Load a candidate from a ``.py`` file or an ``evolutionary.json`` run."""
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"strategy file not found for {label}: {path}")
    if path.suffix == ".json":
        return load_representative_from_path(label, agent_type, path)
    code = path.read_text(encoding="utf-8")
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


def parse_candidate(values: list[str]) -> tuple[str, EvolvedSource]:
    if len(values) != 1:
        raise ValueError("exactly one --candidate LABEL=[AGENT_TYPE=]PATH is required")
    parts = values[0].split("=", 2)
    if len(parts) == 3:
        label, agent_type, raw = parts
    elif len(parts) == 2:
        label, raw = parts
        agent_type = "agent-type1"
    else:
        raise ValueError(
            f"Invalid --candidate {values[0]!r}; expected LABEL=[AGENT_TYPE=]PATH"
        )
    if not label:
        raise ValueError("candidate label must be non-empty")
    return label, load_candidate(label, agent_type, raw)


def probe_source(probe: str) -> EvolvedSource:
    """A canonical norm probe as an EvolvedSource carrying the baseline code."""
    code = BASELINES[probe]
    return EvolvedSource(
        agent_type="agent-type1",
        path=Path(f"<baseline:{probe}>"),
        agent_id=-1,
        lineage_id=-1,
        root_lineage_id=-1,
        root_family_size=0,
        fitness=0.0,
        code=code,
        code_sha256=_sha(code),
    )


# ---------------------------------------------------------------------------
# Stationary two-type mixture
# ---------------------------------------------------------------------------
def _flip(action: str, rng: random.Random, probability: float) -> str:
    if probability <= 0.0 or rng.random() >= probability:
        return action
    return "defect" if action == "cooperate" else "cooperate"


def stationary_mixture(
    mutant: EvolvedSource,
    resident: EvolvedSource,
    mutant_count: int,
    seed: int,
    population_size: int = DEFAULT_POPULATION,
    burn_in: int = DEFAULT_BURN_IN,
    measure: int = DEFAULT_MEASURE,
    beta: float = DEFAULT_BETA,
    action_error: float = 0.0,
    observation_error: float = 0.0,
    block_size: int = 2_000,
) -> dict[str, Any]:
    """Run one composition to stationarity and return the two type payoffs.

    ``burn_in`` interactions are played before any measurement; the following
    ``measure`` interactions are accumulated. Reputations are never reset, so
    the measured phase is drawn from the (approximately) stationary reputation
    distribution rather than a transient.

    Returns the payoff per participation for each type, plus block-wise payoffs
    so callers can check that stationarity was actually reached.
    """
    if not 1 <= mutant_count < population_size:
        raise ValueError("mutant_count must be in 1..N-1")
    resident_count = population_size - mutant_count
    rng = random.Random(seed)

    # Randomised placement so slot identity cannot bias the result.
    slots = list(range(population_size))
    rng.shuffle(slots)
    mutant_slots = set(slots[:mutant_count])
    population = [
        Competitor.create(
            slot,
            "mutant" if slot in mutant_slots else "resident",
            f"m{mutant_count}",
            mutant if slot in mutant_slots else resident,
        )
        for slot in slots
    ]

    payoff_total = {"mutant": 0.0, "resident": 0.0}
    participation = {"mutant": 0, "resident": 0}
    block_payoffs: list[dict[str, float]] = []
    block_total = {"mutant": 0.0, "resident": 0.0}
    block_participation = {"mutant": 0, "resident": 0}
    next_block = burn_in + block_size

    total_rounds = burn_in + measure
    cooperation_events = 0
    completed = 0
    while completed < total_rounds:
        order = list(range(population_size))
        rng.shuffle(order)
        for offset in range(0, population_size - 1, 2):
            first, second = population[order[offset]], population[order[offset + 1]]
            first_intended = "cooperate" if first.choose(second.agent_id) else "defect"
            second_intended = "cooperate" if second.choose(first.agent_id) else "defect"
            first_action = _flip(first_intended, rng, action_error)
            second_action = _flip(second_intended, rng, action_error)
            first_coop = first_action == "cooperate"
            second_coop = second_action == "cooperate"

            payoffs = (
                (first, 2 * int(second_coop) - int(first_coop), first_coop),
                (second, 2 * int(first_coop) - int(second_coop), second_coop),
            )
            if completed >= burn_in:
                for member, payoff, _ in payoffs:
                    payoff_total[member.kind] += payoff
                    participation[member.kind] += 1
                    block_total[member.kind] += payoff
                    block_participation[member.kind] += 1
                cooperation_events += int(first_coop) + int(second_coop)

            # Observers judge with private, possibly noisy, information.
            for observer in population:
                seen_first = _flip(first_action, rng, observation_error)
                seen_second = _flip(second_action, rng, observation_error)
                if observer.agent_id == second.agent_id:
                    observer.observe(
                        second.agent_id, seen_second, first.agent_id, seen_first
                    )
                else:
                    observer.observe(
                        first.agent_id, seen_first, second.agent_id, seen_second
                    )
            completed += 1

        if completed >= next_block or completed >= total_rounds:
            block_payoffs.append({
                kind: (
                    block_total[kind] / block_participation[kind]
                    if block_participation[kind] else 0.0
                )
                for kind in ("mutant", "resident")
            })
            block_total = {"mutant": 0.0, "resident": 0.0}
            block_participation = {"mutant": 0, "resident": 0}
            next_block = completed + block_size

    payoff_mutant = (
        payoff_total["mutant"] / participation["mutant"]
        if participation["mutant"] else 0.0
    )
    payoff_resident = (
        payoff_total["resident"] / participation["resident"]
        if participation["resident"] else 0.0
    )
    return {
        "mutant_count": mutant_count,
        "resident_count": resident_count,
        "payoff_mutant": payoff_mutant,
        "payoff_resident": payoff_resident,
        "payoff_difference": payoff_mutant - payoff_resident,
        "mutant_participation": participation["mutant"],
        "resident_participation": participation["resident"],
        "cooperation_rate": (
            cooperation_events / (2 * measure) if measure else 0.0
        ),
        "block_payoffs": block_payoffs,
        "seed": seed,
        "beta": beta,
    }


def fixation_probability(payoff_differences: list[float], beta: float) -> float:
    """Traulsen-Hauert fixation probability rho_MR.

    ``payoff_differences[k-1]`` must be ``pi_M(k) - pi_R(k)`` for
    ``k = 1, ..., N-1``. Higher mutant payoff lowers every exponentiated factor,
    shrinking the sum and raising rho.
    """
    if not payoff_differences:
        raise ValueError("payoff_differences must be non-empty")
    total = 0.0
    product = 1.0
    for difference in payoff_differences:
        product *= math.exp(-beta * difference)
        total += product
    return 1.0 / (1.0 + total)


def payoff_difference_curve(
    mutant: EvolvedSource,
    resident: EvolvedSource,
    seed: int,
    population_size: int,
    burn_in: int,
    measure: int,
    beta: float,
    action_error: float,
    observation_error: float,
    workers: int,
    replicates: int = 1,
) -> list[dict[str, Any]]:
    """Sweep every two-type composition, in parallel, and return the curve.

    ``replicates`` independent seeds are run per composition and averaged.

    Replication is not a luxury here. At compositions where one type is a tiny
    minority the reputation dynamics can settle into more than one basin (e.g.
    "everyone cooperates" versus "the minority is ostracised"), and the
    transition between them is slow. A single run then reports whichever basin
    it happened to fall into, so the payoff difference is a draw from a
    bimodal distribution rather than a stationary value. Averaging over
    replicates estimates the mean of that distribution and the recorded
    standard error exposes how wide it is.
    """
    payloads = [
        (mutant, resident, k, seed + 1_000_003 * rep + k, population_size,
         burn_in, measure, beta, action_error, observation_error)
        for k in range(1, population_size)
        for rep in range(replicates)
    ]
    results: list[dict[str, Any] | None] = [None] * len(payloads)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(stationary_mixture, *payload): index
            for index, payload in enumerate(payloads)
        }
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
    runs = [r for r in results if r is not None]

    if replicates == 1:
        return runs

    by_composition: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        by_composition.setdefault(run["mutant_count"], []).append(run)

    curve: list[dict[str, Any]] = []
    for k in range(1, population_size):
        group = by_composition.get(k, [])
        if not group:
            continue
        diffs = [g["payoff_difference"] for g in group]
        mean = sum(diffs) / len(diffs)
        variance = (
            sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
            if len(diffs) > 1 else 0.0
        )
        base = group[0]
        curve.append({
            **base,
            "payoff_mutant": sum(g["payoff_mutant"] for g in group) / len(group),
            "payoff_resident": sum(g["payoff_resident"] for g in group) / len(group),
            "payoff_difference": mean,
            "payoff_difference_std": math.sqrt(variance),
            "replicates": len(group),
            "replicate_differences": diffs,
            # Concatenate block series so stationarity is judged over all
            # replicates rather than one arbitrary draw.
            "block_payoffs": [b for g in group for b in g["block_payoffs"]],
        })
    return curve


def _summarise_curve(curve: list[dict[str, Any]], beta: float, n: int) -> dict[str, Any]:
    differences = [c["payoff_difference"] for c in curve]
    rho = fixation_probability(differences, beta)
    summary = {
        "rho": rho,
        "neutral_rho": 1.0 / n,
        "advantage": rho - 1.0 / n,
        "mean_payoff_difference": sum(differences) / len(differences),
        "min_payoff_difference": min(differences),
        "max_payoff_difference": max(differences),
        "curve": curve,
    }
    # How much would rho move if each composition had landed in its other
    # basin? Report the extreme reachable rhos from the per-composition spread.
    spreads = [c.get("payoff_difference_std", 0.0) for c in curve]
    if any(s > 0 for s in spreads):
        summary["rho_at_minus_1sd"] = fixation_probability(
            [d - s for d, s in zip(differences, spreads)], beta
        )
        summary["rho_at_plus_1sd"] = fixation_probability(
            [d + s for d, s in zip(differences, spreads)], beta
        )
        summary["max_composition_std"] = max(spreads)
    return summary


def benchmark_pair(
    candidate: EvolvedSource,
    candidate_label: str,
    probe: str,
    population_size: int,
    burn_in: int,
    measure: int,
    beta: float,
    action_error: float,
    observation_error: float,
    workers: int,
    base_seed: int,
    replicates: int = 1,
) -> dict[str, Any]:
    """Fixation probability of the candidate against one probe, from one sweep.

    ``curve[k-1]`` holds ``pi_mutant(k) - pi_resident(k)`` for the composition
    with ``k`` candidates, which is exactly what the Traulsen--Hauert formula
    needs for "candidate invades probe". The opposite ordering is a different
    experiment and is not inferred here.
    """
    source_probe = probe_source(probe)
    curve = payoff_difference_curve(
        candidate, source_probe, base_seed, population_size, burn_in, measure,
        beta, action_error, observation_error, workers, replicates,
    )
    return {
        "probe": probe,
        "candidate_invades_probe": _summarise_curve(curve, beta, population_size),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _verdict(rho: float, neutral: float, tol: float = 0.01) -> str:
    if rho > neutral + tol:
        return "invades >neutral"
    if rho < neutral - tol:
        return "blocked <neutral"
    return "neutral"


def cache_matches(
    existing: dict[str, Any], args: Any, candidate: EvolvedSource, label: str,
) -> bool:
    """True when a stored benchmark used the same settings and the same strategy.

    ``seed`` is deliberately NOT part of the key: the process is seeded
    deterministically from ``--seed + k``, so changing only the seed changes the
    numbers, but including it here would make ``--seed`` alone silently
    invalidate every cached file. Callers who want a different realisation
    should pass ``--force``.
    """
    cfg = existing.get("config", {})
    return bool(
        cfg.get(KEY_POPULATION) == args.population_size
        and cfg.get(KEY_BURN_IN) == args.burn_in
        and cfg.get(KEY_MEASURE) == args.measure
        and cfg.get(KEY_BETA) == args.beta
        and cfg.get(KEY_ACTION_ERROR) == args.action_error
        and cfg.get(KEY_OBSERVATION_ERROR) == args.observation_error
        and cfg.get("replicates", 1) == args.replicates
        and existing.get("candidate", {}).get("label") == label
        and existing.get("candidate", {}).get("code_sha256") == candidate.code_sha256
        and sorted(existing.get("probes", [])) == sorted(args.probes)
    )


def stationarity_report(curve: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise how far the measured payoffs are from stationarity.

    Each composition's measurement window is split into halves and the mean
    payoff difference of each half is compared. A large disagreement means the
    reputation system was still drifting, so the recorded payoff is a transient
    average rather than a stationary one, and the resulting ``rho`` should not
    be trusted for that composition.

    The drift is not uniform: it is concentrated at the extreme compositions,
    where one type is a very small minority and its reputation takes longest to
    settle. ``worst_composition`` identifies it so ``--burn-in`` can be raised
    where it matters.
    """
    worst = 0.0
    worst_at: dict[str, Any] | None = None
    drifting: list[int] = []
    for entry in curve:
        blocks = entry["block_payoffs"]
        if len(blocks) < 2:
            continue
        diffs = [b["mutant"] - b["resident"] for b in blocks]
        half = len(diffs) // 2
        first = sum(diffs[:half]) / half
        second = sum(diffs[half:]) / (len(diffs) - half)
        gap = abs(first - second)
        if gap > worst:
            worst = gap
            worst_at = {"mutant_count": entry["mutant_count"], "gap": gap}
        if gap > 0.10:
            drifting.append(entry["mutant_count"])
    return {
        "max_half_to_half_gap": worst,
        "worst_composition": worst_at,
        "drifting_compositions": drifting,
        "block_size_assumed_pairs": len(curve[0]["block_payoffs"]) if curve else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", default=[],
        metavar="LABEL=[AGENT_TYPE=]PATH",
        help="Candidate strategy (.py or evolutionary.json).",
    )
    parser.add_argument(
        "--probes", nargs="+", choices=PROBE_CHOICES,
        default=["ALLC", "ALLD", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--burn-in", type=int, default=DEFAULT_BURN_IN)
    parser.add_argument("--measure", type=int, default=DEFAULT_MEASURE)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA,
                        help="Selection strength in the fixation formula.")
    parser.add_argument("--action-error", type=float, default=0.0)
    parser.add_argument("--observation-error", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--replicates", type=int, default=5,
        help=(
            "Independent seeds per composition, averaged to estimate the payoff "
            "difference. At extreme compositions the reputation dynamics can "
            "settle into more than one basin, so a single run is a draw from a "
            "bimodal distribution. 5 is a practical floor; the paper used long "
            "single runs instead."
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.candidate:
        parser.error("--candidate LABEL=[AGENT_TYPE=]PATH is required")
    for name, value in (("--action-error", args.action_error),
                        ("--observation-error", args.observation_error)):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be in [0, 1]")
    if args.population_size < 3:
        parser.error("--population-size must be >= 3")
    if args.burn_in < 1 or args.measure < 1:
        parser.error("--burn-in and --measure must be positive")
    if args.replicates < 1:
        parser.error("--replicates must be >= 1")
    try:
        label, candidate = parse_candidate(args.candidate)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    output = args.output or (default_output() / label)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "fixation_benchmark.json"

    if result_path.exists() and not args.force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if cache_matches(existing, args, candidate, label):
            print(f"cached: {result_path} (use --force to recompute)")
            _print_report(existing, label)
            return

    started = time.perf_counter()
    print(
        f"=== fixation benchmark: {label} vs {len(args.probes)} probes, "
        f"N={args.population_size}, "
        f"{args.population_size - 1} compositions/probe x "
        f"{args.replicates} replicate(s) ===",
        flush=True,
    )
    results: dict[str, Any] = {}
    for probe in args.probes:
        t0 = time.perf_counter()
        results[probe] = benchmark_pair(
            candidate, label, probe, args.population_size, args.burn_in,
            args.measure, args.beta, args.action_error, args.observation_error,
            args.workers, args.seed, args.replicates,
        )
        rho = results[probe]["candidate_invades_probe"]["rho"]
        neutral = 1.0 / args.population_size
        print(
            f"  {probe:5s}  {label}->{probe}: rho={rho:.4f} "
            f"({_verdict(rho, neutral)})   "
            f"[{time.perf_counter() - t0:.1f}s]",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "experiment": "schmid_style_fixation_benchmark",
        "method": {
            "source": "Schmid et al. 2023, Nat Commun 14:2086",
            "doi": "10.1038/s41467-023-37817-x",
            "steps": [
                "two-type mixture for every composition k in 1..N-1",
                "reputations not reset; payoffs measured after a burn-in phase",
                "Traulsen-Hauert fixation probability from the payoff-difference curve",
            ],
        },
        "candidate": {
            "label": label,
            "agent_type": candidate.agent_type,
            "path": str(candidate.path),
            "code_sha256": candidate.code_sha256,
        },
        "probes": list(args.probes),
        "config": {
            KEY_POPULATION: args.population_size,
            KEY_BURN_IN: args.burn_in,
            KEY_MEASURE: args.measure,
            KEY_BETA: args.beta,
            "neutral_fixation_probability": 1.0 / args.population_size,
            KEY_ACTION_ERROR: args.action_error,
            KEY_OBSERVATION_ERROR: args.observation_error,
            "seed": args.seed,
            "replicates": args.replicates,
            "reputation_reset_between_rounds": False,
        },
        "stationarity": {
            probe: stationarity_report(entry["candidate_invades_probe"]["curve"])
            for probe, entry in results.items()
        },
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(result_path, payload)
    print(f"Wrote {result_path}  ({payload['elapsed_seconds']:.1f}s)")
    print()
    _print_report(payload, label)
    _warn_if_not_stationary(payload)


def _warn_if_not_stationary(payload: dict[str, Any]) -> None:
    reports = payload.get("stationarity", {})
    bad = {
        probe: rep for probe, rep in reports.items()
        if rep["max_half_to_half_gap"] > 0.10
    }
    if not bad:
        print()
        print("stationarity: OK (every composition's two half-windows agree "
              "to within 0.10 payoff units)")
        return
    print()
    print("!! STATIONARITY WARNING --------------------------------------------")
    print("These probes still drifted during the measurement window, so their")
    print("payoff differences (and therefore rho) are not converged:")
    for probe, rep in sorted(bad.items(), key=lambda kv: -kv[1]["max_half_to_half_gap"]):
        worst = rep["worst_composition"] or {}
        print(f"   {probe:>5}  max gap {rep['max_half_to_half_gap']:.3f} "
              f"at k={worst.get('mutant_count')}  "
              f"(drifting k = {rep['drifting_compositions']})")
    print("Raise --burn-in and/or --measure and re-run with --force.")
    print("-------------------------------------------------------------------")


def _print_report(payload: dict[str, Any], label: str) -> None:
    neutral = payload["config"]["neutral_fixation_probability"]
    print()
    print(f"=== {label}: invasion ability (neutral rho = {neutral:.4f}) ===")
    print(f"{'probe':>6} {label + '->probe':>16} {'verdict':>18}")
    for probe, entry in payload["results"].items():
        forward = entry["candidate_invades_probe"]["rho"]
        print(f"{probe:>6} {forward:>16.4f} {_verdict(forward, neutral):>18}")


if __name__ == "__main__":
    main()
