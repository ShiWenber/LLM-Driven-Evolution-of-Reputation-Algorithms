"""Tests for the Schmid-style fixation benchmark."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from experiments.analysis.invasion.run_fixation_benchmark import (
    KEY_BURN_IN,
    KEY_MEASURE,
    KEY_POPULATION,
    cache_matches,
    fixation_probability,
    probe_source,
    stationarity_report,
    stationary_mixture,
)
from experiments.v2_quantitative.baselines import BASELINES


# ---------------------------------------------------------------------------
# The invasion experiment has no direction dimension
# ---------------------------------------------------------------------------
def test_invasion_sweep_has_no_direction_dimension():
    """One frequency axis only: no direction constant, and no direction in paths."""
    from experiments.analysis.invasion import core as invasion_core
    from experiments.analysis.invasion import run_invasion as uni

    assert not hasattr(invasion_core, "DIRECTION")
    path = uni.result_path(Path("/out"), "label", "L1", 5, 0)
    # The resident is folded into the path so distinct pairs cannot collide.
    assert path == Path("/out/label/invades_L1/n5_seed0/invasion.json")


def test_archived_results_are_still_found_for_cache_reuse(tmp_path):
    """Archived layouts stay readable so re-runs hit the cache."""
    from experiments.analysis.invasion import run_invasion as uni

    # Old flat sweep layout: <label>/<norm>/n<k>_seed<s>/
    flat = tmp_path / "label" / "L1" / "n5_seed0" / "invasion.json"
    flat.parent.mkdir(parents=True)
    flat.write_text("{}", encoding="utf-8")
    assert uni.existing_result_path(tmp_path, "label", "L1", 5, 0) == flat

    # Old direction-nested layout: <label>/<direction>/<norm>/n<k>_seed<s>/
    nested = (
        tmp_path / "label" / uni.ARCHIVED_DIRECTION_LABEL / "L2"
        / "n5_seed0" / "invasion.json"
    )
    nested.parent.mkdir(parents=True)
    nested.write_text("{}", encoding="utf-8")
    assert uni.existing_result_path(tmp_path, "label", "L2", 5, 0) == nested

    # With no archived copy the current path is used.
    assert uni.existing_result_path(tmp_path, "label", "L3", 5, 0) == (
        uni.result_path(tmp_path, "label", "L3", 5, 0)
    )


def test_plotter_accepts_flat_and_archived_summaries():
    """The plotter reads both layouts, but only one panel row either way."""
    from experiments.analysis.plot_n100_invasion_count_sweep import _norm_groups

    cell = lambda: {"runs": 3, "fixations": 0, "extinctions": 0,
                    "mean_final_invader_frequency": 0.5}
    flat_cells = {"L1": {"1": cell(), "5": cell()}, "ALLC": {"1": cell(), "5": cell()}}
    archived_cells = {"evolved_invades_norm": flat_cells}

    assert _norm_groups({"groups": {"seed0": flat_cells}}, "seed0") == flat_cells
    assert _norm_groups(
        {"groups": {"seed0": archived_cells}}, "seed0"
    ) == flat_cells


# ---------------------------------------------------------------------------
# Fixation probability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [5, 10, 50])
def test_flat_payoffs_give_exactly_neutral_fixation(n: int):
    """d_k = 0 for all k must give rho = 1/N exactly.

    Each exponentiated factor is 1, so the sum is N-1 and rho = 1/(1+N-1).
    This is the analytic check that the formula's indexing is right.
    """
    rho = fixation_probability([0.0] * (n - 1), beta=1.0)
    assert rho == pytest.approx(1.0 / n, rel=1e-12)


def test_positive_payoff_advantage_is_favoured():
    n = 20
    neutral = 1.0 / n
    favoured = fixation_probability([0.2] * (n - 1), beta=1.0)
    suppressed = fixation_probability([-0.2] * (n - 1), beta=1.0)
    assert favoured > neutral > suppressed
    # A small per-step edge accumulates but does not make fixation certain:
    # sum_i e^{-0.2 i} ~ 4.4 over 19 steps, so rho ~ 0.185.
    assert favoured == pytest.approx(1.0 / (1.0 + 4.416), rel=0.02)
    # Only a large edge drives rho near certainty.
    assert fixation_probability([3.0] * (n - 1), beta=1.0) > 0.9
    assert fixation_probability([-3.0] * (n - 1), beta=1.0) < 0.1


def test_fixation_probability_is_monotone_in_the_advantage():
    n = 12
    values = [
        fixation_probability([d] * (n - 1), beta=1.0)
        for d in (-0.3, -0.1, 0.0, 0.1, 0.3)
    ]
    assert values == sorted(values)
    assert values[2] == pytest.approx(1.0 / n, rel=1e-12)


def test_beta_scales_the_response_to_the_same_advantage():
    diffs = [0.1] * 9
    weak = fixation_probability(diffs, beta=0.5)
    strong = fixation_probability(diffs, beta=5.0)
    assert strong > weak


def test_fixation_probability_stays_a_probability():
    for diffs in ([-5.0] * 9, [5.0] * 9, [0.0] * 9, list(range(-4, 5))):
        rho = fixation_probability(diffs, beta=3.0)
        assert 0.0 <= rho <= 1.0
        assert math.isfinite(rho)


def test_empty_curve_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        fixation_probability([], beta=1.0)


# ---------------------------------------------------------------------------
# Probe construction and the stationary mixture
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("probe", ["ALLC", "ALLD", "L1", "L8"])
def test_probe_source_carries_the_published_baseline_code(probe: str):
    source = probe_source(probe)
    assert source.code == BASELINES[probe]
    assert source.agent_type == "agent-type1"


def test_unknown_probe_is_rejected():
    with pytest.raises(KeyError):
        probe_source("NOT_A_NORM")


def test_stationary_mixture_reports_both_type_payoffs_without_resetting():
    """Reputations must accumulate across the run, so payoffs are stationary."""
    result = stationary_mixture(
        mutant=probe_source("L1"),
        resident=probe_source("ALLD"),
        mutant_count=3,
        seed=0,
        population_size=8,
        burn_in=200,
        measure=200,
        beta=1.0,
        block_size=100,
    )
    assert result["mutant_count"] == 3
    assert result["resident_count"] == 5
    assert result["mutant_participation"] > 0
    assert result["resident_participation"] > 0
    # Block series exists and is finite, so stationarity is inspectable.
    assert len(result["block_payoffs"]) >= 2
    for block in result["block_payoffs"]:
        assert set(block) == {"mutant", "resident"}
        assert all(math.isfinite(v) for v in block.values())


def test_all_cooperators_are_pairwise_neutral():
    """ALLC vs ALLC is the analytic neutral case: every payoff difference is 0."""
    result = stationary_mixture(
        mutant=probe_source("ALLC"),
        resident=probe_source("ALLC"),
        mutant_count=4,
        seed=1,
        population_size=8,
        burn_in=100,
        measure=100,
        block_size=100,
    )
    assert result["payoff_difference"] == pytest.approx(0.0, abs=1e-12)
    assert fixation_probability([result["payoff_difference"]] * 7, 1.0) == pytest.approx(
        1.0 / 8, rel=1e-12
    )


def test_mixture_rejects_out_of_range_counts():
    with pytest.raises(ValueError, match="1..N-1"):
        stationary_mixture(
            probe_source("L1"), probe_source("L1"), 0, 0, population_size=8
        )
    with pytest.raises(ValueError, match="1..N-1"):
        stationary_mixture(
            probe_source("L1"), probe_source("L1"), 8, 0, population_size=8
        )


# ---------------------------------------------------------------------------
# Cache key consistency
#
# An earlier version wrote "burn_in_interactions"/"measure_interactions" but
# compared "burn_in"/"measure", so the predicate was never true and every rerun
# silently recomputed. These tests pin the writer's key names to the predicate.
# ---------------------------------------------------------------------------
class _Args:
    def __init__(self, **kw):
        self.population_size = kw.get("population_size", 50)
        self.burn_in = kw.get("burn_in", 1000)
        self.measure = kw.get("measure", 1000)
        self.beta = kw.get("beta", 1.0)
        self.action_error = kw.get("action_error", 0.0)
        self.observation_error = kw.get("observation_error", 0.0)
        self.probes = kw.get("probes", ["L1", "ALLD"])
        self.replicates = kw.get("replicates", 1)


def _stored(candidate, args, probes=None):
    """Exactly the config block the writer emits, using the same constants."""
    return {
        "config": {
            KEY_POPULATION: args.population_size,
            KEY_BURN_IN: args.burn_in,
            KEY_MEASURE: args.measure,
            "beta": args.beta,
            "action_error_probability": args.action_error,
            "observation_error_probability": args.observation_error,
            "replicates": args.replicates,
        },
        "candidate": {"label": "c", "code_sha256": candidate.code_sha256},
        "probes": list(probes if probes is not None else args.probes),
    }


def test_cache_matches_its_own_written_config():
    """Round-trip: what the writer stores must satisfy the predicate."""
    candidate = probe_source("L1")
    args = _Args()
    assert cache_matches(_stored(candidate, args), args, candidate, "c") is True


def test_cache_key_names_are_the_documented_ones():
    """Guards against the writer/predicate drift that disabled the cache."""
    assert (KEY_POPULATION, KEY_BURN_IN, KEY_MEASURE) == (
        "population_size", "burn_in_interactions", "measure_interactions"
    )


@pytest.mark.parametrize("field, value", [
    ("population_size", 12),
    ("burn_in", 999),
    ("measure", 999),
    ("beta", 2.0),
    ("action_error", 0.05),
    ("observation_error", 0.05),
    ("replicates", 4),
])
def test_cache_rejects_a_changed_setting(field, value):
    candidate = probe_source("L1")
    args = _Args()
    stored = _stored(candidate, args)
    assert cache_matches(stored, args, candidate, "c") is True
    stored["config"][{
        "population_size": KEY_POPULATION,
        "burn_in": KEY_BURN_IN,
        "measure": KEY_MEASURE,
        "beta": "beta",
        "action_error": "action_error_probability",
        "observation_error": "observation_error_probability",
        "replicates": "replicates",
    }[field]] = value
    assert cache_matches(stored, args, candidate, "c") is False


def test_cache_rejects_a_different_strategy():
    args = _Args()
    stored = _stored(probe_source("L1"), args)
    assert cache_matches(stored, args, probe_source("ALLD"), "c") is False


def test_cache_rejects_a_different_label():
    candidate = probe_source("L1")
    args = _Args()
    assert cache_matches(_stored(candidate, args), args, candidate, "other") is False


def test_cache_rejects_a_different_probe_set():
    candidate = probe_source("L1")
    args = _Args()
    stored = _stored(candidate, args, probes=["L1"])
    assert cache_matches(stored, args, candidate, "c") is False


# ---------------------------------------------------------------------------
# Stationarity reporting
# ---------------------------------------------------------------------------
def _entry(k, diffs):
    return {
        "mutant_count": k,
        "block_payoffs": [{"mutant": 1.0 + d, "resident": 1.0} for d in diffs],
    }


def test_stationarity_report_flags_a_drifting_composition():
    curve = [
        _entry(1, [0.0, 0.0, 0.0, 0.0]),          # flat
        _entry(2, [-0.5, -0.4, 0.9, 1.0]),        # drifting
    ]
    report = stationarity_report(curve)
    assert report["max_half_to_half_gap"] > 0.10
    assert report["worst_composition"]["mutant_count"] == 2
    assert report["drifting_compositions"] == [2]


def test_stationarity_report_is_clean_for_flat_curves():
    curve = [_entry(k, [0.25, 0.25, 0.25, 0.25]) for k in range(1, 5)]
    report = stationarity_report(curve)
    assert report["max_half_to_half_gap"] == pytest.approx(0.0, abs=1e-12)
    assert report["drifting_compositions"] == []


def test_stationarity_report_tolerates_single_block_curves():
    report = stationarity_report([_entry(1, [0.5])])
    assert report["max_half_to_half_gap"] == 0.0
    assert report["drifting_compositions"] == []


def test_stationarity_report_handles_an_empty_curve():
    report = stationarity_report([])
    assert report["max_half_to_half_gap"] == 0.0
    assert report["worst_composition"] is None


# ---------------------------------------------------------------------------
# Replicate averaging
#
# Extreme compositions are bimodal (e.g. "everyone cooperates" vs "the minority
# is ostracised"), so a single run is a draw, not an estimate. These tests use
# a stub so the averaging logic is checked without paying for real simulations.
# ---------------------------------------------------------------------------
class InlinePool:
    """Runs submitted work immediately instead of spawning processes.

    Lets the averaging logic be tested without pickling locally-defined stubs.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        import concurrent.futures

        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # pragma: no cover - defensive
            future.set_exception(exc)
        return future


@pytest.fixture
def inline_pool(monkeypatch):
    import experiments.analysis.invasion.run_fixation_benchmark as fb

    monkeypatch.setattr(fb.concurrent.futures, "ProcessPoolExecutor", InlinePool)
    return fb


def test_single_replicate_returns_raw_runs(inline_pool, monkeypatch):
    fb = inline_pool
    calls = []

    def fake(mutant, resident, k, seed, *rest, **kw):
        calls.append((k, seed))
        return {
            "mutant_count": k, "payoff_difference": 0.5, "payoff_mutant": 1.0,
            "payoff_resident": 0.5, "block_payoffs": [],
        }

    monkeypatch.setattr(fb, "stationary_mixture", fake)
    curve = fb.payoff_difference_curve(
        probe_source("L1"), probe_source("ALLD"), 0, 4, 10, 10, 1.0, 0.0, 0.0,
        workers=1, replicates=1,
    )
    assert [c["mutant_count"] for c in curve] == [1, 2, 3]
    assert len(calls) == 3
    assert all("payoff_difference_std" not in c for c in curve)


def test_replicates_are_averaged_with_a_recorded_spread(inline_pool, monkeypatch):
    fb = inline_pool

    def fake(mutant, resident, k, seed, *rest, **kw):
        # Deliberately bimodal: replicate parity decides the basin.
        value = 1.0 if seed % 2 == 0 else -1.0
        return {
            "mutant_count": k, "payoff_difference": value,
            "payoff_mutant": 1.0 + value, "payoff_resident": 1.0,
            "block_payoffs": [{"mutant": 1.0 + value, "resident": 1.0}],
        }

    monkeypatch.setattr(fb, "stationary_mixture", fake)
    curve = fb.payoff_difference_curve(
        probe_source("L1"), probe_source("ALLD"), 0, 4, 10, 10, 1.0, 0.0, 0.0,
        workers=1, replicates=2,
    )
    assert [c["mutant_count"] for c in curve] == [1, 2, 3]
    for entry in curve:
        assert entry["replicates"] == 2
        assert len(entry["replicate_differences"]) == 2
        assert entry["payoff_difference"] == pytest.approx(0.0, abs=1e-12)
        assert entry["payoff_difference_std"] == pytest.approx(1.4142135, rel=1e-6)


def test_replicate_blocks_are_concatenated_for_stationarity(inline_pool, monkeypatch):
    """Stationarity must be judged over every replicate, not one draw."""
    fb = inline_pool

    def fake_mixture(mutant, resident, k, seed, *rest, **kw):
        return {
            "mutant_count": k, "payoff_difference": 0.0, "payoff_mutant": 1.0,
            "payoff_resident": 1.0,
            "block_payoffs": [{"mutant": 1.0, "resident": 1.0}],
        }

    monkeypatch.setattr(fb, "stationary_mixture", fake_mixture)
    curve = fb.payoff_difference_curve(
        probe_source("L1"), probe_source("ALLD"), 0, 4, 10, 10, 1.0, 0.0,
        0.0, workers=1, replicates=3,
    )
    assert all(len(c["block_payoffs"]) == 3 for c in curve)


def test_summary_reports_rho_sensitivity_to_the_spread():
    import experiments.analysis.invasion.run_fixation_benchmark as fb

    curve = [
        {"payoff_difference": 0.0, "payoff_difference_std": 0.5},
        {"payoff_difference": 0.0, "payoff_difference_std": 0.0},
    ]
    summary = fb._summarise_curve(curve, beta=1.0, n=3)
    # A larger advantage raises rho, so the -1sd bound is the pessimistic one.
    assert summary["rho_at_minus_1sd"] < summary["rho"] < summary["rho_at_plus_1sd"]
    assert summary["max_composition_std"] == 0.5


def test_summary_omits_spread_fields_without_replicates():
    import experiments.analysis.invasion.run_fixation_benchmark as fb

    summary = fb._summarise_curve(
        [{"payoff_difference": 0.1}, {"payoff_difference": 0.1}], beta=1.0, n=3
    )
    assert "rho_at_minus_1sd" not in summary
    assert "max_composition_std" not in summary
