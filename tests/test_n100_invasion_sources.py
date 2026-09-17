import hashlib
from pathlib import Path

import pytest

from experiments.analysis.invasion.core import EvolvedSource, norm_source
from experiments.analysis.invasion.run_invasion import (
    cache_matches,
    load_candidate,
    noisy_output,
    parse_pairs,
    parse_sources,
)


def test_noisy_output_encodes_both_error_probabilities():
    path = noisy_output(0.01, 0.05)
    assert path.name == "n20_noisy_invasion_count_sweep_ae0p01_oe0p05"


def test_noisy_output_encodes_the_population_size():
    """The archived sweeps are N=100; the live default is N=20."""
    assert noisy_output(0.01, 0.01, 100).name == (
        "n100_noisy_invasion_count_sweep_ae0p01_oe0p01"
    )
    assert noisy_output(0.01, 0.01).name == (
        "n20_noisy_invasion_count_sweep_ae0p01_oe0p01"
    )


def test_source_agent_type_is_optional_and_defaults_to_type1(monkeypatch):
    """A bare ``LABEL=PATH`` is accepted for hand-written type-1 strategies."""
    monkeypatch.setattr(
        "experiments.analysis.invasion.run_invasion.load_candidate",
        lambda label, agent_type, raw_path: (label, agent_type, Path(raw_path)),
    )
    assert parse_sources(["seed0=some.py"]) == {
        "seed0": ("seed0", "agent-type1", Path("some.py"))
    }
    assert parse_sources(["seed0=agent-type2=run.json"]) == {
        "seed0": ("seed0", "agent-type2", Path("run.json"))
    }


def test_explicit_source_labels_are_unique(monkeypatch):
    monkeypatch.setattr(
        "experiments.analysis.invasion.run_invasion.load_candidate",
        lambda label, agent_type, raw_path: (label, agent_type, Path(raw_path)),
    )
    with pytest.raises(ValueError, match="unique"):
        parse_sources([
            "seed0=agent-type1=first.json",
            "seed0=agent-type1=second.json",
        ])


def test_load_candidate_dispatches_on_suffix(tmp_path):
    """One --source flag serves .py candidates and evolutionary.json runs."""
    strategy = tmp_path / "s.py"
    strategy.write_text(
        "def observe(A_rep, A_action, B_rep, B_action, my_reputation):\n"
        "    return A_rep\n"
        "def decide(my_reputation, opponent_reputation):\n"
        "    return True\n",
        encoding="utf-8",
    )
    source = load_candidate("s", "agent-type1", str(strategy))
    assert source.path == strategy.resolve()
    assert "def observe" in source.code

    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        load_candidate("m", "agent-type1", str(missing))


def test_pairs_filter_and_validation():
    sources = {"A": object(), "B": object(), "C": object()}
    assert parse_pairs(["A>B", "B>C", "A>B"], sources) == [("A", "B"), ("B", "C")]
    with pytest.raises(ValueError, match="Invalid pair"):
        parse_pairs(["A"], sources)
    with pytest.raises(ValueError, match="Unknown"):
        parse_pairs(["A>Z"], sources)
    with pytest.raises(ValueError, match="cannot invade itself"):
        parse_pairs(["A>A"], sources)


def _source(code: str) -> EvolvedSource:
    return EvolvedSource(
        agent_type="agent-type1", path=Path(f"<{code}>"), agent_id=-1,
        lineage_id=-1, root_lineage_id=-1, root_family_size=0, fitness=0.0,
        code=code, code_sha256=hashlib.sha256(code.encode()).hexdigest(),
    )


# Archived runs must stay cache-resident, otherwise every re-run recomputes
# thousands of simulations. They record `evolved_source` + `norm` and no
# resident hash, unlike results written by run_invasion.
# Args are (generations, interactions, fitness_window_fraction, ae, oe,
# population_size); the 0.2 share reproduces the archived 200/1000 burn-in
# split, and N=100 is the population every archived sweep was produced at.
_RUN_ARGS = (50, 1000, 0.2, 0.01, 0.01, 100)
_CONFIG = {
    "population_size": 100,
    "num_generations": 50,
    "interactions_per_generation": 1000,
    "fitness_interactions_per_generation": 200,
    "selection": "synchronous_deterministic_payoff_imitation",
    "updates_per_generation": 100,
    "generation_lifecycle": "fresh_agent_and_reputation_reset",
    "absorbing_state_early_stop": True,
    "action_error_probability": 0.01,
    "observation_error_probability": 0.01,
    # The matrix the archived sweeps were actually paid from (the engine
    # hardcoded it). _RUN_ARGS leaves benefit/cost at their defaults, which are
    # these same values, so an archived result stays cache-resident.
    "benefit": 2.0,
    "cost": 1.0,
}


def test_cache_rejects_a_result_recording_a_different_payoff_matrix():
    candidate = _source("candidate-code")
    l1 = norm_source("L1")
    stored = {
        "config": {**_CONFIG, "benefit": 3.0},
        "candidate_source": {"code_sha256": candidate.code_sha256},
        "norm": "L1",
    }
    assert not cache_matches(stored, candidate, l1, "L1", *_RUN_ARGS)


def test_cache_rejects_a_result_without_a_recorded_payoff_matrix():
    """Pre-change results recorded no matrix; they must be recomputed.

    Reusing them would silently apply whatever matrix the caller resolved now,
    which is exactly the mismatch that recording the matrix prevents.
    """
    candidate = _source("candidate-code")
    l1 = norm_source("L1")
    config = dict(_CONFIG)
    del config["benefit"]
    del config["cost"]
    stored = {
        "config": config,
        "candidate_source": {"code_sha256": candidate.code_sha256},
        "norm": "L1",
    }
    assert not cache_matches(stored, candidate, l1, "L1", *_RUN_ARGS)


def test_cache_matches_accepts_archived_norm_results():
    candidate = _source("candidate-code")
    l1, l2 = norm_source("L1"), norm_source("L2")
    archived = {
        "config": dict(_CONFIG),
        "evolved_source": {"code_sha256": candidate.code_sha256},
        "norm": "L1",
    }
    assert cache_matches(archived, candidate, l1, "L1", *_RUN_ARGS)
    assert not cache_matches(archived, candidate, l2, "L2", *_RUN_ARGS)
    assert not cache_matches(archived, _source("other"), l1, "L1", *_RUN_ARGS)
    # Different run parameters must invalidate the cache.
    assert not cache_matches(
        {"config": {**_CONFIG, "num_generations": 25},
         "evolved_source": {"code_sha256": candidate.code_sha256}, "norm": "L1"},
        candidate, l1, "L1", *_RUN_ARGS,
    )


def test_cache_matches_uses_resident_hash_when_present():
    """Results from run_invasion pin the resident by code hash."""
    candidate = _source("candidate-code")
    l1, l2 = norm_source("L1"), norm_source("L2")
    modern = {
        "config": dict(_CONFIG),
        "candidate_source": {"code_sha256": candidate.code_sha256},
        "resident_source": {"code_sha256": l1.code_sha256},
    }
    assert cache_matches(modern, candidate, l1, "L1", *_RUN_ARGS)
    # The hash wins over any label, so a mismatched resident is rejected even
    # if the caller passes the label recorded in the file.
    assert not cache_matches(modern, candidate, l2, "L1", *_RUN_ARGS)


def test_strategy_labels_covers_residents_that_never_invade():
    """``groups`` only keys the invader, so residents must be unioned in."""
    from experiments.analysis.plot_pairwise_invasion import strategy_labels

    # Archived pairwise summaries carry a `strategies` block.
    assert strategy_labels({"strategies": {"A": {}, "B": {}}}) == ["A", "B"]
    # Unified summaries carry `sources` + `residents`.
    assert strategy_labels(
        {"sources": {"A": {}}, "residents": {"A": {}, "B": {}},
         "groups": {"A": {"B": {}}}}
    ) == ["A", "B"]
    # Last resort: whatever the groups happen to contain.
    assert strategy_labels({"groups": {"A": {}}}) == ["A"]


def test_share_label_reads_the_seed_count_from_the_summary():
    """The y-label must not hard-code a seed count (it used to say "3 seeds")."""
    from experiments.analysis.plot_pairwise_invasion import share_label

    assert share_label({"seeds": [0, 1, 2]}) == "Mean final invader share (3 seeds)"
    assert share_label({"seeds": [0]}) == "Mean final invader share (1 seed)"
    assert share_label({"seeds": list(range(24))}) == (
        "Mean final invader share (24 seeds)"
    )
    # Archived summaries without a `seeds` block still get a usable label.
    assert share_label({}) == "Mean final invader share"
    assert share_label({"seeds": []}) == "Mean final invader share"
