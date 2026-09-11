"""Tests for the private-reputation heatmap and its replay/metric support.

The remaining consensus deliverable is the per-target reputation matrix heatmap.
It has two layers, both covered here:

* **core** (``experiments.analysis.consensus.core``) - loading a run's final
  population, instantiating a population from its codes, and the single
  ``disagreement`` metric that the heatmap captions are built from;
* **heatmap helpers** (``plot_private_reputation_matrix``) - the mask that
  blanks the diagonal and unseen entries, per-observer severity, the
  pairwise-disagreement matrix, and the render-loop plumbing.

The metric is a pure function over synthetic matrices, which is where the
subtle cases live (a constant matrix, a pure per-observer offset, partial
coverage, saturation at the clip bounds).
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.analysis.consensus import core
from experiments.v2_quantitative.baselines import BASELINES


def _matrix(n: int, fill) -> np.ndarray:
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = fill(i, j)
    return matrix


def _one_target_column(levels: np.ndarray) -> np.ndarray:
    """Observers ``0..m-1`` each rate a single target ``m`` that never rates back.

    Needed because ``_matrix`` masks the diagonal, so in an ``n x n`` matrix
    every column loses a *different* entry and no two columns share a value
    set.  Putting the lone target outside the observer range keeps the live
    values of that column exactly ``levels``.
    """
    m = len(levels)
    matrix = np.full((m + 1, m + 1), np.nan)
    for i, value in enumerate(levels):
        matrix[i, m] = value
    return matrix


# ---------------------------------------------------------------------------
# core: loading, pool construction, absolute differences
# ---------------------------------------------------------------------------


def test_disagreement_is_zero_for_a_perfectly_consistent_matrix():
    # Every observer gives every target the same number -> nothing left over.
    matrix = _matrix(5, lambda i, j: 0.25)
    assert core.disagreement(matrix, ~np.isnan(matrix)) == pytest.approx(0.0)


def test_disagreement_is_the_within_target_standard_deviation():
    n = 5
    # Row i shifts target j by a target-specific amount, so the columns are
    # NOT constant and the residual is not simply the row offset.
    matrix = _matrix(n, lambda i, j: (0.4 if j % 2 else -0.4) * (1 + 0.1 * i))
    observed = ~np.isnan(matrix)
    masked = np.where(observed, matrix, np.nan)
    leftover = masked - np.nanmean(masked, axis=0)[None, :]
    expected = float(np.sqrt(np.nanmean(leftover[observed] ** 2)))
    assert core.disagreement(matrix, observed) == pytest.approx(expected)


def test_disagreement_ignores_the_self_diagonal():
    # The diagonal holds NaN; a diagonal value must not leak into the metric.
    matrix = _matrix(4, lambda i, j: 0.5)
    clean = core.disagreement(matrix, ~np.isnan(matrix))
    polluted = matrix.copy()
    for i in range(4):
        polluted[i, i] = 1e6          # a self-score the observer never holds
    assert core.disagreement(polluted, ~np.isnan(matrix)) == pytest.approx(clean)


def test_disagreement_is_set_by_the_within_target_mean_not_a_pair_average():
    # A pure per-observer offset on a single shared target: every observer is
    # "self" for exactly one column, so masking the diagonal would leave each
    # column holding a *different* subset of the levels.  Here the one rated
    # target is not an observer at all, so the live set is the whole levels
    # vector and the expected value is plain.
    levels = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    matrix = _one_target_column(levels)
    observed = ~np.isnan(matrix)
    expected = float(np.sqrt(np.mean((levels - levels.mean()) ** 2)))
    assert core.disagreement(matrix, observed) == pytest.approx(expected)


def test_disagreement_squares_match_the_pairwise_identity():
    # mean_{i<j}(v_i - v_j)^2 == 2m/(m-1) * D^2 for the m observers of one target.
    rng = np.random.default_rng(7)
    levels = rng.uniform(-1, 1, 6)
    matrix = _one_target_column(levels)
    d = core.disagreement(matrix, ~np.isnan(matrix))
    m = len(levels)
    diffs = [
        (levels[i] - levels[j]) ** 2
        for i in range(m) for j in range(i + 1, m)
    ]
    assert float(np.mean(diffs)) == pytest.approx(2 * m / (m - 1) * d**2)


def test_disagreement_equals_the_mean_absolute_difference_when_the_split_is_balanced():
    # Balanced +/-1 split: |a-b| is 2 for every disagreeing pair, and both
    # summaries hit their maximum of 1.0 / 2.0.
    levels = np.array([1.0, 1.0, -1.0, -1.0])
    matrix = _one_target_column(levels)
    assert core.disagreement(matrix, ~np.isnan(matrix)) == pytest.approx(1.0)


def test_disagreement_never_exceeds_the_reputation_scale():
    # D is bounded by 1 by construction (values sit in [-1, 1], so no deviation
    # from a column mean can exceed 1).  The bound is attained by a balanced
    # half/half split; an unbalanced one stays strictly below it.
    rng = np.random.default_rng(11)
    for m in (4, 6, 20):
        levels = rng.uniform(-1.0, 1.0, m)
        assert core.disagreement(_one_target_column(levels), None) <= 1.0
    for m in (4, 6, 20):
        balanced = np.array([1.0] * (m // 2) + [-1.0] * (m - m // 2))
        assert core.disagreement(_one_target_column(balanced), None) == \
            pytest.approx(1.0)


def test_disagreement_is_nan_without_any_rating():
    assert np.isnan(core.disagreement(np.full((3, 3), np.nan), None))


def test_disagreement_respects_the_observed_mask():
    # A cell present in the matrix but never actually observed must not count.
    matrix = np.array([
        [np.nan, 0.0, 1.0],
        [0.0, np.nan, 0.0],
        [1.0, 0.0, np.nan],
    ])
    observed = np.array([
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ])
    # Only the two cells (0,1) and (1,2) survive, and each is its column's only
    # entry, so the residual is exactly zero.
    assert core.disagreement(matrix, observed) == pytest.approx(0.0)
    assert core.disagreement(matrix, ~np.isnan(matrix)) > 0.0


def test_disagreement_rejects_a_non_square_matrix():
    with pytest.raises(ValueError):
        core.disagreement(np.zeros((3, 4)), None)


def test_build_from_codes_cycles_the_evolved_pool():
    population = core.build_from_codes([BASELINES["ALLC"], BASELINES["ALLD"]], 5)
    assert [agent.agent_id for agent in population] == list(range(5))
    assert population[0].code == BASELINES["ALLC"]
    assert population[4].code == BASELINES["ALLC"]


def test_build_from_codes_rejects_degenerate_sizes():
    with pytest.raises(ValueError):
        core.build_from_codes([BASELINES["ALLC"]], 1)


def test_load_run_rejects_non_type1_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        core, "load_evolution_json",
        lambda path: {"config": {"agent_type": "agent-type2"}, "final_population": []},
    )
    with pytest.raises(ValueError, match="agent-type1"):
        core.load_run(tmp_path / "evolutionary.json")


def test_load_run_rejects_empty_final_population(monkeypatch, tmp_path):
    monkeypatch.setattr(
        core, "load_evolution_json",
        lambda path: {"config": {"agent_type": "agent-type1"}, "final_population": []},
    )
    with pytest.raises(ValueError, match="No final_population"):
        core.load_run(tmp_path / "evolutionary.json")


def test_load_run_reads_members_in_agent_id_order(monkeypatch, tmp_path):
    records = [
        {"agent_id": 2, "code": BASELINES["ALLD"], "fitness": 1.0, "lineage_id": 2, "origin": "imitate"},
        {"agent_id": 0, "code": BASELINES["ALLC"], "fitness": 3.0, "lineage_id": 0, "origin": "initial"},
    ]
    monkeypatch.setattr(
        core, "load_evolution_json",
        lambda path: {"config": {"agent_type": "agent-type1", "population_size": 2},
                      "final_population": records},
    )
    run = core.load_run(tmp_path / "evolutionary.json", "demo")
    assert run.label == "demo"
    assert [member.agent_id for member in run.members] == [0, 2]
    assert run.members[0].code == BASELINES["ALLC"]


# ---------------------------------------------------------------------------
# heatmap helpers (plot_private_reputation_matrix)
# ---------------------------------------------------------------------------


def test_effective_matrix_blanks_unseen_entries_and_the_diagonal():
    from experiments.analysis.consensus.plot_private_reputation_matrix import effective_matrix

    reputation = np.array([
        [0.0, 0.5, -1.0],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])
    # Agent 0 never actually rated agent 2; its stored value is just the prior.
    observed = np.array([
        [False, True, False],
        [True, False, True],
        [True, True, False],
    ])
    masked = effective_matrix(reputation, observed)
    assert np.isnan(masked[0, 2]) and np.isnan(masked[0, 0])
    assert masked[0, 1] == 0.5 and masked[2, 1] == 0.5


def test_effective_matrix_rejects_shape_mismatch():
    from experiments.analysis.consensus.plot_private_reputation_matrix import effective_matrix

    with pytest.raises(ValueError):
        effective_matrix(np.zeros((3, 3)), np.zeros((2, 2), dtype=bool))


def test_row_severity_recovers_the_constant_offset():
    from experiments.analysis.consensus.plot_private_reputation_matrix import row_severity

    row_levels = (-1.0, -0.5, 0.25)
    matrix = _matrix(3, lambda i, j: row_levels[i])
    severity = row_severity(matrix, ~np.isnan(matrix))
    assert severity == pytest.approx(np.array(row_levels))


def test_pairwise_disagreement_matrix_matches_mae_and_has_nan_diagonal():
    from experiments.analysis.consensus.plot_private_reputation_matrix import (
        pairwise_disagreement_matrix,
    )

    matrix = _matrix(4, lambda i, j: (-0.8, -0.3, 0.1, 0.6)[i])
    observed = ~np.isnan(matrix)
    pair = pairwise_disagreement_matrix(matrix, observed)
    assert np.all(np.isnan(np.diag(pair)))
    # Constant rows: the pairwise number collapses to the severity difference.
    assert pair[0, 3] == pytest.approx(1.4)
    assert pair[1, 2] == pytest.approx(0.4)


def test_pairwise_disagreement_matrix_respects_partial_coverage():
    from experiments.analysis.consensus.plot_private_reputation_matrix import (
        pairwise_disagreement_matrix,
    )

    matrix = np.array([
        [np.nan, 0.0, 1.0],        # agent 0 rated agents 1 and 2
        [0.0, np.nan, np.nan],     # agent 1 rated only agent 0
        [1.0, np.nan, np.nan],     # agent 2 rated only agent 0
    ])
    pair = pairwise_disagreement_matrix(matrix, ~np.isnan(matrix))
    # Agent 0 shares no rated target with 1 or 2, so those pairs are undefined
    # rather than zero.
    assert np.isnan(pair[0, 1]) and np.isnan(pair[0, 2])
    # Agents 1 and 2 share exactly one target (agent 0), differing there by 1.0.
    assert pair[1, 2] == pytest.approx(1.0)


def test_annotation_threshold_is_sane_for_small_populations():
    from experiments.analysis.consensus.plot_private_reputation_matrix import ANNOTATE_MAX_AGENTS

    assert ANNOTATE_MAX_AGENTS >= 8
