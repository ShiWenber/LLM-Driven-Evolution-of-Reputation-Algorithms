"""Guards for the norm-probe and survivor logic behind the code-landscape figure.

The figure marks a point as "matches a canonical norm" based on truth tables
recovered by *calling* the compiled strategy, so a regression here silently
mis-labels the figure rather than crashing it. These tests pin the probe to the
declared ``baselines.py`` constants and check that near-miss rules are not
force-matched.

Everything is hermetic: no GPU, no LLM, no dependence on local results data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.analysis.clustering.survivors import (
    NORM_ALIASES,
    _batched_clusters,
    action_table,
    assessment_table,
    canonical_norm_tables,
    discover_latest_archived_family,
    discover_latest_family,
    load_final_survivors,
    match_norm,
    match_survivors_to_cache,
    sample_cluster_codes,
    summarise_survivors,
    survivor_coverage,
    verify_canonical_tables,
)
from experiments.analysis.clustering.comments import strip_code_comments
from experiments.evolution_log import (
    F_CONFIG_POPULATION_SIZE,
    EvolutionLogError,
    build_evolution_results,
    lineage_event,
    population_entry,
    trajectory_entry,
    write_evolution_json,
)
from experiments.v2_quantitative.baselines import (
    ACTION_TABLES,
    ASSESSMENT_TABLES,
    BASELINES,
)
from experiments.v2_quantitative.executor import V2StrategyExecutor

# A rule that is *not* any canonical norm: its tables match none of the twelve.
NON_NORM_RULE = """
def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    good = A_action == 'cooperate' and B_rep > 0.5
    return A_rep + 1.0 if good else A_rep - 1.0

def decide(my_reputation, opponent_reputation):
    return opponent_reputation > 0.5 and my_reputation < 0
"""

# Numerically re-written Image Scoring: different constants, same behaviour.
IS_REWRITTEN = """
def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    return A_rep + 0.1 if A_action == 'cooperate' else A_rep - 0.2

def decide(my_reputation, opponent_reputation):
    return opponent_reputation > 0.375
"""


# --------------------------------------------------------------- norm probe


def test_probe_reproduces_declared_leading_eight_tables():
    """The probe must agree with baselines.py, or every label is suspect."""
    assert verify_canonical_tables() == []


@pytest.mark.parametrize("name", sorted(ASSESSMENT_TABLES))
def test_assessment_and_action_tables_match_constants(name: str):
    executor = V2StrategyExecutor(BASELINES[name])
    assert assessment_table(executor) == ASSESSMENT_TABLES[name]
    assert action_table(executor) == ACTION_TABLES[name]


def test_canonical_tables_exclude_synonyms():
    tables = canonical_norm_tables()
    assert len(tables) == 12
    for alias in NORM_ALIASES:
        assert alias not in tables
    assert set(tables) == {
        "ALLC", "ALLD", "IS", "SH",
        "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
    }


@pytest.mark.parametrize("name", sorted(canonical_norm_tables()))
def test_each_baseline_matches_itself(name: str):
    assert match_norm(BASELINES[name]) == name


@pytest.mark.parametrize("alias,canonical", sorted(NORM_ALIASES.items()))
def test_synonym_codes_report_canonical_name(alias: str, canonical: str):
    assert match_norm(BASELINES[alias]) == canonical


def test_numerically_rewritten_norm_is_recognised():
    """Behaviour, not source text, decides the label."""
    assert match_norm(IS_REWRITTEN) == "IS"


def test_non_norm_rule_is_not_force_matched():
    assert match_norm(NON_NORM_RULE) is None


def test_uncompilable_code_is_marked_unknown():
    assert match_norm("def observe(:") == "?"


def test_compiling_a_norm_does_not_saturate_the_sign_test():
    """A +/-0.5 probe score must survive the +/-1/3 update without clamping.

    If the probe used +/-1.0, L1-L8 would report identical tables because the
    update would clamp and the sign would be lost.
    """
    for name in sorted(ASSESSMENT_TABLES):
        executor = V2StrategyExecutor(BASELINES[name])
        for action in ("cooperate", "defect"):
            for actor_rep in (0.5, -0.5):
                for partner_rep in (1.0, -1.0):
                    new_rep = executor.observe(
                        actor_rep, action, partner_rep, "cooperate", 0.0
                    )
                    assert abs(new_rep) < 1.0, (
                        "probe score clamped; the sign test would be ambiguous"
                    )


# ------------------------------------------------------- cluster sampling


def test_sample_cluster_codes_is_deterministic():
    labels = [0] * 200 + [1] * 7
    codes = [f"code_{i}" for i in range(207)]
    assert sample_cluster_codes(labels, codes, 50, seed=42) == sample_cluster_codes(
        labels, codes, 50, seed=42
    )


def test_sample_cluster_codes_caps_and_takes_small_clusters_whole():
    labels = [0] * 200 + [1] * 7
    codes = [f"code_{i}" for i in range(207)]
    sampled = sample_cluster_codes(labels, codes, 50, seed=42)
    assert len(sampled[0]) == 50
    assert len(sampled[1]) == 7
    assert set(sampled[1]) == set(codes[200:])
    assert set(sampled[0]) <= set(codes[:200])


def test_sample_cluster_codes_samples_without_duplicates():
    labels = [0] * 40
    codes = [f"code_{i}" for i in range(40)]
    sampled = sample_cluster_codes(labels, codes, 10, seed=7)
    assert len(sampled[0]) == len(set(sampled[0])) == 10


# --------------------------------------------------- request batching


def test_batched_clusters_fits_one_request_when_small():
    samples = {0: ["a" * 10], 1: ["b" * 10]}
    batches = _batched_clusters(samples, budget_chars=10_000, max_code_chars=600)
    assert len(batches) == 1
    assert batches[0] == samples


def test_batched_clusters_splits_when_over_budget():
    """50 codes per cluster is large; the packing must respect the budget."""
    samples = {c: ["x" * 500] * 50 for c in range(6)}
    batches = _batched_clusters(samples, budget_chars=30_000, max_code_chars=600)
    assert len(batches) > 1
    for batch in batches:
        cost = sum(sum(len(code) for code in members) for members in batch.values())
        assert cost <= 30_000


def test_batched_clusters_loses_no_cluster_and_never_duplicates():
    samples = {c: ["x" * 500] * 50 for c in range(7)}
    batches = _batched_clusters(samples, budget_chars=20_000, max_code_chars=600)
    seen = [cid for batch in batches for cid in batch]
    assert sorted(seen) == sorted(samples)
    assert len(seen) == len(set(seen))


def test_batched_clusters_caps_cost_by_max_code_chars():
    """Truncation length, not raw length, must drive the packing decision.

    The codes themselves are stored whole (truncation happens when the prompt is
    rendered, via ``code[:max_code_chars]``); only the *cost estimate* uses the
    truncated length. Two huge codes therefore pack together when the render
    limit is small, and must split when it is large.
    """
    two_huge = {0: ["y" * 100_000], 1: ["y" * 100_000]}
    packed = _batched_clusters(two_huge, budget_chars=2_000, max_code_chars=100)
    split = _batched_clusters(two_huge, budget_chars=2_000, max_code_chars=100_000)
    assert len(packed) == 1, "100-char render limit should fit both clusters"
    assert len(split) == 2, "100k-char render limit should not fit both"


def test_batched_clusters_handles_a_single_oversized_cluster():
    """One cluster bigger than the budget still has to be named eventually."""
    samples = {0: ["z" * 900] * 50}
    batches = _batched_clusters(samples, budget_chars=1_000, max_code_chars=900)
    assert len(batches) == 1
    assert batches[0] == samples


# ------------------------------------------------------------ K selection


def _blobs(centers, per_center=40, seed=0):
    """A small synthetic embedding set with an obvious cluster structure."""
    rng = np.random.default_rng(seed)
    return np.vstack([
        rng.normal(loc=center, scale=0.05, size=(per_center, len(center)))
        for center in centers
    ])


def test_choose_k_respects_the_minimum_bound():
    """Silhouette on a continuum-like space picks tiny K; --min-k must override."""
    from experiments.analysis.plot_code_landscape import choose_k

    # two tight, far-apart blobs: silhouette's outright winner is K=2
    X = _blobs([(0.0, 0.0, 0.0), (10.0, 10.0, 10.0)], per_center=40)
    assert choose_k(X, seed=0, maximum=6) == 2
    assert choose_k(X, seed=0, minimum=4, maximum=6) >= 4


def test_choose_k_respects_the_maximum_bound():
    from experiments.analysis.plot_code_landscape import choose_k

    X = _blobs([(0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (0.0, 10.0, 0.0)])
    assert choose_k(X, seed=0, maximum=3) <= 3


def test_choose_k_returns_one_for_degenerate_inputs():
    from experiments.analysis.plot_code_landscape import choose_k

    assert choose_k(np.zeros((2, 4)), seed=0) == 1
    # an empty search range must degrade gracefully rather than raise
    assert choose_k(_blobs([(0.0, 0.0), (5.0, 5.0)]), seed=0, minimum=9, maximum=3) == 1


def test_choose_k_is_deterministic():
    from experiments.analysis.plot_code_landscape import choose_k

    X = _blobs([(0.0, 0.0, 0.0), (6.0, 6.0, 6.0), (0.0, 6.0, 6.0)], per_center=30)
    assert choose_k(X, seed=7, maximum=5) == choose_k(X, seed=7, maximum=5)


# ------------------------------------------------------ clustering method


def test_hierarchical_is_refused_above_the_scale_limit():
    """Ward needs an O(n^2) distance matrix; the guard must fire before OOM."""
    from experiments.analysis.plot_code_landscape import (
        MAX_HIERARCHICAL_CODES,
        check_clustering_method,
    )

    check_clustering_method("hierarchical", MAX_HIERARCHICAL_CODES)  # at the limit: fine
    check_clustering_method("kmeans", 10 ** 6)  # kmeans is unbounded
    with pytest.raises(SystemExit, match="hierarchical"):
        check_clustering_method("hierarchical", MAX_HIERARCHICAL_CODES + 1)


# --------------------------------------------------- zoom-inset geometry

# Real measured PCA coordinates of the canonical baselines (2026-09-14 run), used
# to keep these tests tied to the geometry that motivated the inset.
LEADING_EIGHT_NAMES = ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8")
REAL_BASELINES = {
    "ALLC": (-0.001, 0.068), "ALLD": (0.028, 0.141),
    "IS": (0.093, 0.215), "SH": (0.105, 0.206),
    "L1": (0.316, 0.148), "L2": (0.310, 0.141), "L3": (0.315, 0.149),
    "L4": (0.316, 0.145), "L5": (0.311, 0.140), "L6": (0.305, 0.139),
    "L7": (0.314, 0.140), "L8": (0.299, 0.137),
}
CLOUD_SPAN = 0.515


def test_single_linkage_groups_chains_transitively():
    """Single linkage: a chain of close points is one group even if the ends are far."""
    from experiments.analysis.plot_code_landscape import _single_linkage_groups

    points = [(0.0, 0.0), (0.6, 0.0), (1.2, 0.0), (10.0, 0.0)]
    groups = _single_linkage_groups(points, threshold=0.7)
    assert sorted(len(g) for g in groups) == [1, 3]
    chained = max(groups, key=len)
    assert chained == [0, 1, 2]


def test_single_linkage_groups_separates_distant_points():
    from experiments.analysis.plot_code_landscape import _single_linkage_groups

    points = [(0.0, 0.0), (5.0, 5.0)]
    assert sorted(len(g) for g in _single_linkage_groups(points, threshold=0.1)) == [1, 1]


def test_leading_eight_form_one_group_but_allc_and_alld_do_not():
    """The grouping rule must isolate L1-L8, which is what the inset magnifies."""
    from experiments.analysis.plot_code_landscape import (
        ZOOM_GROUP_THRESHOLD,
        _single_linkage_groups,
    )

    names = list(REAL_BASELINES)
    positions = np.array([REAL_BASELINES[n] for n in names])
    groups = _single_linkage_groups(positions, ZOOM_GROUP_THRESHOLD * CLOUD_SPAN)
    biggest = max(groups, key=len)
    assert {names[i] for i in biggest} == set(LEADING_EIGHT_NAMES)


def test_leading_eight_span_is_a_tiny_fraction_of_the_cloud():
    """Quantifies why the inset is needed at all."""
    positions = np.array([REAL_BASELINES[n] for n in LEADING_EIGHT_NAMES])
    span = float(max(np.ptp(positions[:, 0]), np.ptp(positions[:, 1])))
    assert span / CLOUD_SPAN < 0.05, "if this grows, the inset may be unnecessary"
    # ... and they are far closer than the marker glyph is wide
    assert span < 0.02


def test_zoom_window_contains_every_point_and_pads_beyond_it():
    from experiments.analysis.plot_code_landscape import _zoom_window

    points = np.array([REAL_BASELINES[n] for n in LEADING_EIGHT_NAMES])
    wx0, wx1, wy0, wy1 = _zoom_window(points, CLOUD_SPAN, margin_fraction=0.6)
    assert wx0 < points[:, 0].min() and wx1 > points[:, 0].max()
    assert wy0 < points[:, 1].min() and wy1 > points[:, 1].max()
    # padding is real work, not decoration: the window is wider than the points
    assert (wx1 - wx0) > (points[:, 0].max() - points[:, 0].min())


def test_zoom_window_keeps_a_minimum_size_for_coincident_points():
    """All-identical points must not produce a zero-width window."""
    from experiments.analysis.plot_code_landscape import _zoom_window

    points = np.zeros((5, 2))
    wx0, wx1, wy0, wy1 = _zoom_window(points, CLOUD_SPAN)
    assert (wx1 - wx0) > 0
    assert (wy1 - wy0) > 0


def test_zoom_window_differentiates_labels_that_markers_cannot():
    """The magnification must separate the leading eight in *display* space."""
    from experiments.analysis.plot_code_landscape import _zoom_window

    points = np.array([REAL_BASELINES[n] for n in LEADING_EIGHT_NAMES])
    wx0, wx1, _, _ = _zoom_window(points, CLOUD_SPAN, margin_fraction=0.6)

    axes_width_px = 1500  # representative plot width in pixels
    glyph_px = 15  # diameter of a s=190 diamond marker at 200 dpi
    all_markers_span_px = np.ptp(points[:, 0]) / CLOUD_SPAN * axes_width_px
    full_scale_gap_px = all_markers_span_px / len(LEADING_EIGHT_NAMES)
    # At full scale adjacent diamonds are ~6 px apart while each glyph is ~15 px
    # wide, so the eight of them merge into one blob -- the reason for the inset.
    assert full_scale_gap_px < glyph_px

    # Inside the inset the window is shown at `magnification`, mapped onto a box
    # whose long side is `target_long_side` of the axes (see _inset_box).
    magnification = CLOUD_SPAN / (wx1 - wx0)
    inset_long_side_fraction = 0.34
    inset_gap_px = full_scale_gap_px * magnification * inset_long_side_fraction
    assert inset_gap_px > glyph_px, (
        f"magnified gap {inset_gap_px:.1f}px must clear the {glyph_px}px glyph"
    )


def test_inset_box_preserves_the_window_aspect():
    """An inset that stretches the window would misstate the geometry."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import _inset_box, _zoom_window

    points = np.array([REAL_BASELINES[n] for n in LEADING_EIGHT_NAMES])
    window = _zoom_window(points, CLOUD_SPAN)
    fig, ax = plt.subplots(figsize=(13.5, 8.6))
    try:
        ax.set_xlim(-0.15, 0.37)
        ax.set_ylim(-0.20, 0.25)
        fig.tight_layout()
        fig.canvas.draw()
        w_frac, h_frac = _inset_box(ax, window)
        assert 0 < w_frac <= 0.46 and 0 < h_frac <= 0.46

        extent = ax.get_window_extent()
        box_px = (w_frac * extent.width, h_frac * extent.height)
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        want_px = (
            (window[1] - window[0]) / (x_hi - x_lo) * extent.width,
            (window[3] - window[2]) / (y_hi - y_lo) * extent.height,
        )
        assert (box_px[0] / box_px[1]) == pytest.approx(want_px[0] / want_px[1], rel=1e-6)
    finally:
        plt.close(fig)


def test_place_inset_never_covers_a_caption():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import _place_inset

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        rng = np.random.default_rng(0)
        Z = rng.normal(size=(500, 2))
        ax.set_xlim(Z[:, 0].min(), Z[:, 0].max())
        ax.set_ylim(Z[:, 1].min(), Z[:, 1].max())
        fig.tight_layout()
        fig.canvas.draw()
        extent = ax.get_window_extent()
        box = (0.3, 0.3)
        # an obstacle planted at each corner: every candidate must be rejected,
        # so the chooser has to fall back rather than crash
        obstacles = [
            (0.02 * extent.width, 0.02 * extent.height),
            (0.98 * extent.width, 0.98 * extent.height),
        ]
        placed = _place_inset(ax, box, Z, obstacles)
        assert len(placed) == 4
        assert 0 <= placed[0] <= 1 and 0 <= placed[1] <= 1
        # the chosen corner must not contain either obstacle
        for ox, oy in obstacles:
            inside = (
                placed[0] * extent.width <= ox <= (placed[0] + placed[2]) * extent.width
                and placed[1] * extent.height <= oy
                <= (placed[1] + placed[3]) * extent.height
            )
            assert not inside
    finally:
        plt.close(fig)


def test_draw_zoom_inset_magnifies_the_leading_eight():
    """End-to-end on the real geometry: an inset is drawn and reported."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import _draw_zoom_inset

    labelled = [(name, REAL_BASELINES[name]) for name in LEADING_EIGHT_NAMES]
    labelled.append(("ALLC", REAL_BASELINES["ALLC"]))
    labelled.append(("ALLD", REAL_BASELINES["ALLD"]))
    Z = np.array([REAL_BASELINES[n] for n in (*LEADING_EIGHT_NAMES, "ALLC", "ALLD")])

    fig, ax = plt.subplots(figsize=(13.5, 8.6))
    try:
        ax.scatter(Z[:, 0], Z[:, 1])
        ax.set_xlim(-0.15, 0.37)
        ax.set_ylim(-0.20, 0.25)
        fig.tight_layout()
        fig.canvas.draw()

        names, setup = _draw_zoom_inset(ax, fig, labelled, Z, [])
        assert set(names) == set(LEADING_EIGHT_NAMES)
        assert setup is not None
        assert setup["magnification"] > 5
        assert setup["window"][0] < REAL_BASELINES["L8"][0]
        assert setup["window"][3] > REAL_BASELINES["L1"][1]

        # a zoom rectangle was added, and exactly one inset axes created
        assert any(isinstance(p, plt.Rectangle) for p in ax.patches)
        assert len(ax.child_axes) == 1
    finally:
        plt.close(fig)


def test_draw_zoom_inset_declines_when_nothing_is_crowded():
    """Well-separated baselines need no inset; the de-overlap pass suffices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import _draw_zoom_inset

    labelled = [("A", (0.0, 0.0)), ("B", (10.0, 0.0)), ("C", (0.0, 10.0))]
    Z = np.array([position for _, position in labelled])
    fig, ax = plt.subplots()
    try:
        ax.scatter(Z[:, 0], Z[:, 1])
        fig.canvas.draw()
        names, setup = _draw_zoom_inset(ax, fig, labelled, Z, [])
        assert names == [] and setup is None
        assert len(ax.child_axes) == 0
    finally:
        plt.close(fig)


def test_draw_zoom_inset_needs_at_least_three_baselines():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import _draw_zoom_inset

    fig, ax = plt.subplots()
    try:
        names, setup = _draw_zoom_inset(
            ax, fig, [("A", (0.0, 0.0)), ("B", (0.001, 0.0))], np.zeros((2, 2)), []
        )
        assert names == [] and setup is None
    finally:
        plt.close(fig)


# ------------------------------------------------- baseline label coverage


def test_every_drawn_baseline_is_labelled_by_default():
    """A drawn-but-unnamed diamond is indistinguishable from a mislabelled one.

    Every baseline the script embeds is plotted with the same glyph, so the
    default label set must cover the whole drawn set. Only the zoom inset may
    take names off the main axes, and it redraws them inside itself.
    """
    from experiments.analysis.plot_code_landscape import (
        CANONICAL_BASELINE_NAMES,
        DEFAULT_LABELLED_BASELINES,
    )

    drawn = {n for n in CANONICAL_BASELINE_NAMES if n in BASELINES}
    missing = drawn - set(DEFAULT_LABELLED_BASELINES)
    assert not missing, f"plotted without a name: {sorted(missing)}"


def test_labelled_baselines_only_names_real_baselines():
    from experiments.analysis.plot_code_landscape import DEFAULT_LABELLED_BASELINES

    unknown = set(DEFAULT_LABELLED_BASELINES) - set(BASELINES)
    assert not unknown, f"labels that match no baseline: {sorted(unknown)}"


def test_drawn_baselines_match_the_canonical_names():
    """Skipping the synonyms (SC/SS/SJ) must not drop a genuinely distinct rule."""
    from experiments.analysis.plot_code_landscape import CANONICAL_BASELINE_NAMES

    assert CANONICAL_BASELINE_NAMES == (
        "ALLC", "ALLD", "IS", "SH",
        "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
    )
    assert all(name in BASELINES for name in CANONICAL_BASELINE_NAMES)
    # the three synonym codes are the only BASELINES entries left out
    assert set(BASELINES) - set(CANONICAL_BASELINE_NAMES) == {"SC", "SS", "SJ"}


def test_label_overlaps_detects_intersection_and_ignores_touching():
    """Exact box geometry beats re-detecting colours in the rendered PNG."""
    from experiments.analysis.plot_code_landscape import label_overlaps

    boxes = [
        {"name": "A", "x0": 0, "y0": 0, "x1": 10, "y1": 10},
        {"name": "B", "x0": 2, "y0": 2, "x1": 8, "y1": 8},      # inside A
        {"name": "C", "x0": 20, "y0": 20, "x1": 30, "y1": 30},  # clear of everything
        {"name": "D", "x0": 10, "y0": 0, "x1": 20, "y1": 10},   # touches A's right edge
    ]
    collisions = label_overlaps(boxes)
    assert ("A", "B") in collisions
    assert ("A", "D") not in collisions, "edge contact is not an overlap"
    assert sorted(collisions) == [("A", "B")]


def test_label_rectangles_records_every_annotation():
    """The figure must report its own label geometry for verification."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import (
        _label_rectangles,
        label_overlaps,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        annotations = []
        for i, name in enumerate(("ALLC", "ALLD", "L1")):
            annotations.append((name, ax.annotate(
                name, (i / 10, i / 10), xytext=(20, -20),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white"},
            )))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.canvas.draw()
        boxes = _label_rectangles(annotations, fig)
        assert [b["name"] for b in boxes] == ["ALLC", "ALLD", "L1"]
        for box in boxes:
            assert box["x1"] > box["x0"] and box["y1"] > box["y0"]
        assert label_overlaps(boxes) == []
    finally:
        plt.close(fig)


def test_de_overlap_actually_produces_disjoint_boxes():
    """The greedy placer's whole purpose: no two labels may intersect.

    Uses a deliberately crowded anchor set (the real IS/SH/ALLC/ALLD coordinates
    pushed close together) so the placer has to search.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import (
        _label_rectangles,
        _resolve_label_offsets,
        label_overlaps,
    )

    names = ["ALLC", "ALLD", "IS", "SH"]
    anchors = [(0.00, 0.00), (0.02, 0.01), (0.01, 0.03), (0.03, 0.02)]
    fig, ax = plt.subplots(figsize=(10, 7))
    try:
        ax.set_xlim(-0.05, 0.10)
        ax.set_ylim(-0.05, 0.10)
        fig.tight_layout()
        fig.canvas.draw()
        offsets = _resolve_label_offsets(ax, fig, anchors, names, fontsize=9.5)
        annotations = []
        for name, (x, y), (dx, dy) in zip(names, anchors, offsets):
            annotations.append((name, ax.annotate(
                name, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=9.5, fontweight="bold",
                bbox={"boxstyle": "round,pad=0.26", "facecolor": "#FFF7CC"},
            )))
        fig.canvas.draw()
        assert label_overlaps(_label_rectangles(annotations, fig)) == []
    finally:
        plt.close(fig)


def test_is_and_sh_need_no_inset_but_l1_l8_do():
    """The inset trigger is group *size*, not distance alone.

    Measured IS/SH separation is ~2.9% of the plotted span, which is *inside* the
    5% grouping threshold -- so distance alone would group them. What keeps them
    on the main axes is the ``len(group) >= 3`` rule: a pair of well-separated
    comparators gets ordinary label offsets, while the eight near-coincident
    norms cannot be separated by any offset and need magnification.
    """
    from experiments.analysis.plot_code_landscape import (
        ZOOM_GROUP_THRESHOLD,
        _single_linkage_groups,
    )

    is_pos = np.array(REAL_BASELINES["IS"])
    sh_pos = np.array(REAL_BASELINES["SH"])

    # IS and SH are grouped together but the group is only two members...
    assert float(np.linalg.norm(is_pos - sh_pos)) / CLOUD_SPAN < ZOOM_GROUP_THRESHOLD
    is_sh_group = _single_linkage_groups(
        np.vstack([is_pos, sh_pos]), ZOOM_GROUP_THRESHOLD * CLOUD_SPAN
    )
    assert len(is_sh_group) == 1 and len(is_sh_group[0]) == 2

    # ...whereas the leading eight form a group that clears the size threshold.
    l8_positions = np.array([REAL_BASELINES[n] for n in LEADING_EIGHT_NAMES])
    groups = _single_linkage_groups(l8_positions, ZOOM_GROUP_THRESHOLD * CLOUD_SPAN)
    assert len(groups) == 1 and len(groups[0]) >= 3

    # and the two groups never chain into each other
    everything = _single_linkage_groups(
        np.vstack([np.array([is_pos, sh_pos]), l8_positions]),
        ZOOM_GROUP_THRESHOLD * CLOUD_SPAN,
    )
    assert len(everything) == 2
    assert sorted(len(g) for g in everything) == [2, 8]


# ------------------------------------------- inset reproduces main styles


def _face_rgb(collection):
    """First face colour of a scatter collection, or ``None`` when unfilled.

    Outline-only collections (``facecolors="none"``) report an empty array, so
    indexing them blindly raises.
    """
    facecolors = collection.get_facecolor()
    if len(facecolors) == 0:
        return None
    return tuple(facecolors[0][:3])


def test_inset_replays_cluster_colours_and_sizes_unchanged():
    """The magnified panel must not restyle the points it magnifies.

    A grey or rescaled inset shows a different-looking population than the
    overview it is a detail of, so the two panels stop being readable as one
    figure. This pins the inset's cloud to the exact colour/size/alpha the main
    axes use for the same clusters.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import (
        CLOUD_POINT_ALPHA,
        CLOUD_POINT_SIZE,
        _cluster_colors,
        _draw_zoom_inset,
    )

    labelled = [(name, REAL_BASELINES[name]) for name in LEADING_EIGHT_NAMES]
    labelled.append(("ALLC", REAL_BASELINES["ALLC"]))
    labelled.append(("ALLD", REAL_BASELINES["ALLD"]))

    # a cloud whose points deliberately fall inside the leading-eight window
    rng = np.random.default_rng(0)
    window_centre = np.array([
        (REAL_BASELINES["L1"][0] + REAL_BASELINES["L8"][0]) / 2,
        (REAL_BASELINES["L1"][1] + REAL_BASELINES["L8"][1]) / 2,
    ])
    inside_points = window_centre + rng.normal(scale=0.002, size=(60, 2))
    outside_points = rng.normal(scale=0.05, size=(40, 2)) + np.array([-0.05, -0.05])
    Z_cloud = np.vstack([inside_points, outside_points])

    Z = np.vstack([
        Z_cloud,
        np.array([REAL_BASELINES[n]
                  for n in (*LEADING_EIGHT_NAMES, "ALLC", "ALLD")]),
    ])
    # ``labels`` must cover every embedded row, baselines included (production
    # passes km.labels_, which spans the whole matrix).
    labels = np.array([0] * 60 + [1] * (len(Z) - 60))
    colors = _cluster_colors(2)

    fig, ax = plt.subplots(figsize=(13.5, 8.6))
    try:
        ax.set_xlim(-0.15, 0.37)
        ax.set_ylim(-0.20, 0.25)
        fig.tight_layout()
        fig.canvas.draw()

        names, setup = _draw_zoom_inset(
            ax, fig, labelled, Z, [], labels=labels, colors=colors,
        )
        assert names, "the leading eight should have produced an inset"
        inset = ax.child_axes[0]

        cloud_facecolors = [
            colour for colour in (_face_rgb(c) for c in inset.collections)
            if colour is not None
        ]
        sizes = [float(collection.get_sizes()[0])
                 for collection in inset.collections]

        expected_colors = {
            tuple(matplotlib.colors.to_rgb(c)) for c in colors
        }
        assert expected_colors.issubset(set(cloud_facecolors)), (
            "the inset must use the per-cluster colours, not one flat colour"
        )
        assert not any(
            colour == tuple(matplotlib.colors.to_rgb("#98A2B3"))
            for colour in cloud_facecolors
        ), "the old hard-coded grey must not come back"

        # Marker sizes split into two legitimate groups: the cloud points stay at
        # the overview's size, and the baseline diamonds keep theirs. Anything
        # else means the inset rescaled something.
        from experiments.analysis.plot_code_landscape import (
            BASELINE_MARKER_SIZE,
        )

        assert set(sizes) <= {CLOUD_POINT_SIZE, BASELINE_MARKER_SIZE}, sizes
        diamond_sizes = [
            float(collection.get_sizes()[0])
            for collection in inset.collections
            if collection.get_paths()[0].vertices.shape[0] == 4  # a diamond
        ]
        assert set(diamond_sizes) <= {BASELINE_MARKER_SIZE}
        assert CLOUD_POINT_SIZE in sizes, "the magnified cloud must still be drawn"
    finally:
        plt.close(fig)


def test_inset_replays_survivor_marks_instead_of_dropping_them():
    """A survivor inside the magnified region must not silently disappear."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import (
        SURVIVOR_COLOR,
        _draw_zoom_inset,
    )

    labelled = [(name, REAL_BASELINES[name]) for name in LEADING_EIGHT_NAMES]
    labelled += [("ALLC", REAL_BASELINES["ALLC"]), ("ALLD", REAL_BASELINES["ALLD"])]
    Z = np.array([REAL_BASELINES[n] for n in
                  (*LEADING_EIGHT_NAMES, "ALLC", "ALLD")])

    survivor_xy = np.array([
        [REAL_BASELINES["L3"][0] + 0.001, REAL_BASELINES["L3"][1] - 0.001],
        [-0.10, -0.10],  # far outside the window
    ])
    cooperation = np.array([0.99, 0.10])

    fig, ax = plt.subplots(figsize=(13.5, 8.6))
    try:
        ax.set_xlim(-0.15, 0.37)
        ax.set_ylim(-0.20, 0.25)
        fig.tight_layout()
        fig.canvas.draw()
        names, setup = _draw_zoom_inset(
            ax, fig, labelled, Z, [],
            survivor_points=survivor_xy, survivor_cooperation=cooperation,
            min_cooperation=0.90,
        )
        assert names
        inset = ax.child_axes[0]
        wanted = tuple(matplotlib.colors.to_rgb(SURVIVOR_COLOR))
        marks = [c for c in inset.collections if _face_rgb(c) == wanted]
        assert marks, "the survivor layer must be replayed inside the inset"

        # exactly one survivor is inside the window, so only one mark is drawn
        n_points = sum(len(collection.get_offsets()) for collection in marks)
        assert n_points == 1, f"expected 1 in-window survivor, drew {n_points}"
    finally:
        plt.close(fig)


def test_survivor_helpers_respect_the_window():
    """Out-of-window survivors are excluded, in-window ones kept."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import _plot_survivor_layers

    points = np.array([[0.0, 0.0], [5.0, 5.0]])
    coop = np.array([0.99, 0.95])
    fig, ax = plt.subplots()
    try:
        _plot_survivor_layers(
            ax, points, coop, min_cooperation=0.90,
            window=(-0.5, 0.5, -0.5, 0.5),
        )
        drawn = sum(len(c.get_offsets()) for c in ax.collections)
        assert drawn == 2, "one fill + one high-cooperation outline for one point"
    finally:
        plt.close(fig)


def test_survivor_helpers_handle_nothing_to_draw():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import _plot_survivor_layers

    fig, ax = plt.subplots()
    try:
        _plot_survivor_layers(ax, np.empty((0, 2)), np.empty(0),
                             min_cooperation=0.9)
        _plot_survivor_layers(ax, np.array([[9.0, 9.0]]), np.array([1.0]),
                             min_cooperation=0.9, window=(0, 1, 0, 1))
        assert len(ax.collections) == 0
    finally:
        plt.close(fig)


def test_plot_cloud_window_restricts_points_to_the_region():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import (
        _cluster_colors,
        _plot_cloud,
    )

    Z = np.array([[0.0, 0.0], [0.1, 0.1], [9.0, 9.0]])
    labels = np.array([0, 0, 0])
    fig, ax = plt.subplots()
    try:
        _plot_cloud(ax, Z, labels, _cluster_colors(1), window=(-1, 1, -1, 1))
        total = sum(len(c.get_offsets()) for c in ax.collections)
        assert total == 2, "only the in-window points may be drawn"
    finally:
        plt.close(fig)


# ------------------------------------------ label boxes, leaders, placement


def test_label_alignment_follows_the_offset_direction():
    from experiments.analysis.plot_code_landscape import _label_alignment

    assert _label_alignment(10, 5) == ("left", "bottom")
    assert _label_alignment(-10, -5) == ("right", "top")
    assert _label_alignment(0, 10) == ("center", "bottom")
    assert _label_alignment(0, -10) == ("center", "top")
    # a purely horizontal offset is vertically centred, not pinned to an edge
    assert _label_alignment(10, 0) == ("left", "center")
    assert _label_alignment(-10, 0) == ("right", "center")


@pytest.mark.parametrize("offset", [(30, 0), (-30, 0), (0, 30), (0, -30),
                                    (24, 24), (-24, -24)])
def test_planned_label_box_sits_on_the_same_side_as_the_rendered_box(offset):
    """The planner's box must be the box matplotlib actually draws.

    The previous planner assumed text always extended right and was vertically
    centred. Labels placed left or above therefore collided in the output while
    the planner believed they were clear -- which is why crowded clusters ended up
    with names flung far from their markers. This compares the planned rectangle
    against the real rendered extent, per side.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import _label_box_px

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.canvas.draw()

        anchor = (0.5, 0.5)
        anchor_px = ax.transData.transform(anchor)
        fontsize = 9.5
        planned = _label_box_px(anchor_px, offset, "L1",
                                fontsize=fontsize, px_per_pt=fig.dpi / 72.0)
        annotation = ax.annotate(
            "L1", anchor, xytext=offset, textcoords="offset points",
            fontsize=fontsize, bbox={"boxstyle": "round,pad=0.24"},
        )
        fig.canvas.draw()
        rendered = annotation.get_window_extent(renderer=fig.canvas.get_renderer())
        rendered_box = (rendered.x0, rendered.x1, rendered.y0, rendered.y1)

        eps = 1.0
        for label, box in (("planned", planned), ("rendered", rendered_box)):
            if offset[0] > 0:
                assert box[0] >= anchor_px[0] - eps, f"{label} should start right of the anchor"
            elif offset[0] < 0:
                assert box[1] <= anchor_px[0] + eps, f"{label} should end left of the anchor"
            if offset[1] > 0:
                assert box[2] >= anchor_px[1] - eps, f"{label} should sit above the anchor"
            elif offset[1] < 0:
                assert box[3] <= anchor_px[1] + eps, f"{label} should sit below the anchor"
    finally:
        plt.close(fig)


def test_point_box_gap_is_zero_when_the_point_is_inside():
    from experiments.analysis.plot_code_landscape import _point_box_gap

    box = (10.0, 20.0, 30.0, 40.0)
    assert _point_box_gap((15.0, 35.0), box) == 0.0
    assert _point_box_gap((10.0, 40.0), box) == 0.0  # on the corner
    assert _point_box_gap((4.0, 35.0), box) == pytest.approx(6.0)
    # diagonal: 3 left, 4 below
    assert _point_box_gap((7.0, 26.0), box) == pytest.approx(5.0)


def test_leaders_are_drawn_only_when_a_label_drifts_away():
    """Close labels stay clean; drifting labels keep a connector to their marker.

    The inset overrides this with ``always_leader=True``, because there every label
    is displaced by a crowded neighbour and the connector is the only way to tell
    which diamond a name belongs to.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import _draw_baseline_labels

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.canvas.draw()

        anchor = (0.5, 0.5)

        def draw(offset, **kwargs):
            return _draw_baseline_labels(
                ax, fig, [anchor], ["L1"], [offset], fontsize=9.5,
                facecolor="white", edgecolor="black", **kwargs,
            )[0][1]

        # tucked against the marker: no connector
        assert draw((0, 8)).arrow_patch is None, "a snug label needs no leader"
        # pushed well clear: connector
        assert draw((0, 84)).arrow_patch is not None, "a distant label needs one"
        # the inset's policy: always connected
        assert draw((0, 8), always_leader=True).arrow_patch is not None
        assert draw((0, 84), always_leader=True).arrow_patch is not None
    finally:
        plt.close(fig)


def test_labels_prefer_staying_inside_the_panel():
    """A crowded label must not be pushed outside the axes to find free space."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import (
        _label_box_px,
        _resolve_label_offsets,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        ax.set_xlim(-0.5, 0.8)
        ax.set_ylim(-0.5, 0.8)
        fig.tight_layout()
        fig.canvas.draw()
        bounds = tuple(ax.get_window_extent().extents)

        names = [f"L{i}" for i in range(1, 9)]
        anchors = [(0.15 + 0.002 * i, 0.15 + 0.001 * i) for i in range(8)]
        offsets = _resolve_label_offsets(ax, fig, anchors, names,
                                         fontsize=9.0, bounds=bounds)
        for name, anchor, offset in zip(names, anchors, offsets):
            box = _label_box_px(ax.transData.transform(anchor), offset, name,
                                fontsize=9.0, px_per_pt=fig.dpi / 72.0)
            # allow a little slack: the scoring penalises, it does not forbid
            assert box[0] > bounds[0] - 30, f"{name} escaped left"
            assert box[1] < bounds[2] + 30, f"{name} escaped right"
            assert box[2] > bounds[1] - 30, f"{name} escaped below"
            assert box[3] < bounds[3] + 30, f"{name} escaped above"
    finally:
        plt.close(fig)


def test_placement_is_deterministic():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import _resolve_label_offsets

    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.canvas.draw()
        names = ["A", "B", "C", "D"]
        anchors = [(0.5, 0.5), (0.503, 0.502), (0.506, 0.504), (0.501, 0.506)]
        first = _resolve_label_offsets(ax, fig, anchors, names, fontsize=9.0)
        second = _resolve_label_offsets(ax, fig, anchors, names, fontsize=9.0)
        assert first == second
    finally:
        plt.close(fig)


def test_placement_keeps_crowded_labels_apart():
    """Eight near-coincident anchors must not all land on the same offset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.analysis.plot_code_landscape import (
        _label_box_px,
        _resolve_label_offsets,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.canvas.draw()
        names = [f"L{i}" for i in range(1, 9)]
        anchors = [(0.5 + 0.001 * i, 0.5 + 0.0005 * i) for i in range(8)]
        offsets = _resolve_label_offsets(ax, fig, anchors, names, fontsize=9.5)
        assert len(set(offsets)) >= 5, "offsets should spread, not stack"
        boxes = [
            _label_box_px(ax.transData.transform(anchor), offset, name,
                          fontsize=9.5, px_per_pt=fig.dpi / 72.0)
            for name, anchor, offset in zip(names, anchors, offsets)
        ]
        heavy = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[1] and a[1] > b[0] and a[2] < b[3] and a[3] > b[2]:
                    heavy += 1
        assert heavy <= 1, f"{heavy} label pairs still collide ({offsets})"
    finally:
        plt.close(fig)


# ------------------------------------------------------------- survivor I/O


def _valid_record_data(label: str, seed: int, codes, coop) -> dict:
    """A minimal schema-v4 record, built with the project's own builders."""
    population = [
        population_entry(
            agent_id=i, code=code, fitness=1.0, cooperation_rate=rate,
            self_reputation=0.0, lineage_id=i, parent_id=None,
            parent_lineage_id=None, origin="initial", birth_gen=0,
        )
        for i, (code, rate) in enumerate(zip(codes, coop))
    ]
    return build_evolution_results(
        trajectory=[trajectory_entry(0, sum(coop) / len(coop), 10, 1.0, 1.0, population)],
        final_population=population,
        lineage_events=[lineage_event(i, origin="initial", birth_gen=0)
                        for i in range(len(codes))],
        config={
            "agent_type": "agent-type1",
            "seed": seed,
            "label": label,
            F_CONFIG_POPULATION_SIZE: len(codes),
        },
    )


def _write_record(directory: Path, label: str, seed: int, codes, coop) -> Path:
    """Write a minimal schema-v4 record to ``<directory>/<label>/evolutionary.json``."""
    path = directory / label / "evolutionary.json"
    write_evolution_json(path, _valid_record_data(label, seed, codes, coop))
    return path


def test_load_final_survivors_reads_every_agent(tmp_path):
    path = _write_record(tmp_path, "exp_seed0", 0, ["a", "b"], [0.99, 0.10])
    survivors = load_final_survivors([path])
    assert [s.agent_id for s in survivors] == [0, 1]
    assert survivors[0].cooperation_rate == pytest.approx(0.99)
    assert survivors[0].run_label == "exp_seed0"
    assert survivors[0].seed == "seed0"


def test_load_final_survivors_rejects_an_empty_population(tmp_path):
    """A structurally valid record with no final agents must fail loudly.

    ``validate_evolution_results`` only requires the ``final_population`` *key*,
    so an empty list reaches our own guard. Silently returning no survivors
    would turn the highlight layer into a no-op without telling anyone.
    """
    valid = _valid_record_data("exp_seed0", 0, ["a"], [1.0])
    valid["final_population"] = []
    path = tmp_path / "exp_seed0" / "evolutionary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(valid), encoding="utf-8")

    with pytest.raises(ValueError, match="empty final_population"):
        load_final_survivors([path])


def test_load_final_survivors_rejects_a_schema_violating_record(tmp_path):
    path = tmp_path / "bad" / "evolutionary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"final_population": []}), encoding="utf-8")

    with pytest.raises(EvolutionLogError):
        load_final_survivors([path])


def test_discover_latest_family_groups_seeds_of_one_experiment(tmp_path):
    """Seeds of the newest experiment are grouped; another family is excluded."""
    import os

    first = _write_record(tmp_path, "expA_seed0", 0, ["a"], [1.0])
    _write_record(tmp_path, "expA_seed1", 1, ["a"], [1.0])
    _write_record(tmp_path, "expB_seed0", 0, ["a"], [1.0])
    # both written just now; make the expB record the newest of all
    newest = tmp_path / "expB_seed0" / "evolutionary.json"
    os.utime(newest, None)
    os.utime(first, (1, 1))

    family = discover_latest_family(tmp_path)
    names = sorted(p.parent.name for p in family)
    assert names == ["expB_seed0"], names


def test_discover_latest_family_raises_when_nothing_matches(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_latest_family(tmp_path)


# ----------------------------------------------------------- summarising


def test_summarise_survivors_counts_high_cooperation_subset(tmp_path):
    paths = [
        _write_record(tmp_path, "exp_seed0", 0, ["a", "b", "c"], [0.99, 0.95, 0.10]),
        _write_record(tmp_path, "exp_seed1", 1, ["a", "d"], [1.00, 0.20]),
    ]
    survivors = load_final_survivors(paths)
    summary = summarise_survivors(survivors, 0.90)
    assert summary["runs"] == 2
    assert summary["individuals"] == 5
    assert summary["unique_codes"] == 4
    assert summary["high_cooperation_individuals"] == 3
    assert summary["high_cooperation_unique_codes"] == 2  # "a" and "b"
    assert summary["min_cooperation"] == pytest.approx(0.10)
    assert summary["max_cooperation"] == pytest.approx(1.00)


def test_summarise_survivors_threshold_is_inclusive(tmp_path):
    path = _write_record(tmp_path, "exp_seed0", 0, ["a"], [0.90])
    survivors = load_final_survivors([path])
    assert summarise_survivors(survivors, 0.90)["high_cooperation_individuals"] == 1
    assert summarise_survivors(survivors, 0.9001)["high_cooperation_individuals"] == 0


# ------------------------------------- cache matching / family discovery


def test_match_survivors_splits_on_cache_presence(tmp_path):
    path = _write_record(tmp_path, "exp_seed0", 0, ["cached", "absent"], [1.0, 0.5])
    survivors = load_final_survivors([path])
    drawable, missing = match_survivors_to_cache(survivors, {"cached"})
    assert [s.code for s in drawable] == ["cached"]
    assert [s.code for s in missing] == ["absent"]


def test_match_survivors_compares_comment_stripped_code(tmp_path):
    """The cache stores stripped code, so a commented survivor still matches."""
    commented = "# explain\ndef observe(A_rep):\n    return A_rep\n"
    path = _write_record(tmp_path, "exp_seed0", 0, [commented], [1.0])
    survivors = load_final_survivors([path])
    drawable, missing = match_survivors_to_cache(
        survivors, {strip_code_comments(commented)}
    )
    assert len(drawable) == 1
    assert missing == []


def test_survivor_coverage_uses_distinct_codes(tmp_path):
    """Duplicated codes must not inflate coverage."""
    path = _write_record(tmp_path, "exp_seed0", 0, ["x", "x", "y"], [1.0, 1.0, 1.0])
    survivors = load_final_survivors([path])
    # 1 of 2 distinct codes present -> 50%, not 2 of 3 individuals
    assert survivor_coverage(survivors, {"x"}) == pytest.approx(0.5)


def test_survivor_coverage_is_zero_for_an_empty_input():
    assert survivor_coverage([], {"x"}) == 0.0


def test_cache_aware_discovery_skips_an_unclustered_newer_family(tmp_path):
    """The core motivation: newest-by-mtime can have no cache overlap.

    An observability run finished later but was never fed to the clustering
    pipeline, so picking it purely by recency would highlight almost nothing.
    """
    import os

    good = _write_record(tmp_path, "good_seed0", 0, ["in_cache_a", "in_cache_b"], [1.0, 1.0])
    newer = _write_record(tmp_path, "unclustered_seed0", 0, ["nowhere_a", "nowhere_b"], [1.0, 0.9])
    os.utime(good, (1, 1))
    os.utime(newer, None)  # strictly newest

    # pure recency picks the unclustered one
    by_mtime = discover_latest_family(tmp_path)
    assert by_mtime[0].parent.name == "unclustered_seed0"

    # cache-aware discovery skips it for the covered family
    paths, coverage = discover_latest_archived_family(
        tmp_path, {"in_cache_a", "in_cache_b"}, min_coverage=0.5
    )
    assert paths[0].parent.name == "good_seed0"
    assert coverage == pytest.approx(1.0)


def test_cache_aware_discovery_falls_back_to_newest_when_nothing_qualifies(tmp_path):
    import os

    only = _write_record(tmp_path, "unclustered_seed0", 0, ["absent"], [1.0])
    os.utime(only, None)

    paths, coverage = discover_latest_archived_family(
        tmp_path, {"some_other_code"}, min_coverage=0.5
    )
    assert paths[0].parent.name == "unclustered_seed0"
    assert coverage == pytest.approx(0.0)


def test_cache_aware_discovery_raises_without_records(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_latest_archived_family(tmp_path, set())
