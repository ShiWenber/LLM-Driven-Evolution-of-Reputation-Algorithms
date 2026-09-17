"""Vectorize + cluster + PCA-project EVERY strategy code in the analysis cache.

Reads the global code set from ``results/.analysis_cache/strategy_analysis.sqlite3``
(one row per distinct strategy ever archived), embeds each code with the local
code transformer, clusters the embeddings, projects them to 2-D with centered
PCA, and names each cluster from a *uniform random sample* of its members.

Three things are overlaid on the scatter:

  * **canonical baseline markers** -- the ALLC / ALLD / L1-L8 strategies (plus
    IS, SH, SS, SJ) are embedded into the same space and labelled in place, so
    the figure shows where the evolved cloud sits relative to the hand-written
    norms.
  * **survivors of a final generation** -- taken from the ``final_population``
    section of evolution-log records, because the cache stores codes without
    provenance. They are drawn in a translucent accent colour.
  * **high-cooperation survivors** -- the subset that also clears a
    cooperation threshold, marked with a distinct outline. Survivors that
    exactly implement a canonical ``(observe, decide)`` norm get a third style.

Usage:
  uv run python -m experiments.analysis.plot_code_landscape
  uv run python -m experiments.analysis.plot_code_landscape --k 18 --out landscape.png
  uv run python -m experiments.analysis.plot_code_landscape --survivors-glob "results/.../seed*/evolutionary.json"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from experiments.analysis.clustering.cache import AnalysisCache
from experiments.analysis.clustering.cli_args import add_clustering_method_args
from experiments.analysis.clustering.survivors import (
    canonical_norm_tables,
    discover_latest_archived_family,
    load_final_survivors,
    match_norm,
    match_survivors_to_cache,
    name_clusters_from_samples,
    sample_cluster_codes,
    summarise_survivors,
    verify_canonical_tables,
)
from experiments.analysis.clustering.pipeline import (
    embed_codes,
    fit_clusterer,
)
from experiments.analysis.clustering.comments import strip_code_comments
from experiments.v2_quantitative.baselines import BASELINES

# Ward agglomerative clustering materialises a full pairwise distance matrix, so
# it needs O(n^2) memory: ~7 GB at 43k codes and beyond. The global landscape is
# therefore K-means-only past this size; the flag still works for small runs.
MAX_HIERARCHICAL_CODES = 5000

# Which baselines get a text label. Every distinct baseline that is drawn must be
# named: the marker glyph is identical whether or not a name accompanies it, so an
# unlabelled diamond is indistinguishable from a mislabelled one. IS and SH are not
# part of the leading eight (they are extra comparators) but they are still plotted,
# hence they are labelled too. Their separation is ~3% of the plotted span, so the
# de-overlap pass can place them on the main axes without a zoom inset.
DEFAULT_LABELLED_BASELINES = (
    "ALLC", "ALLD", "IS", "SH",
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
)

# ``baselines.BASELINES`` also carries synonym entries (SC == IS, SS == L3,
# SJ == L6). Only the distinct codes are embedded and drawn, otherwise the same
# marker would be plotted several times on top of itself.
CANONICAL_BASELINE_NAMES = (
    "ALLC", "ALLD", "IS", "SH",
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
)

# The zoom inset gets its own colour so the rectangle, the inset frame and its
# labels read as one object and are not confused with cluster colours (which are
# tab10/turbo) or the gold ring that marks a norm-matching survivor.
ZOOM_COLOR = "#7C3AED"
# Baselines closer than this fraction of the plotted span are treated as one
# crowd needing magnification. Measured separation: the leading eight sit
# 0.2-1.3% of the span apart, IS/SH 3.0%, ALLC/ALLD 15%.
ZOOM_GROUP_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Point styles. Named constants because the main axes AND the zoom inset draw the
# same layers: a magnified detail must show the same colours, sizes and alphas as
# the overview, so the two panels can be read as one figure.
# ---------------------------------------------------------------------------
CLOUD_POINT_SIZE = 7
CLOUD_POINT_ALPHA = 0.34
BASELINE_MARKER_SIZE = 190
SURVIVOR_POINT_SIZE = 150
SURVIVOR_ALPHA = 0.38
SURVIVOR_COLOR = "#17B3A3"
HIGH_COOP_EDGE = "#0B7A6F"
NORM_RING_EDGE = "#B8860B"
BASELINE_FACE = "#111827"

CLUSTER_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]


def _cluster_colors(n: int):
    if n <= len(CLUSTER_PALETTE):
        return CLUSTER_PALETTE[:n]
    cmap = plt.colormaps["turbo"]
    return [cmap(x) for x in np.linspace(0.05, 0.95, n)]


def check_clustering_method(method: str, n: int) -> None:
    """Reject configurations that cannot finish at this scale.

    ``fit_clusterer`` supports Ward agglomerative clustering, which needs the
    full pairwise distance matrix; at tens of thousands of codes that is many
    gigabytes. Failing early with guidance beats an out-of-memory kill.
    """
    if method == "hierarchical" and n > MAX_HIERARCHICAL_CODES:
        raise SystemExit(
            f"--clustering-method hierarchical needs O(n^2) memory and is limited "
            f"to {MAX_HIERARCHICAL_CODES} codes here (got {n}). Use --limit-codes "
            f"or the default kmeans."
        )


def choose_k(X, seed: int, *, method: str = "kmeans", minimum: int = 2,
             maximum: int = 20, sample: int = 6000) -> int:
    """Silhouette-based K selection, subsampled for large code sets.

    Full silhouette scoring is O(n^2); with tens of thousands of archived codes
    that is hours of work, so the score is estimated on a random subsample while
    the clustering itself still runs on every point.

    Silhouette on a large embedding space is biased toward very small K: with
    tens of thousands of near-continuum codes the argmax usually sits at K=2,
    which names the whole archive with two labels. That is a real property of
    the data, not a bug, so the search range is exposed instead of being
    second-guessed -- pass ``--k`` or ``--min-k`` for a finer partition.
    """
    n = len(X)
    if n < 3:
        print(f"-> too few codes ({n}); using K=1")
        return 1
    if n > sample:
        rng = np.random.default_rng(seed)
        subset = rng.choice(n, size=sample, replace=False)
        Xs = X[subset]
        print(f"  silhouette on a {sample}/{n} subsample")
    else:
        subset, Xs = None, X

    lower = max(2, minimum)
    upper = min(maximum, n - 1)
    if lower > upper:
        print(f"-> K range empty ({lower}..{upper}); using K=1")
        return 1

    best_k, best_score = lower, -1.0
    for k in range(lower, upper + 1):
        # Few restarts: this is only ranking candidate K values, and the winner
        # is re-fitted with the full n_init before anything is plotted.
        model = fit_clusterer(X, k, seed, method, n_init=3)
        scored_labels = (
            model.labels_ if subset is None else np.asarray(model.labels_)[subset]
        )
        score = silhouette_score(Xs, scored_labels)
        print(f"  K={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    print(f"-> chosen K={best_k} (silhouette={best_score:.4f})")
    return best_k


def read_cache_codes(cache_path, *, limit: int | None = None) -> list[str]:
    """Return every distinct strategy code in the cache, in a stable order."""
    cache = AnalysisCache(cache_path)
    with cache.connection() as connection:
        rows = connection.execute(
            "SELECT code_text FROM codes ORDER BY code_hash"
        ).fetchall()
    codes = [row[0] for row in rows]
    if limit is not None:
        codes = codes[:limit]
    if not codes:
        raise ValueError(f"no codes found in {cache.path}")
    return codes


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--k", type=int, default=None,
                    help="fixed cluster count; default = subsampled silhouette")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-k", type=int, default=2,
                    help="lower bound for automatic K selection (silhouette on a "
                         "large continuum-like space tends to peak at K=2)")
    ap.add_argument("--max-k", type=int, default=20,
                    help="upper bound for automatic K selection")
    ap.add_argument("--silhouette-sample", type=int, default=6000,
                    help="subsample size for silhouette scoring on large code sets")
    ap.add_argument("--limit-codes", type=int, default=None,
                    help="cap the number of cached codes (for quick drafts)")
    ap.add_argument("--name-samples", type=int, default=50,
                    help="member codes randomly sampled per cluster for LLM naming")
    ap.add_argument("--no-llm-names", action="store_true",
                    help="skip DeepSeek naming (offline runs; clusters stay unnamed)")
    ap.add_argument("--projection-sample", type=int, default=None,
                    help="subsample used to fit the PCA basis (all points are projected)")
    ap.add_argument("--survivors-glob", type=str, default=None,
                    help="glob of evolution-log records whose final_population is "
                         "highlighted (default: newest experiment family)")
    ap.add_argument("--survivors-root", type=str,
                    default="results/quantitative_baseline",
                    help="root searched for the newest experiment family")
    ap.add_argument("--no-survivors", action="store_true",
                    help="draw the landscape without the survivor overlay")
    ap.add_argument("--min-cooperation", type=float, default=0.90,
                    help="cooperation_rate at or above which a survivor counts as "
                         "high-cooperation (default 0.90)")
    ap.add_argument("--min-survivor-coverage", type=float, default=0.5,
                    help="when auto-selecting an experiment family, require this "
                         "fraction of its final codes to be present in the cache "
                         "(default 0.5)")
    ap.add_argument("--label-baselines", type=str,
                    default=",".join(DEFAULT_LABELLED_BASELINES),
                    help="comma-separated baseline names to label in place")
    ap.add_argument("--no-zoom-inset", dest="zoom_inset", action="store_false",
                    help="do not magnify the crowd of near-coincident baselines "
                         "(they overlap into one blob without it)")
    ap.add_argument("--zoom-group-threshold", type=float,
                    default=ZOOM_GROUP_THRESHOLD,
                    help="baselines closer than this fraction of the plotted span "
                         "count as one crowd needing magnification")
    ap.add_argument("--zoom-margin", type=float, default=0.6,
                    help="padding around the magnified region, as a fraction of "
                         "that region's size")
    ap.add_argument("--out", type=str, default="code_landscape_pca.png")
    ap.add_argument("--report", type=str, default=None,
                    help="path for a JSON sidecar of counts/norm matches "
                         "(default: <out>.json)")
    add_clustering_method_args(ap)
    return ap


def _label_alignment(dx: float, dy: float) -> tuple[str, str]:
    """Text alignment implied by a label's offset from its anchor.

    Shared by the placer and the drawer: the two must agree, otherwise the placer
    reasons about a box somewhere other than where the label actually renders.
    """
    horizontal = "left" if dx > 0 else ("right" if dx < 0 else "center")
    vertical = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
    return horizontal, vertical


def _label_box_px(anchor_px, offset, text, *, fontsize, px_per_pt):
    """Rendered label rectangle in display pixels, for placement planning.

    Derived from the *same* alignment rule the drawer uses, so the planned box and
    the drawn box coincide. Estimating a box that ignores the alignment is what
    previously made left/above labels collide while the planner believed they were
    clear.
    """
    dx, dy = offset
    anchor_x = anchor_px[0] + dx * px_per_pt
    anchor_y = anchor_px[1] + dy * px_per_pt
    width = 0.63 * fontsize * px_per_pt * (len(text) + 1.6)
    height = 1.9 * fontsize * px_per_pt
    horizontal, vertical = _label_alignment(dx, dy)
    if horizontal == "left":
        left = anchor_x
    elif horizontal == "right":
        left = anchor_x - width
    else:
        left = anchor_x - width / 2
    if vertical == "bottom":
        lower = anchor_y
    elif vertical == "top":
        lower = anchor_y - height
    else:  # centred: only reachable for a purely horizontal offset
        lower = anchor_y - height / 2
    return (left, left + width, lower, lower + height)


def _box_overlap_area(a, b) -> float:
    """Area shared by two ``(left, right, lower, upper)`` boxes."""
    width = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    return width * height


def _outside_area(box, bounds) -> float:
    """Area of ``box`` lying outside ``bounds``."""
    if bounds is None:
        return 0.0
    left = max(bounds[0], box[0])
    right = min(bounds[2], box[1])
    lower = max(bounds[1], box[2])
    upper = min(bounds[3], box[3])
    inside = max(0.0, right - left) * max(0.0, upper - lower)
    return (box[1] - box[0]) * (box[3] - box[2]) - inside


def _resolve_label_offsets(ax, fig, anchors, names, fontsize, *, bounds=None):
    """Pick per-label offsets (in points), scoring candidates for tidiness.

    The canonical norms are hand-written rules, so their embeddings land close
    together and naive in-place labels collide into an unreadable blob. Each label
    is given the cheapest offset from a ring of candidates, where the cost counts
    (in order) overlap with already-placed labels, area hanging outside the axes,
    and area covering another marker -- so labels stay near their own point and
    inside the panel instead of being flung to a free corner.

    Returns offsets in the same order as ``anchors``/``names``.
    """
    px_per_pt = fig.dpi / 72.0
    radii_pt = (20.0, 30.0, 42.0, 56.0, 72.0, 90.0, 110.0)
    # Above/below first (least ambiguous in a wide panel), then the diagonals,
    # then left/right. The rank breaks ties, so a label only reaches further out
    # when the nearer positions are genuinely taken.
    angle_preference = (90.0, 270.0, 45.0, 135.0, 225.0, 315.0, 0.0, 180.0)
    candidates = [
        (
            radius * np.cos(np.radians(angle)),
            radius * np.sin(np.radians(angle)),
            radius_rank * len(angle_preference) + angle_rank,
        )
        for radius_rank, radius in enumerate(radii_pt)
        for angle_rank, angle in enumerate(angle_preference)
    ]
    candidates.sort(key=lambda candidate: candidate[2])

    marker_radius_px = np.sqrt(BASELINE_MARKER_SIZE) / 2 * px_per_pt
    anchor_px = []
    for (x, y) in anchors:
        try:
            anchor_px.append(tuple(ax.transData.transform((x, y))))
        except Exception:  # noqa: BLE001 - transform unavailable
            anchor_px.append(None)

    marker_boxes = [
        None if point is None else (
            point[0] - marker_radius_px, point[0] + marker_radius_px,
            point[1] - marker_radius_px, point[1] + marker_radius_px,
        )
        for point in anchor_px
    ]

    # Strict priority order, not a weighted sum: a candidate that overlaps another
    # label is never chosen over one that does not, however much distance it saves.
    # Only inside a tier do the lower tiers matter.
    OVERLAP_WEIGHT = 1e9   # per px^2 of label-on-label overlap
    OUTSIDE_WEIGHT = 1e3   # per px^2 hanging outside the axes
    MARKER_WEIGHT = 1e2    # per px^2 covering another marker
    DISTANCE_WEIGHT = 1.0  # per px from the anchor (tie-break inside a tier)

    def place(order):
        """Greedily assign offsets, processing anchors in ``order``."""
        placed: list[tuple[float, float, float, float] | None] = [None] * len(names)
        chosen: list[tuple[float, float] | None] = [None] * len(names)
        accepted_boxes: list[tuple[float, float, float, float]] = []
        for index in order:
            point, name = anchor_px[index], names[index]
            if point is None:
                chosen[index] = candidates[0][:2]
                continue
            other_markers = [
                box for other_index, box in enumerate(marker_boxes)
                if box is not None and other_index != index
            ]
            best_offset = candidates[0][:2]
            best_box = _label_box_px(point, best_offset, name,
                                     fontsize=fontsize, px_per_pt=px_per_pt)
            best_cost = None
            for dx, dy, _rank in candidates:
                box = _label_box_px(point, (dx, dy), name,
                                    fontsize=fontsize, px_per_pt=px_per_pt)
                cost = OVERLAP_WEIGHT * sum(
                    _box_overlap_area(box, other) for other in accepted_boxes
                )
                cost += OUTSIDE_WEIGHT * _outside_area(box, bounds)
                cost += MARKER_WEIGHT * sum(
                    _box_overlap_area(box, marker) for marker in other_markers
                )
                cost += DISTANCE_WEIGHT * float(np.hypot(dx, dy))
                if best_cost is None or cost < best_cost:
                    best_cost, best_offset, best_box = cost, (dx, dy), box
            chosen[index] = best_offset
            placed[index] = best_box
            accepted_boxes.append(best_box)
        return chosen, placed

    def total_overlap(boxes):
        boxes = [box for box in boxes if box is not None]
        return sum(
            _box_overlap_area(a, b)
            for i, a in enumerate(boxes) for b in boxes[i + 1:]
        )

    natural = list(range(len(names)))
    # A purely sequential pass lets an early label take the only free slot a later
    # one needed. Retrying with the most crowded anchors first gives the tightest
    # spots to the labels that need them, and the better of the two layouts wins.
    crowded_first = sorted(
        natural,
        key=lambda index: sum(
            _box_overlap_area(marker_boxes[index], marker)
            for other, marker in enumerate(marker_boxes)
            if marker is not None and other != index
        ),
        reverse=True,
    )

    best_offsets, best_boxes = place(natural)
    if total_overlap(best_boxes) > 0:
        retry_offsets, retry_boxes = place(crowded_first)
        if total_overlap(retry_boxes) < total_overlap(best_boxes):
            best_offsets, best_boxes = retry_offsets, retry_boxes
    return best_offsets


def _point_box_gap(point, box) -> float:
    """Shortest distance from a point to a ``(left, right, lower, upper)`` box."""
    dx = max(box[0] - point[0], 0.0, point[0] - box[1])
    dy = max(box[2] - point[1], 0.0, point[1] - box[3])
    return float(np.hypot(dx, dy))


# Above this gap (in pixels) a label is far enough from its marker that the
# association is no longer obvious, so a leader line is drawn. Close labels stay
# clean; crowded clusters stay readable.
LEADER_MIN_GAP_PX = 12.0


def _draw_baseline_labels(
    ax, fig, anchors, names, offsets, *,
    fontsize, facecolor, edgecolor, zorder: int = 10, always_leader: bool = False,
):
    """Draw placed labels, with a leader line whenever a label drifts away.

    Shared by the main axes and the zoom inset. Each label keeps a thin connector
    to its own marker once it is pushed clear of it, which is what makes a crowded
    cluster legible: without it, a name sitting in free space cannot be traced back
    to the point it describes.

    ``always_leader`` forces a connector regardless of distance -- the zoom inset
    uses it, because resolving exactly that ambiguity is the inset's whole purpose.
    """
    px_per_pt = fig.dpi / 72.0
    marker_shrink_pt = float(np.sqrt(BASELINE_MARKER_SIZE) / 2)
    annotations = []
    for name, (x, y), (dx, dy) in zip(names, anchors, offsets):
        horizontal, vertical = _label_alignment(dx, dy)
        needs_leader = always_leader
        if not needs_leader:
            try:
                anchor_px = tuple(ax.transData.transform((x, y)))
                box = _label_box_px(anchor_px, (dx, dy), name,
                                    fontsize=fontsize, px_per_pt=px_per_pt)
                needs_leader = _point_box_gap(anchor_px, box) > LEADER_MIN_GAP_PX
            except Exception:  # noqa: BLE001 - transform unavailable
                needs_leader = False
        arrowprops = None
        if needs_leader:
            arrowprops = {
                "arrowstyle": "-", "color": edgecolor, "linewidth": 0.8,
                "shrinkA": 2.0, "shrinkB": marker_shrink_pt,
                "connectionstyle": "arc3,rad=0", "zorder": zorder - 1,
            }
        annotations.append((name, ax.annotate(
            name, (x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=fontsize, fontweight="bold", color="#111827",
            ha=horizontal, va=vertical, zorder=zorder,
            arrowprops=arrowprops,
            bbox={"boxstyle": "round,pad=0.24", "facecolor": facecolor,
                  "edgecolor": edgecolor, "linewidth": 0.8, "alpha": 0.95},
        )))
    return annotations


def _single_linkage_groups(points, threshold: float) -> list[list[int]]:
    """Group indices so any two members are chained within ``threshold``."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    seen = [False] * n
    groups: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        members = []
        while stack:
            k = stack.pop()
            members.append(k)
            distances = np.linalg.norm(points - points[k], axis=1)
            for j in np.flatnonzero(distances <= threshold):
                if not seen[j]:
                    seen[j] = True
                    stack.append(int(j))
        groups.append(sorted(members))
    return groups


def _zoom_window(
    points, span: float, *, margin_fraction: float = 0.6, min_span_fraction: float = 0.004
) -> tuple[float, float, float, float]:
    """Bounding box around ``points``, padded so labels have room.

    ``min_span_fraction`` guards the degenerate case of near-identical points,
    where an unpadded box would be a zero-width window.
    """
    points = np.asarray(points, dtype=float)
    low = points.min(axis=0)
    high = points.max(axis=0)
    minimum = min_span_fraction * span
    width = max(high[0] - low[0], minimum)
    height = max(high[1] - low[1], minimum)
    centre = 0.5 * (low + high)
    margin = margin_fraction * max(width, height)
    return (
        float(centre[0] - width / 2 - margin),
        float(centre[0] + width / 2 + margin),
        float(centre[1] - height / 2 - margin),
        float(centre[1] + height / 2 + margin),
    )


def _inset_box(
    ax, window, *, target_long_side: float = 0.34, max_side: float = 0.46
) -> tuple[float, float]:
    """Inset size ``(w, h)`` in axes fraction that shows ``window`` undistorted.

    The box aspect is derived from the live axes transform, so the inset presents
    the same geometry as the main plot rather than stretching the window to fit an
    arbitrary rectangle.
    """
    wx0, wx1, wy0, wy1 = window
    extent = ax.get_window_extent()
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    pixels_per_x = extent.width / max(x_hi - x_lo, 1e-12)
    pixels_per_y = extent.height / max(y_hi - y_lo, 1e-12)
    want_w = max(wx1 - wx0, 1e-12) * pixels_per_x
    want_h = max(wy1 - wy0, 1e-12) * pixels_per_y
    width_fraction = target_long_side
    height_fraction = (
        width_fraction * (extent.width / extent.height) * (want_h / want_w)
    )
    longest = max(width_fraction, height_fraction)
    if longest > max_side:
        scale = max_side / longest
        width_fraction *= scale
        height_fraction *= scale
    return width_fraction, height_fraction


def _place_inset(ax, box, Z, obstacles) -> list[float]:
    """Pick the corner of the axes that hides the fewest points.

    ``obstacles`` are display-space points (cluster captions) that must never be
    covered; any candidate containing one is rejected outright.
    """
    width_fraction, height_fraction = box
    extent = ax.get_window_extent()
    display = np.asarray(ax.transData.transform(Z), dtype=float)
    best_score = None
    best = (0.02, 0.02)
    x_options = (0.02, max(0.02, 1.0 - width_fraction - 0.02))
    y_options = (0.02, max(0.02, 1.0 - height_fraction - 0.02))
    for x0 in x_options:
        for y0 in y_options:
            left = x0 * extent.width
            bottom = y0 * extent.height
            right = left + width_fraction * extent.width
            top = bottom + height_fraction * extent.height
            inside = (
                (display[:, 0] >= left) & (display[:, 0] <= right)
                & (display[:, 1] >= bottom) & (display[:, 1] <= top)
            )
            score = int(inside.sum())
            for ox, oy in obstacles:
                if left <= ox <= right and bottom <= oy <= top:
                    score += 10 ** 6
            if best_score is None or score < best_score:
                best_score, best = score, (x0, y0)
    return [best[0], best[1], width_fraction, height_fraction]


def _label_rectangles(annotations, fig) -> list[dict]:
    """Record each annotation's rendered box for the sidecar report.

    Label placement is otherwise only checkable by re-detecting coloured regions
    in the saved PNG, which is both fragile (a label's background is split by the
    glyphs and grid lines drawn over it) and blind to the cause of a collision.
    Recording the boxes lets overlap be asserted exactly, in display pixels.

    An annotation without a usable box is skipped rather than fatal: the report
    is diagnostic, and dropping the figure over a missing measurement would be
    the wrong trade.
    """
    renderer = fig.canvas.get_renderer()
    boxes = []
    for name, annotation in annotations:
        try:
            box = annotation.get_window_extent(renderer=renderer)
        except Exception as exc:  # noqa: BLE001 - renderer without a usable box
            print(f"  note: no label box for {name!r} ({type(exc).__name__})")
            box = None
        if box is None:
            continue
        boxes.append(
            {
                "name": name,
                "x0": float(box.x0),
                "y0": float(box.y0),
                "x1": float(box.x1),
                "y1": float(box.y1),
            }
        )
    return boxes


def label_overlaps(boxes: list[dict]) -> list[tuple[str, str]]:
    """Return the pairs of recorded label boxes that intersect."""
    collisions = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if a["x0"] < b["x1"] and a["x1"] > b["x0"] and a["y0"] < b["y1"] and a["y1"] > b["y0"]:
                collisions.append((a["name"], b["name"]))
    return collisions


def _in_window(Z: np.ndarray, window) -> np.ndarray:
    """Boolean mask of the points inside an ``(x0, x1, y0, y1)`` data window."""
    x0, x1, y0, y1 = window
    return (
        (Z[:, 0] >= x0) & (Z[:, 0] <= x1) & (Z[:, 1] >= y0) & (Z[:, 1] <= y1)
    )


def _plot_cloud(
    ax, Z, labels, colors, *, names=None, window=None, zorder: int = 2
) -> None:
    """Scatter the strategy cloud, one call per cluster, in this axes' frame.

    Shared by the main axes and the zoom inset so the magnified detail shows
    exactly the same marks as the overview: same per-cluster colour, same marker
    size and alpha. Recolouring the inset (or dropping it to a single grey) would
    make the two panels show different-looking populations, which defeats the
    point of magnifying a region of the first one.

    ``names`` enables legend entries (main axes only -- the inset has no legend).
    ``window`` restricts the drawn subset; ``None`` draws everything.
    """
    if len(labels) != len(Z):
        # A silent mismatch would filter with one array and index with the other,
        # drawing a plausible-looking but wrong subset.
        raise ValueError(
            f"labels ({len(labels)}) and points ({len(Z)}) must describe the same "
            "rows: cluster labels cover every embedded row, baselines included"
        )
    inside = None if window is None else _in_window(Z, window)
    for cluster_id in range(len(colors)):
        mask = labels == cluster_id
        if inside is not None:
            mask = mask & inside
        if not mask.any():
            continue
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            s=CLOUD_POINT_SIZE, alpha=CLOUD_POINT_ALPHA,
            color=colors[cluster_id], linewidths=0, zorder=zorder,
            label=(names[cluster_id] if names is not None else None),
        )


def _plot_survivor_layers(
    ax,
    points: np.ndarray,
    cooperation: np.ndarray,
    *,
    min_cooperation: float,
    window=None,
    with_legend: bool = False,
) -> None:
    """Draw the three survivor marks with the styles used on every panel.

    Same reasoning as :func:`_plot_cloud`: the inset must reproduce these marks
    rather than omit them, otherwise a survivor sitting inside the magnified
    region would silently vanish from the figure.
    """
    if len(points) == 0:
        return
    keep = np.ones(len(points), dtype=bool) if window is None else _in_window(points, window)
    if not keep.any():
        return
    x, y = points[keep, 0], points[keep, 1]
    high = cooperation[keep] >= min_cooperation

    ax.scatter(
        x, y, s=SURVIVOR_POINT_SIZE, alpha=SURVIVOR_ALPHA, color=SURVIVOR_COLOR,
        edgecolor="none", zorder=6,
        label="final-generation survivor" if with_legend else None,
    )
    if high.any():
        ax.scatter(
            x[high], y[high], s=SURVIVOR_POINT_SIZE, facecolors="none",
            edgecolor=HIGH_COOP_EDGE, linewidths=1.9, zorder=7,
            label=(f"high-cooperation survivor (coop ≥ {min_cooperation:g})"
                   if with_legend else None),
        )


def _draw_zoom_inset(
    ax,
    fig,
    labelled,
    Z,
    obstacles,
    *,
    labels=None,
    colors=None,
    survivor_points=None,
    survivor_cooperation=None,
    min_cooperation: float = 0.90,
    group_threshold: float = ZOOM_GROUP_THRESHOLD,
    margin_fraction: float = 0.6,
    label_fontsize: float = 9.0,
):
    """Magnify the tightest crowd of canonical baselines into an inset panel.

    The leading-eight norms are near-coincident once embedded -- pairwise
    distances are 0.22-1.26% of the plotted span, i.e. about three pixels apart
    -- so at full scale their diamonds merge and no label placement can say which
    name belongs to which point. The largest chained group of baselines is
    therefore boxed and re-drawn magnified, with its labels placed inside the
    inset where there is room.

    The cloud and survivor layers are replayed through the same helpers the main
    axes use, so the inset is a true magnification rather than a restyled
    redrawing: identical per-cluster colours, marker sizes and alphas.

    Returns ``(names_in_inset, setup)``; ``([], None)`` when no grouping is tight
    enough to need magnification. ``setup`` carries the geometry for the sidecar.
    """
    if len(labelled) < 3:
        return [], None

    positions = np.asarray([position for _, position in labelled], dtype=float)
    span = float(max(np.ptp(Z[:, 0]), np.ptp(Z[:, 1])))
    groups = [
        group
        for group in _single_linkage_groups(positions, group_threshold * span)
        if len(group) >= 3
    ]
    if not groups:
        return [], None

    group = max(groups, key=len)
    window = _zoom_window(positions[group], span, margin_fraction=margin_fraction)
    names = [labelled[i][0] for i in group]
    position_of = {name: position for name, position in labelled}
    # Drawn outside the inset, these names would duplicate the inset's labels.
    wx0, wx1, wy0, wy1 = window
    x_lo, x_hi = ax.get_xlim()
    magnification = (x_hi - x_lo) / max(wx1 - wx0, 1e-12)

    ax.add_patch(
        # ``plt.Rectangle`` is matplotlib's documented pyplot alias for
        # ``matplotlib.patches.Rectangle``.
        plt.Rectangle(
            (wx0, wy0), wx1 - wx0, wy1 - wy0, fill=False, edgecolor=ZOOM_COLOR,
            linewidth=1.5, linestyle=(0, (4, 2)), zorder=11,
        )
    )

    box = _inset_box(ax, window)
    inset = ax.inset_axes(_place_inset(ax, box, Z, obstacles), facecolor="white",
                          zorder=12)
    inset.set_xlim(wx0, wx1)
    inset.set_ylim(wy0, wy1)
    for spine in inset.spines.values():
        spine.set_color(ZOOM_COLOR)
        spine.set_linewidth(1.5)

    inside = _in_window(Z, window)
    # Replay the same layers with the same styles the main axes used, restricted
    # to the magnified window -- identical colour/size/alpha, not a restyle.
    if labels is not None and colors is not None:
        _plot_cloud(inset, Z, labels, colors, window=window, zorder=1)
    if survivor_points is not None and survivor_cooperation is not None:
        _plot_survivor_layers(
            inset, survivor_points, survivor_cooperation,
            min_cooperation=min_cooperation, window=window,
        )
    inset.grid(color="#EEF1F5", linewidth=0.6, alpha=0.9)
    inset.set_axisbelow(True)
    inset.tick_params(labelsize=7, colors="#667085", length=3)
    inset.set_title(f"{', '.join(names)} detail · x{magnification:.0f}",
                    fontsize=8.5, color=ZOOM_COLOR, pad=4)

    anchors = []
    for name in names:
        x, y = position_of[name]
        anchors.append((x, y))
        inset.scatter(x, y, s=BASELINE_MARKER_SIZE, marker="D",
                      facecolor=BASELINE_FACE, edgecolor="white",
                      linewidths=1.3, zorder=9)

    # The inset was created after the canvas draw, so its transform needs one
    # more render before offsets can be resolved in real pixels.
    fig.canvas.draw()
    offsets = _resolve_label_offsets(
        inset, fig, anchors, names, fontsize=label_fontsize,
        bounds=tuple(inset.get_window_extent().extents),
    )
    annotations = _draw_baseline_labels(
        inset, fig, anchors, names, offsets,
        fontsize=label_fontsize, facecolor="#F3E8FF", edgecolor=ZOOM_COLOR,
        always_leader=True,
    )
    fig.canvas.draw()
    label_boxes = _label_rectangles(annotations, fig)
    collisions = label_overlaps(label_boxes)
    if collisions:
        print(f"  WARNING: inset labels overlap: {collisions}")

    print(f"  zoom inset: {', '.join(names)} · x{magnification:.0f}"
          f" · {int(inside.sum()):,} context points")
    setup = {
        "names": names,
        "window": [float(v) for v in window],
        "magnification": float(magnification),
        "inset_box": [float(v) for v in box],
        "context_points": int(inside.sum()),
        "label_boxes": label_boxes,
        "label_overlaps": collisions,
    }
    return names, setup


def plot_landscape(
    *,
    codes: list[str],
    X: np.ndarray,
    labels: np.ndarray,
    Z: np.ndarray,
    pca_variance,
    cluster_names: dict[int, str],
    baseline_names: list[str],
    survivors=None,
    labelled_baselines=(),
    norm_tables=None,
    min_cooperation: float = 0.90,
    survivor_source=None,
    zoom_inset: bool = True,
    zoom_group_threshold: float = ZOOM_GROUP_THRESHOLD,
    zoom_margin: float = 0.6,
    out_path: Path,
    clustering_method: str = "kmeans",
):
    """Render the landscape: clusters + baseline markers + survivor overlays.

    ``codes`` must carry the same text that was embedded, and its rows for cache
    codes are already comment-stripped (that is what ``embed_codes`` stores), so
    the lookup index is built directly from it. Only the handful of queries --
    survivors and baselines -- are stripped at lookup time, because stripping all
    tens of thousands of codes here would dominate the runtime.
    """
    strip_index = {code: i for i, code in enumerate(codes)}

    n_clusters = len({int(x) for x in labels})
    colors = _cluster_colors(n_clusters)

    fig, ax = plt.subplots(figsize=(13.5, 8.6), facecolor="white")
    ax.set_axisbelow(True)
    ax.grid(color="#E3E7ED", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAB2BF")

    # --- the global cloud, one scatter per cluster so the legend carries names
    caption_anchors: list[tuple[float, float]] = []
    _plot_cloud(
        ax, Z, labels, colors,
        names=[f"C{c} · {cluster_names.get(c, f'cluster {c}')}"
               for c in range(n_clusters)],
    )
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        name = cluster_names.get(cluster_id, f"cluster {cluster_id}")
        # in-place cluster caption at the cloud's median position
        if mask.sum():
            anchor = (float(np.median(Z[mask, 0])), float(np.median(Z[mask, 1])))
            caption_anchors.append(anchor)
            ax.annotate(
                f"C{cluster_id} · {name}",
                anchor,
                fontsize=8, color="#2b3648", ha="center", va="center", zorder=5,
                fontweight="semibold",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
                      "edgecolor": colors[cluster_id], "alpha": 0.86,
                      "linewidth": 1.0},
            )

    # --- survivor overlays
    survivor_rows: list[dict] = []
    survivor_points = None
    survivor_cooperation = None
    if survivors:
        rows = []
        for s in survivors:
            idx = strip_index.get(strip_code_comments(s.code))
            if idx is None:
                continue
            norm = match_norm(s.code, norm_tables) if norm_tables else None
            rows.append((s, idx, norm))
            survivor_rows.append(
                {
                    "run": s.run_label, "agent_id": s.agent_id,
                    "cooperation_rate": s.cooperation_rate, "canonical_norm": norm,
                }
            )
        if rows:
            survivor_points = np.array([[Z[i, 0], Z[i, 1]] for _, i, _ in rows])
            survivor_cooperation = np.array([s.cooperation_rate for s, _, _ in rows])

            _plot_survivor_layers(
                ax, survivor_points, survivor_cooperation,
                min_cooperation=min_cooperation, with_legend=True,
            )

            # 3. survivors that are exactly a canonical norm: third style
            normed = np.array([n is not None and n != "?" for _, _, n in rows])
            if normed.any():
                ax.scatter(
                    survivor_points[normed, 0], survivor_points[normed, 1],
                    s=300, facecolors="none",
                    edgecolor=NORM_RING_EDGE, linewidths=1.4, linestyle=(0, (2, 1.6)),
                    zorder=8, label="survivor matching a canonical norm",
                )

    # --- canonical baselines, drawn last. Labels are placed after the axes are
    # finalised so the de-overlap pass works in real screen coordinates.
    baseline_xy: dict[str, tuple[float, float]] = {}
    for name in baseline_names:
        idx = strip_index.get(strip_code_comments(BASELINES[name]))
        if idx is None:
            print(f"  baseline {name}: not among the clustered codes (skipped)")
            continue
        x, y = float(Z[idx, 0]), float(Z[idx, 1])
        baseline_xy[name] = (x, y)
        ax.scatter(x, y, s=BASELINE_MARKER_SIZE, marker="D", facecolor=BASELINE_FACE,
                   edgecolor="white", linewidths=1.3, zorder=9)
    if baseline_xy:
        ax.scatter([], [], s=BASELINE_MARKER_SIZE, marker="D", facecolor=BASELINE_FACE,
                   edgecolor="white", linewidths=1.3, label="canonical baseline")

    ax.set_xlabel(f"PCA component 1 · {pca_variance[0]:.1%} variance",
                  fontsize=11, color="#344054", labelpad=8)
    ax.set_ylabel(f"PCA component 2 · {pca_variance[1]:.1%} variance",
                  fontsize=11, color="#344054", labelpad=8)
    ax.set_title("Global strategy-code landscape", loc="left", fontsize=17,
                 fontweight="semibold", color="#172033", pad=24)
    method_label = "Ward hierarchical" if clustering_method == "hierarchical" else "K-means"
    subtitle = (
        f"{len(codes):,} archived strategy codes · code embedding + {method_label} "
        f"· K = {n_clusters} · centered PCA"
    )
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=9.5,
            color="#667085", va="bottom")
    # Spell out which run supplied the highlight: the survivor layer is loaded
    # from an evolution-log record, not from the cache, so the figure must not
    # leave the reader guessing which experiment it refers to.
    if survivor_source:
        names = sorted({label.rsplit("_seed", 1)[0] for label in survivor_source})
        ax.text(0, 1.048,
                "highlighted survivors from: " + ", ".join(names),
                transform=ax.transAxes, fontsize=8.6, color="#0B7A6F", va="bottom")

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.2,
              frameon=False, title="Clusters, baselines, survivors",
              title_fontsize=9.2)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    # Both the label de-overlap pass and the inset placement work in display
    # pixels, so the renderer has to reflect the final layout first.
    fig.canvas.draw()

    labelled = [
        (name, baseline_xy[name])
        for name in baseline_names
        if name in baseline_xy and name in labelled_baselines
    ]
    obstacles = [tuple(ax.transData.transform(anchor)) for anchor in caption_anchors]

    # Canonical norms that land on top of each other go into a magnified inset;
    # their labels are drawn there instead of on the main axes.
    inset_names: list[str] = []
    zoom_setup = None
    if zoom_inset:
        inset_names, zoom_setup = _draw_zoom_inset(
            ax, fig, labelled, Z, obstacles,
            labels=labels, colors=colors,
            survivor_points=survivor_points,
            survivor_cooperation=survivor_cooperation,
            min_cooperation=min_cooperation,
            group_threshold=zoom_group_threshold, margin_fraction=zoom_margin,
        )

    # Labels come last: the de-overlap pass needs the final data-to-pixel
    # transform, which only exists once the axes limits and layout are settled.
    on_main = [(name, position) for name, position in labelled
               if name not in inset_names]
    offsets = _resolve_label_offsets(
        ax, fig,
        [position for _, position in on_main],
        [name for name, _ in on_main],
        fontsize=9.5,
        bounds=tuple(ax.get_window_extent().extents),
    )
    main_annotations = _draw_baseline_labels(
        ax, fig,
        [position for _, position in on_main],
        [name for name, _ in on_main],
        offsets,
        fontsize=9.5, facecolor="#FFF7CC", edgecolor="#111827",
    )
    unlabelled = [n for n in baseline_xy if n not in labelled_baselines]
    if unlabelled:
        print(f"  baselines drawn without an inline label: {', '.join(sorted(unlabelled))}")

    fig.canvas.draw()
    main_boxes = _label_rectangles(main_annotations, fig)
    main_overlaps = label_overlaps(main_boxes)
    if main_overlaps:
        print(f"  WARNING: main-axes labels overlap: {main_overlaps}")
    print(f"  labels: {len(main_boxes)} on the main axes + "
          f"{len((zoom_setup or {}).get('label_boxes', []))} in the inset, "
          f"{len(main_overlaps)} overlaps")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "survivor_rows": survivor_rows,
        "baseline_positions": {k: [float(v[0]), float(v[1])] for k, v in baseline_xy.items()},
        "zoom": zoom_setup,
        "main_label_boxes": main_boxes,
        "main_label_overlaps": main_overlaps,
        "unlabelled_baselines": sorted(unlabelled),
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    out_path = Path(args.out)
    report_path = Path(args.report) if args.report else out_path.with_suffix(".json")
    cache = AnalysisCache(args.embedding_cache_path)

    # ---- 1. global code set -------------------------------------------------
    codes = read_cache_codes(args.embedding_cache_path, limit=args.limit_codes)
    print(f"cached strategy codes: {len(codes):,}")

    # ---- 2. baselines join the same embedding space -------------------------
    # Stored pre-stripped: ``embed_codes`` strips internally anyway, and keeping
    # the same text in ``codes`` lets the draw-time lookup index be exact.
    baseline_names = [n for n in CANONICAL_BASELINE_NAMES if n in BASELINES]
    codes = codes + [strip_code_comments(BASELINES[name]) for name in baseline_names]
    print(f"plus {len(baseline_names)} canonical baselines: {len(codes):,} codes total")

    # ---- 3. embed ----------------------------------------------------------
    cache_code_set = set(codes)
    X, _ = embed_codes(
        codes,
        code_embedding_model=args.code_embedding_model,
        code_embedding_revision=args.code_embedding_revision,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_cache=args.embedding_cache,
        embedding_cache_path=args.embedding_cache_path,
    )

    # ---- 4. cluster --------------------------------------------------------
    check_clustering_method(args.clustering_method, len(X))
    k = args.k if args.k is not None else choose_k(
        X, args.seed, method=args.clustering_method,
        minimum=args.min_k, maximum=args.max_k, sample=args.silhouette_sample,
    )
    km = fit_clusterer(X, k, args.seed, args.clustering_method)
    labels = np.asarray(km.labels_)
    print(f"clustered into K={k} ({args.clustering_method})")

    # ---- 5. PCA ------------------------------------------------------------
    if args.projection_sample and args.projection_sample < len(X):
        rng = np.random.default_rng(args.seed)
        fit_rows = rng.choice(len(X), size=args.projection_sample, replace=False)
        reducer = PCA(n_components=2, random_state=args.seed).fit(X[fit_rows])
    else:
        reducer = PCA(n_components=2, random_state=args.seed).fit(X)
    Z = reducer.transform(X)
    pca_variance = [float(v) for v in reducer.explained_variance_ratio_]
    print(f"PCA variance explained: {pca_variance[0]:.3f} / {pca_variance[1]:.3f}")

    # ---- 6. name clusters from a random sample ------------------------------
    samples = sample_cluster_codes(labels, codes, args.name_samples, seed=args.seed)
    cluster_names = name_clusters_from_samples(
        samples,
        llm_model=args.llm_model,
        cache=cache if args.embedding_cache else None,
        use_llm=not args.no_llm_names,
    )
    for cluster_id in sorted(cluster_names):
        print(f"  C{cluster_id}: {cluster_names[cluster_id]}  (n={int((labels==cluster_id).sum())})")

    # ---- 7. survivors ------------------------------------------------------
    survivors = None
    norm_tables = None
    unmatched_survivors: list = []
    coverage = None
    survivor_record_paths: list[str] = []
    guard = verify_canonical_tables()
    if guard:
        print("WARNING: canonical truth-table probe disagrees with baselines.py:")
        for line in guard:
            print(f"  {line}")
    else:
        print("norm probe guard: OK (L1-L8 tables reproduced exactly)")

    if not args.no_survivors:
        if args.survivors_glob:
            from glob import glob as _glob

            paths = sorted(_glob(args.survivors_glob))
            if not paths:
                raise FileNotFoundError(f"no records match {args.survivors_glob!r}")
            coverage = None
        else:
            paths, coverage = discover_latest_archived_family(
                args.survivors_root, cache_code_set,
                min_coverage=args.min_survivor_coverage,
            )
        print(f"survivor source: {len(paths)} record(s)"
              + (f", cache coverage {coverage:.0%}" if coverage is not None else ""))
        for p in paths:
            print(f"  {p}")
        survivor_record_paths = [str(p) for p in paths]
        all_survivors = load_final_survivors(paths)
        # Only survivors that are actually in the clustered code set can be drawn;
        # reporting the rest would misstate the figure.
        survivors, unmatched_survivors = match_survivors_to_cache(
            all_survivors, cache_code_set
        )
        if unmatched_survivors:
            print(
                f"  WARNING: {len(unmatched_survivors)}/{len(all_survivors)} "
                f"survivors are absent from the cache (their run was never "
                f"clustered) and cannot be drawn"
            )
        norm_tables = canonical_norm_tables()
        print("drawn survivor summary:", json.dumps(
            summarise_survivors(survivors, args.min_cooperation), sort_keys=True))
        print(f"  unmatched: {len(unmatched_survivors)} of {len(all_survivors)}")

    labelled = tuple(n.strip() for n in args.label_baselines.split(",") if n.strip())

    # ---- 8. draw -----------------------------------------------------------
    drawn = plot_landscape(
        codes=codes,
        X=X,
        labels=labels,
        Z=Z,
        pca_variance=pca_variance,
        cluster_names=cluster_names,
        baseline_names=baseline_names,
        survivors=survivors,
        labelled_baselines=labelled,
        norm_tables=norm_tables,
        min_cooperation=args.min_cooperation,
        survivor_source=[s.run_label for s in survivors] if survivors else None,
        zoom_inset=args.zoom_inset,
        zoom_group_threshold=args.zoom_group_threshold,
        zoom_margin=args.zoom_margin,
        out_path=out_path,
        clustering_method=args.clustering_method,
    )
    survivor_rows = drawn["survivor_rows"]

    # ---- 9. sidecar report -------------------------------------------------
    report = {
        "clustering_method": args.clustering_method,
        "codes": len(codes),
        "cached_codes": len(codes) - len(baseline_names),
        "k": int(k),
        "seed": args.seed,
        "pca_explained_variance": pca_variance,
        "cluster_names": {str(k_): v for k_, v in cluster_names.items()},
        "cluster_sizes": {str(c): int((labels == c).sum()) for c in range(int(k))},
        "naming_sample_size": args.name_samples,
        "min_cooperation": args.min_cooperation,
        "norm_probe_guard": guard,
        "baseline_positions": drawn["baseline_positions"],
        "zoom_inset": drawn["zoom"],
        "main_label_boxes": drawn["main_label_boxes"],
        "main_label_overlaps": drawn["main_label_overlaps"],
        "unlabelled_baselines": drawn["unlabelled_baselines"],
        "survivor_source_records": survivor_record_paths,
        "survivor_cache_coverage": coverage,
        "survivors": survivor_rows,
        "survivors_unmatched_individuals": len(unmatched_survivors),
    }
    if survivors:
        report["survivor_summary"] = summarise_survivors(
            survivors, args.min_cooperation
        )
    if unmatched_survivors:
        report["survivor_summary_all"] = summarise_survivors(
            survivors + unmatched_survivors, args.min_cooperation
        )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {out_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
