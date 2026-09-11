"""Per-target reputation **matrix** heatmaps for evolved populations.

Shows the raw private reputation matrix itself - the object any scalar summary
is computed from - so "do observers put the same *number* on a target?" can be
answered by looking rather than by trusting a statistic.

Each run gets one **column**, drawn as two square heatmaps:

* **top - reputation matrix.**  ``[observer, target]``; cell colour is the
  private score that observer holds for that target.  The narrow bar chart to
  its right is the observer's *row mean* - its severity/leniency, i.e. the
  constant offset that rank-based and sign-based statistics cannot see.
* **bottom - pairwise disagreement matrix.**  ``[observer, observer]``; cell
  ``(i, j)`` is the mean ``|score_i - score_j|`` over the targets they both
  rated.

Reading rules:

* cells with **no entry** (the diagonal - an agent never rates itself as a
  third party - and unseen targets under partial observability) are drawn in
  grey, so they cannot be confused with a genuine 0.0;
* white gridlines separate cells, so a matrix whose rows are each constant
  still shows its cell structure instead of collapsing into one flat block;
* the caption under each column carries the single metric ``D``, so the number
  is readable without leaving the figure.

The pair is deliberately redundant: a matrix whose rows are each flat (what
LLM-evolved populations actually produce) looks *tidy* in the top row - one
colour per row - yet the bottom row shows those rows sit at very different
levels.  That is what makes "everyone agrees who is good" and "everyone gives
the same number" two different statements.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from ..paths import project_root
from .core import (
    build_from_codes,
    disagreement,
    load_run,
    play_generation,
)

ROOT = project_root()
DEFAULT_OUTPUT = ROOT / "README.assets" / "private_reputation_matrix.png"

# A diverging map centred on the neutral prior: blue = judged bad, red = good.
REPUTATION_CMAP = "RdBu_r"
DISAGREEMENT_CMAP = "Reds"
# "No entry" must not look like a value.  Both colour maps are near-white at
# one end (RdBu_r at 0.0, Reds at 0.0), so a light grey is reserved for cells
# that hold nothing: the diagonal (never self-rated) and unseen targets.
MISSING_COLOR = "#BDBDBD"
# Observer severity bars reuse the reputation palette so a red bar reads as
# "lenient" and a blue one as "severe", matching the matrix above them.
SEVERITY_CMAP = matplotlib.colormaps["RdBu_r"]

# Above this many agents per side the per-cell number becomes unreadable, so it
# is suppressed and the colour bar carries the values alone.
ANNOTATE_MAX_AGENTS = 12

# Colour scale for the pairwise-disagreement map.  ``mean|a-b|`` can reach 2.0
# in principle but 1.0 is already "the two observers sit at opposite clip
# bounds"; fixing the scale (instead of autoscaling per run) keeps the columns
# comparable, at the cost of saturating the extreme pairs.
PAIR_VMAX = 1.0


# ---------------------------------------------------------------------------
# Pure helpers (no matplotlib) - these are what the tests exercise
# ---------------------------------------------------------------------------


def observer_matrix(
    codes: list[str],
    size: int,
    *,
    seed: int = 0,
    interactions: int = 1_000,
    action_error: float = 0.0,
    observation_error: float = 0.0,
    observability: str = "full",
    observability_p: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Replay one generation and return the raw private reputation matrix.

    Returns ``(reputation, observed, agent_ids)`` where ``reputation`` is
    ``[observer_pos, target_pos]`` with NaN on the diagonal and ``observed``
    is the matching boolean mask (``False`` where the observer holds no entry
    and the score silently reads as the neutral prior).
    """
    population = build_from_codes(codes, size)
    outcome = play_generation(
        population,
        interactions=interactions,
        rng=random.Random(seed),
        action_error=action_error,
        observation_error=observation_error,
        observability=observability,
        observability_p=observability_p,
    )
    return outcome.reputation, outcome.observed, outcome.agent_ids


def effective_matrix(reputation: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Mask unseen entries and the diagonal so they render as "no data".

    A score the observer never formed still *reads* as 0.0 inside the game
    (neutral prior), but drawing it as 0.0 would fake perfect agreement with
    every other indifferent observer.
    """
    if reputation.shape != observed.shape:
        raise ValueError("reputation and observed must share a shape")
    masked = np.array(reputation, dtype=float, copy=True)
    masked[~observed] = np.nan
    n = masked.shape[0]
    for i in range(n):
        masked[i, i] = np.nan
    return masked


def row_severity(reputation: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Mean score each observer hands out - its systematic severity offset."""
    masked = effective_matrix(reputation, observed)
    return np.nanmean(masked, axis=1) if masked.size else np.array([])


def pairwise_disagreement_matrix(
    reputation: np.ndarray, observed: np.ndarray,
) -> np.ndarray:
    """``[observer, observer]`` matrix of mean ``|score_i - score_j|``.

    Entry ``(i, j)`` averages over the targets *both* observers rated, which
    is why it is not simply ``|row_mean_i - row_mean_j|``: it is the honest
    per-pair number.  To collapse the whole matrix into one scalar use
    ``core.disagreement``, which is the RMS residual to the nearest consistent
    matrix rather than a mean over these pairs.  The diagonal is NaN.
    """
    masked = effective_matrix(reputation, observed)
    n = masked.shape[0]
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            both = ~np.isnan(masked[i]) & ~np.isnan(masked[j])
            if both.any():
                out[i, j] = float(np.abs(masked[i, both] - masked[j, both]).mean())
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _annotate(
    ax,
    matrix: np.ndarray,
    fmt: str = "{:.2f}",
    threshold: int = ANNOTATE_MAX_AGENTS,
    force: bool = False,
) -> None:
    """Print the value in every cell when the matrix is small enough to read.

    At n=16 a five-column figure gives each cell ~0.14 inch, so printing
    "−1.00" produces an illegible grey smear.  Text is therefore suppressed
    above ``threshold`` unless explicitly forced.
    """
    rows, cols = matrix.shape
    if not force and max(rows, cols) > threshold:
        return
    size = 6.4 if max(rows, cols) <= 12 else 4.6
    for r in range(rows):
        for c in range(cols):
            value = matrix[r, c]
            if np.isnan(value):
                continue
            # Light text on saturated cells so the numbers stay legible.
            ax.text(
                c + 0.5, r + 0.5, fmt.format(value), ha="center", va="center",
                fontsize=size, color="white" if abs(value) > 0.55 else "#222222",
            )


def _mask_cmap(cmap_name: str, bad: str = MISSING_COLOR):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(bad)
    return cmap


def _draw_cells(ax, matrix: np.ndarray, cmap: str, vmin: float, vmax: float):
    """Draw a matrix as crisp cells (pcolormesh + hairline white edges).

    ``imshow`` is deliberately avoided: with adjacent saturated cells the
    boundaries vanish and a row of equal scores becomes one flat block, hiding
    the very structure the figure exists to show.
    """
    rows, cols = matrix.shape
    mesh = ax.pcolormesh(
        np.arange(cols + 1), np.arange(rows + 1),
        np.ma.masked_invalid(matrix), cmap=_mask_cmap(cmap),
        vmin=vmin, vmax=vmax, edgecolors="white", linewidth=0.6,
    )
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)              # observer 0 at the top, like a table
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def _tick_labels(ax, n: int, which: str) -> None:
    """Index ticks centred on each cell."""
    ticks = np.arange(n) + 0.5
    labels = [str(i) for i in range(n)]
    if which == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=6.0)
        ax.tick_params(axis="x", labelsize=6.0, pad=1.5)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=6.0)
        ax.tick_params(axis="y", labelsize=6.0, pad=1.5)


def _draw_column(
    fig, outer, column: int, label: str, reputation: np.ndarray,
    observed: np.ndarray, annotate: bool,
) -> dict[str, object]:
    """One run = reputation matrix (+severity bars) above a disagreement matrix.

    Ticks are shared across columns (labels appear only on the left column and
    the bottom row) so the panels stay clean; the index sets are identical
    everywhere, so nothing is lost.
    """
    masked = effective_matrix(reputation, observed)
    pair = pairwise_disagreement_matrix(reputation, observed)
    severity = row_severity(reputation, observed)
    n = masked.shape[0]

    sub = outer[0, column].subgridspec(1, 2, width_ratios=[1.0, 0.22], wspace=0.04)
    ax_matrix = fig.add_subplot(sub[0])
    ax_sev = fig.add_subplot(sub[1], sharey=ax_matrix)
    ax_pair = fig.add_subplot(outer[1, column], sharex=ax_matrix)

    # --- top: what each observer privately thinks of each target ----------
    _draw_cells(ax_matrix, masked, REPUTATION_CMAP, -1.0, 1.0)
    _annotate(ax_matrix, masked, force=annotate)

    ax_matrix.set_title(
        label, loc="left", fontsize=9.4, fontweight="bold",
        color="#222222", pad=6,
    )

    # Observer severity: the constant offset rank/sign statistics cannot see.
    ax_sev.barh(
        np.arange(n), severity, height=0.78,
        color=[SEVERITY_CMAP(v) if np.isfinite(v) else "#DDDDDD" for v in severity],
        edgecolor="white", linewidth=0.4,
    )
    ax_sev.axvline(0.0, color="#555555", linewidth=0.7, zorder=3)
    ax_sev.set_xlim(-1.05, 1.05)
    ax_sev.set_xticks([-1.0, 0.0, 1.0])
    ax_sev.set_xticklabels(["-1", "0", "1"], fontsize=5.6)
    ax_sev.tick_params(axis="x", length=2, pad=1.0)
    ax_sev.tick_params(axis="y", left=False, labelleft=False)
    for side in ("top", "right", "left"):
        ax_sev.spines[side].set_visible(False)
    ax_sev.spines["bottom"].set_color("#BBBBBB")
    ax_sev.set_title("observer\nmean", fontsize=6.0, color="#666666", pad=5)

    if column == 0:
        _tick_labels(ax_matrix, n, "y")
        ax_matrix.set_ylabel("observer  →", fontsize=7.4, labelpad=3)
    else:
        ax_matrix.tick_params(axis="y", left=False, labelleft=False)

    # --- bottom: how much each pair of observers disagrees -----------------
    _draw_cells(ax_pair, pair, DISAGREEMENT_CMAP, 0.0, PAIR_VMAX)
    _annotate(ax_pair, pair, force=annotate)

    if column == 0:
        ax_pair.set_ylabel("observer  →", fontsize=7.4, labelpad=3)
    else:
        ax_pair.tick_params(axis="y", left=False, labelleft=False)
    _tick_labels(ax_pair, n, "x")
    if column == 0:
        ax_pair.set_xlabel("target / observer index", fontsize=7.4, labelpad=3)
    return {"axes": [ax_matrix, ax_sev, ax_pair], "matrix": ax_matrix, "pair": ax_pair}


def _stat_line(value: float) -> str:
    if not np.isfinite(value):
        return "no shared targets observed"
    return f"D {value:.3f}"


def plot(
    sources: dict[str, Path],
    output: Path,
    *,
    seed: int = 0,
    interactions: int = 1_000,
    condition: str = "control",
    action_error: float = 0.0,
    observation_error: float = 0.0,
    observability: str = "full",
    observability_p: float = 1.0,
    annotate: bool = False,
) -> dict[str, float]:
    """Draw one column per run: reputation matrix above disagreement matrix."""
    labels = list(sources)
    runs = {label: load_run(path, label) for label, path in sources.items()}

    panels = {}
    for label, run in runs.items():
        panels[label] = observer_matrix(
            run.codes, run.size, seed=seed, interactions=interactions,
            action_error=action_error, observation_error=observation_error,
            observability=observability, observability_p=observability_p,
        )

    n_runs = len(labels)
    # Sizing: pick a target matrix edge length in inches and derive the rest.
    # The matrices are drawn with ``aspect="equal"``, so whichever of (column
    # width, row height) is smaller decides the drawn size and the remainder
    # becomes dead space.  The two are therefore matched here on purpose.
    M = 2.42                       # matrix edge, inches
    bars = 0.22                    # observer-mean bar strip, as a fraction of M
    col_w = M * (1 + bars)
    row_gap = 0.30                 # vertical gap between the two rows, inches
    fig_w = col_w * n_runs + 0.98
    fig_h = 2 * M + row_gap + 1.75
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        2, n_runs + 1,
        width_ratios=[1.0] * n_runs + [0.055],
        height_ratios=[1.0, 1.0],
        left=0.050, right=0.947, top=0.860, bottom=0.190,
        wspace=0.075, hspace=row_gap / M,
    )

    drawn = {}
    for column, label in enumerate(labels):
        reputation, observed, _agent_ids = panels[label]
        drawn[label] = _draw_column(
            fig, outer, column, label, reputation, observed, annotate
        )

    # Vertical colour bars on the right: one per row, no wasted full-width rows.
    cax_matrix = fig.add_subplot(outer[0, n_runs])
    cbar_matrix = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0),
            cmap=_mask_cmap(REPUTATION_CMAP),
        ),
        cax=cax_matrix,
    )
    cbar_matrix.set_label(
        "private reputation the observer holds\n"
        "grey = no entry held (self / never observed)",
        fontsize=7.2, labelpad=6,
    )
    cbar_matrix.ax.axhline(0.5, color="#333333", linewidth=1.1)   # GOOD threshold
    cbar_matrix.ax.text(
        2.6, 0.5, "GOOD ≥ 0", transform=cbar_matrix.ax.get_yaxis_transform(),
        fontsize=6.0, color="#333333", va="center",
    )
    cbar_matrix.ax.tick_params(labelsize=6.4, length=2)

    cax_pair = fig.add_subplot(outer[1, n_runs])
    cbar_pair = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=0.0, vmax=PAIR_VMAX),
            cmap=_mask_cmap(DISAGREEMENT_CMAP),
        ),
        cax=cax_pair,
    )
    cbar_pair.set_label(
        "mean |observer difference| on shared targets",
        fontsize=7.2, labelpad=6,
    )
    cbar_pair.ax.tick_params(labelsize=6.4, length=2)

    # Per-column numbers are placed with ``fig.text`` at the centre of each
    # gridspec cell.  Axes-relative offsets would drift, because aspect="equal"
    # shrinks the axes box and leaves the cell partly empty.
    for column, label in enumerate(labels):
        cell = drawn[label]["pair"].get_position()
        fig.text(
            cell.x0 + cell.width / 2, 0.088,
            _stat_line(disagreement(panels[label][0], panels[label][1])),
            fontsize=6.9, color="#333333", va="top", ha="center",
            linespacing=1.55, family="monospace",
        )

    condition_text = condition
    if action_error:
        condition_text += f", action_err={action_error:g}"
    if observation_error:
        condition_text += f", obs_err={observation_error:g}"
    if observability != "full":
        condition_text += f", observability={observability} p={observability_p:g}"
    fig.suptitle(
        "Private reputation matrices of LLM-evolved populations",
        fontsize=13.0, fontweight="bold", y=0.982,
    )
    fig.text(
        0.5, 0.933,
        f"{condition_text}   ·   seed {seed}   ·   {interactions} interactions   ·   "
        "top: each observer's private opinion   ·   bottom: pairwise disagreement",
        fontsize=8.0, color="#555555", ha="center",
    )
    fig.savefig(output, bbox_inches="tight", dpi=220)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return {
        label: disagreement(panels[label][0], panels[label][1]) for label in labels
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", action="append", default=[], metavar="LABEL=PATH",
                        help="Repeatable evolutionary.json source (one figure column each).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interactions", type=int, default=1_000)
    parser.add_argument("--condition", default="control",
                        help="Label only; the error knobs below set the actual condition.")
    parser.add_argument("--action-error", type=float, default=0.0)
    parser.add_argument("--observation-error", type=float, default=0.0)
    parser.add_argument("--observability", choices=("full", "partial", "private"), default="full")
    parser.add_argument("--observability-p", type=float, default=1.0)
    parser.add_argument("--annotate", action="store_true",
                        help="Force the value into every cell (unreadable above ~12 agents).")
    args = parser.parse_args()

    if not args.source:
        parser.error("at least one --source LABEL=PATH is required")
    sources: dict[str, Path] = {}
    for value in args.source:
        if "=" not in value:
            parser.error(f"Invalid --source {value!r}; expected LABEL=PATH")
        label, raw = value.split("=", 1)
        if not label or label in sources:
            parser.error(f"Source labels must be non-empty and unique: {label!r}")
        sources[label] = Path(raw).resolve()

    stats = plot(
        sources, args.output,
        seed=args.seed, interactions=args.interactions, condition=args.condition,
        action_error=args.action_error, observation_error=args.observation_error,
        observability=args.observability, observability_p=args.observability_p,
        annotate=args.annotate,
    )
    print(f"wrote {args.output} and {args.output.with_suffix('.pdf')}")
    for label, value in stats.items():
        print(f"  {label:10s} D={value:.4f}" if np.isfinite(value)
              else f"  {label:10s} D=NaN (no shared targets)")


if __name__ == "__main__":
    main()
