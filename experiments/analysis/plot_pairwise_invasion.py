"""Plot pairwise invasion results (strategy vs strategy).

Reads the ``summary.json`` written by ``run_invasion --residents`` and produces:
  1. Response curves for selected unordered pairs, with the no-change diagonal.
  2. A dominance-threshold matrix (row invades column).

Every curve plotted here is a stored simulation result. Nothing is derived or
mirrored: a direction that was not simulated is simply absent from the figure.

A pair is "asymmetric" when one direction climbs well above the diagonal while
the other stays below it; that is the signature of strict pairwise dominance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import project_root


ROOT = project_root()
COLORS = ("#2878B5", "#D1495B", "#2A9D8F", "#E9C46A",
          "#7B2CBF", "#F77F00", "#5F6F52", "#6C757D")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")


def _has_pair(summary, inv, res):
    return inv in summary.get("groups", {}) and res in summary["groups"].get(inv, {})


def strategy_labels(summary: dict) -> list[str]:
    """Every strategy label in the summary.

    Archived pairwise summaries recorded a ``strategies`` block; the unified
    runner records ``sources`` (candidates) plus ``residents``. Both are needed:
    ``groups`` is keyed by the invader only, so a resident that never invades
    would be missing from the ``groups`` fallback.
    """
    if "strategies" in summary:
        return list(summary["strategies"])
    labels = list(summary.get("sources", {}))
    for label in summary.get("residents", {}):
        if label not in labels:
            labels.append(label)
    return labels or list(summary.get("groups", {}))


def _curve(summary, inv, res, counts):
    cells = summary["groups"][inv][res]
    return np.asarray([cells[str(c)]["mean_final_invader_frequency"] for c in counts])


def _threshold(summary, inv, res, counts, level=0.5):
    if not _has_pair(summary, inv, res):
        return None
    for c in counts:
        if summary["groups"][inv][res][str(c)]["mean_final_invader_frequency"] >= level:
            return c
    return None


def share_label(summary: dict) -> str:
    """Y-axis label, with the seed count read from the summary.

    The count must not be hard-coded: a summary produced with a different
    ``--seeds`` would otherwise be mislabelled (the earlier version always said
    "3 seeds"). Summaries without a ``seeds`` block fall back to a bare label.
    """
    n_seeds = len(summary.get("seeds") or [])
    suffix = f" ({n_seeds} seed{'s' if n_seeds != 1 else ''})" if n_seeds else ""
    return f"Mean final invader share{suffix}"


def plot(summary_path: Path, output_path: Path, pairs: list[tuple[str, str]]) -> None:
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    counts = [int(c) for c in s["initial_invader_counts"]]
    x = np.asarray(counts, dtype=float) / 100.0
    labels = strategy_labels(s)
    ae = float(s.get("action_error_probability", 0.0))
    oe = float(s.get("observation_error_probability", 0.0))
    ylabel = share_label(s)

    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.9 * nrows),
                             squeeze=False)
    for idx, (a, b) in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        ax.plot([0, 1], [0, 1], "--", color="#222222", linewidth=1.2,
                label="No frequency change")
        j = 0
        for inv, res in ((a, b), (b, a)):
            if not _has_pair(s, inv, res):
                continue
            ys = _curve(s, inv, res, counts)
            thr = _threshold(s, inv, res, counts)
            lbl = f"{inv} invades {res}" + (
                f" (thr n={thr})" if thr else " (no thr)"
            )
            ax.plot(x, ys, color=COLORS[j], marker=MARKERS[j], markersize=5,
                    linewidth=1.9, label=lbl)
            j += 1
        ax.set_title(f"{a} vs {b}: invasion response", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6),
                      [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
        ax.set_yticks(np.linspace(0, 1, 6),
                      [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
        ax.set_xlabel("Initial invader share")
        ax.set_ylabel(ylabel)
        ax.grid(color="#D9D9D9", linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    for idx in range(len(pairs), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(
        f"Pairwise invasion with {ae:.0%} action + {oe:.0%} observation error",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.02, 0.03, 0.99, 0.94), h_pad=2.0, w_pad=1.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---- dominance threshold matrix --------------------------------------
    n = len(labels)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if a == b:
                continue
            t = _threshold(s, a, b, counts) if _has_pair(s, a, b) else None
            M[i, j] = 100 if t is None else t
    fig2, ax2 = plt.subplots(figsize=(1.5 + 1.15 * n, 1.4 + 1.0 * n))
    im = ax2.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=100)
    ax2.set_xticks(range(n), labels, fontsize=10)
    ax2.set_yticks(range(n), labels, fontsize=10)
    ax2.set_xlabel("resident (defender)")
    ax2.set_ylabel("invader")
    for i in range(n):
        for j in range(n):
            if i == j:
                ax2.text(j, i, "—", ha="center", va="center", fontsize=13)
                continue
            v = M[i, j]
            txt = "∞" if v >= 100 else f"{int(v)}"
            ax2.text(j, i, txt, ha="center", va="center", fontsize=12,
                     color="white" if v >= 70 else "black")
    ax2.set_title("Initial share needed to invade (threshold at 50%)\n"
                  "green/low = easy to invade", fontsize=12)
    fig2.colorbar(im, ax=ax2, label="initial invader share (%)", shrink=0.8)
    fig2.tight_layout()
    out2 = output_path.with_name(output_path.stem + "_threshold_matrix.png")
    fig2.savefig(out2, bbox_inches="tight", dpi=200)
    fig2.savefig(out2.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {output_path}")
    print(f"Wrote {out2}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pairs", nargs="*", default=[], metavar="A>B",
        help="Ordered pairs to plot; defaults to all unordered pairs.",
    )
    args = parser.parse_args()
    s = json.loads(args.summary.resolve().read_text(encoding="utf-8"))
    labels = strategy_labels(s)
    if args.pairs:
        pairs = [tuple(p.split(">", 1)) for p in args.pairs]
    else:
        pairs = [(labels[i], labels[j])
                 for i in range(len(labels)) for j in range(i + 1, len(labels))]
    plot(args.summary.resolve(), args.output.resolve(), pairs)


if __name__ == "__main__":
    main()
