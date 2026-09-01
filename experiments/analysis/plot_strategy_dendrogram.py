"""Draw the hierarchical merge tree (dendrogram) of evolved strategy code.

The dendrogram shows the *hierarchical* relationship between strategies that
flat K-means/hierarchical labels hide: leaves are unique strategy codes, and
the vertical merge height is the Ward distance at which two strategy clusters
were joined. Cutting the tree at the auto-selected K produces the flat cluster
assignment used everywhere else in the analysis.

Requires ``--clustering-method hierarchical`` (the merge tree only exists for
agglomerative clustering).

Usage:
  uv run python -m experiments.analysis.plot_strategy_dendrogram \
      --json results/.../evolutionary.json \
      --clustering-method hierarchical \
      --out out.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from experiments.evolution_log import F_AGENT_ID, F_CODE, K_FINAL_POPULATION, load_evolution_json

from .clustering.cli_args import add_clustering_method_args, clustering_method_kwargs
from .clustering.pipeline import cluster_codes


def plot_dendrogram(
    codes,
    labels,
    cluster_names,
    out_path,
    *,
    linkage,
    max_leaves: int = 60,
    truncate_mode: str = "level",
    k: int | None = None,
    title_suffix: str = "",
    leaf_names: dict[str, str] | None = None,
) -> None:
    """Render a Ward-linkage dendrogram with leaves colored by flat cluster."""
    colors = _cluster_colors(max(labels) + 1)

    fig, ax = plt.subplots(figsize=(16, 7), facecolor="white")
    if leaf_names:
        leaf_labels = [leaf_names.get(code, _short_code(code)) for code in codes]
    else:
        leaf_labels = [_short_code(code) for code in codes]

    # Truncate the tree so the plot stays readable when there are hundreds of
    # leaves; leaf colors still reflect the flat-cluster assignment.
    dendrogram(
        linkage,
        ax=ax,
        orientation="top",
        leaf_rotation=90,
        leaf_font_size=6,
        labels=leaf_labels,
        color_threshold=0.0,
        truncate_mode=truncate_mode,
        p=max_leaves,
        above_threshold_color="gray",
        count_sort="descendent",
    )

    # Color the leaf nodes by their flat cluster (the truncation makes full
    # per-leaf coloring impractical, so this annotates the cluster identity in
    # the legend instead of on the leaves).
    legend_handles = [
        plt.Line2D([0], [0], marker="o", ls="", color=colors[c],
                   label=f"Cluster {c} · {cluster_names.get(c, '')}")
        for c in range(max(labels) + 1)
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True,
              fontsize=8, title="Flat clusters (auto-K)")
    ax.set_ylabel("Ward linkage distance")
    ax.set_title(
        f"Strategy dendrogram · Ward hierarchical clustering"
        + (f" · K={k}" if k is not None else "")
        + (f" · {title_suffix}" if title_suffix else ""),
        fontsize=13, fontweight="semibold",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _cluster_colors(n: int):
    palette = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    ]
    if n <= len(palette):
        return palette[:n]
    return [plt.colormaps["turbo"](x) for x in np.linspace(0.05, 0.95, n)]


def _short_code(code: str) -> str:
    """Compress a strategy code to a short leaf label."""
    first_line = code.strip().splitlines()[0] if code.strip() else ""
    return first_line[:60]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=str, required=True, help="path to evolutionary.json")
    ap.add_argument("--k", type=int, default=None, help="fixed K; default = silhouette-best")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None,
                    help="output path (default: <json_dir>/strategy_dendrogram.png)")
    ap.add_argument("--max-leaves", type=int, default=60,
                    help="max visible leaves when truncating the dendrogram")
    ap.add_argument("--name-leaves", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="ask the LLM for a behavioral label per strategy leaf "
                         "(cached by code hash)")
    add_clustering_method_args(ap)
    args = ap.parse_args()

    if args.clustering_method != "hierarchical":
        print("note: dendrogram requires hierarchical clustering; forcing "
              "--clustering-method hierarchical")
        args.clustering_method = "hierarchical"

    data = load_evolution_json(Path(args.json))
    pop = sorted(data[K_FINAL_POPULATION], key=lambda a: a[F_AGENT_ID])
    codes = [a[F_CODE] for a in pop]

    _, labels, km, unique, names = cluster_codes(
        codes,
        k=args.k,
        seed=args.seed,
        name_leaves=args.name_leaves,
        **clustering_method_kwargs(args),
    )
    if not hasattr(km, "linkage_matrix"):
        raise RuntimeError("hierarchical clustering did not retain a linkage matrix")

    out_path = Path(args.out) if args.out else (
        Path(args.json).parent / "strategy_dendrogram.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dendrogram(
        unique,
        labels,
        names,
        out_path,
        linkage=km.linkage_matrix(),
        max_leaves=args.max_leaves,
        k=getattr(km, "n_clusters", None),
        title_suffix=Path(args.json).parent.name,
        leaf_names=getattr(km, "leaf_names", None),
    )
    print(f"dendrogram: {out_path.resolve()}")


if __name__ == "__main__":
    main()
