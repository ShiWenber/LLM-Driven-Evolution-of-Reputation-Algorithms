"""Plot final-generation strategy clusters in a 2-D projection.

Clustering itself lives in the reusable ``clustering.pipeline`` module (deep
in the analysis package); this script only draws the figure and exposes the
CLI.

The scatter shows one point per *unique* strategy code; points are colored by
their selected-embedding + K-means cluster and annotated with agent id(s) that share
that exact strategy string (a bracketed list when several agents share it).

Usage:
  uv run python -m experiments.analysis.plot_strategy_clusters --json results/.../evolutionary.json --out out.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.evolution_log import (
    F_AGENT_ID, F_CODE, K_FINAL_POPULATION, load_evolution_json,
)

from .clustering.cli_args import add_clustering_method_args, clustering_method_kwargs
from .clustering.cache import AnalysisCache
from .clustering.pipeline import (
    DEFAULT_CODE_EMBEDDING_MODEL,
    DEFAULT_CODE_EMBEDDING_REVISION,
    cluster_codes,
    project_embeddings,
)


PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]


def _colors(n: int):
    if n <= len(PALETTE):
        return PALETTE[:n]
    return [plt.colormaps["turbo"](x) for x in np.linspace(0.05, 0.95, n)]


def plot_clusters(
    codes,
    agent_ids,
    k,
    out_path,
    seed: int = 42,
    *,
    code_embedding_model: str = DEFAULT_CODE_EMBEDDING_MODEL,
    code_embedding_revision: str = DEFAULT_CODE_EMBEDDING_REVISION,
    embedding_device: str = "auto",
    embedding_batch_size: int | None = None,
    embedding_cache: bool = True,
    embedding_cache_path: str | Path | None = None,
    llm_model: str | None = None,
    refresh_cluster_names: bool = False,
    analysis_source_path: str | Path | None = None,
):
    """2-D projection scatter of the clustered unique codes.

    Code embeddings use centered PCA. One
    point is drawn per unique strategy and annotated with the agent id(s) that
    share that exact strategy string.
    """
    X, labels, km, unique, names = cluster_codes(
        codes,
        k=k,
        seed=seed,
        code_embedding_model=code_embedding_model,
        code_embedding_revision=code_embedding_revision,
        embedding_device=embedding_device,
        embedding_batch_size=embedding_batch_size,
        embedding_cache=embedding_cache,
        embedding_cache_path=embedding_cache_path,
        llm_model=llm_model,
        refresh_cluster_names=refresh_cluster_names,
        analysis_source_path=analysis_source_path,
    )
    Z, reducer, projection_label = project_embeddings(X, seed=seed)

    code_to_ids = {}
    for code, aid in zip(codes, agent_ids):
        code_to_ids.setdefault(code, []).append(aid)

    n_clusters = len(set(labels))
    colors = _colors(n_clusters)
    fig, ax = plt.subplots(figsize=(10.5, 6.5), facecolor="white")
    for i in range(len(set(labels))):
        m = labels == i
        ax.scatter(Z[m, 0], Z[m, 1], s=105, alpha=0.92, color=colors[i],
                   edgecolor="white", linewidth=0.9, zorder=3,
                   label=f"Cluster {i} · {names[i]}")

    for point_idx, code in enumerate(unique):
        x, y = Z[point_idx]
        ids = code_to_ids.get(code, [])
        if len(ids) == 1:
            text = str(ids[0])
        else:
            text = f"[{', '.join(map(str, ids))}]"
        ax.annotate(text, (x, y), xytext=(7, 5), textcoords="offset points",
                    fontsize=7.5, color="#253044", fontweight="medium",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.8}, zorder=4)

    ax.set_axisbelow(True)
    ax.grid(color="#E3E7ED", linewidth=0.75, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAB2BF")
    ax.tick_params(colors="#4B5563", labelsize=9)
    ax.set_xlabel(f"{projection_label} component 1 ({reducer.explained_variance_ratio_[0]:.0%})",
                  fontsize=10, color="#344054", labelpad=8)
    ax.set_ylabel(f"{projection_label} component 2 ({reducer.explained_variance_ratio_[1]:.0%})",
                  fontsize=10, color="#344054", labelpad=8)
    ax.set_title("Final-generation strategy landscape", loc="left", fontsize=16,
                 fontweight="semibold", color="#172033", pad=14)
    ax.text(0, 1.01, f"Code embedding + K-means · {len(unique)} unique strategies · K = {n_clusters}",
            transform=ax.transAxes, fontsize=9.5, color="#667085", va="bottom")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8,
              frameon=False, title="Strategy clusters", title_fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.76, 1))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    AnalysisCache(embedding_cache_path).put_artifact(
        run_id=getattr(km, "analysis_run_id", None),
        artifact_type="final_strategy_clusters", path=out_path,
        metadata={"dpi": 220, "projection": projection_label},
    )
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=str, required=True, help="path to evolutionary.json")
    ap.add_argument("--k", type=int, default=None, help="fixed K; default = silhouette-best")
    ap.add_argument("--seed", type=int, default=42)
    add_clustering_method_args(ap)
    ap.add_argument("--out", type=str, default="strategy_clusters_pca.png",
                    help="output image path")
    args = ap.parse_args()

    data = load_evolution_json(Path(args.json))
    pop = sorted(data[K_FINAL_POPULATION], key=lambda a: a[F_AGENT_ID])
    codes = [a[F_CODE] for a in pop]
    print(f"final population: {len(pop)} agents, unique code: {len(set(codes))}")

    plot_clusters(
        codes,
        [a[F_AGENT_ID] for a in pop],
        args.k,
        Path(args.out),
        seed=args.seed,
        **clustering_method_kwargs(args),
    )


if __name__ == "__main__":
    main()
