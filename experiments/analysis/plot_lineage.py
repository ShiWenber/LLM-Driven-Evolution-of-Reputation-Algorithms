"""Visualize the evolutionary tree from framework-recorded lineage data.

Produces two views (both read an evolution-log record; see
``experiments.evolution_log`` for the schema):

  1. Lineage survival plot — x = generation, y = collapsed lineage (root), a
     horizontal bar spans the lineage's [birth_gen, death_gen] lifetime,
     colored by the lineage's strategy cluster. Shows which
     lineages survived to the end, which went extinct, and when independent
     rewrites injected fresh lineages.

  2. Lineage tree — a phylogenetic forest (edges = parent_lineage_id ->
     lineage_id), laid out with birth generation on the x-axis and DFS leaf
     order on the y-axis. By default ALL branches (including extinct ones)
     are drawn; pass ``--survivors-only`` to restrict the tree to the
     ancestry paths of final-generation agents only.

Usage:
  uv run python -m experiments.analysis.plot_lineage --json results/.../evolutionary.json
  uv run python -m experiments.analysis.plot_lineage --json results/.../evolutionary.json --survivors-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.evolution_log import (
    F_CODE,
    F_CONFIG_SCHEMA_VERSION,
    F_LINEAGE_ID,
    F_POPULATION,
    K_FINAL_POPULATION,
    K_TRAJECTORY,
    load_evolution_json,
)

from .clustering.cache import AnalysisCache
from .clustering.cli_args import add_clustering_method_args, clustering_method_kwargs
from .clustering.pipeline import (
    DEFAULT_CODE_EMBEDDING_MODEL,
    DEFAULT_CODE_EMBEDDING_REVISION,
    cluster_codes,
)
from .lineage.build import build_lineage_tree


def _cluster_color_lookup(name_map: dict[int, str]) -> dict[int, tuple]:
    """Build a stable cluster->color mapping based on cluster count.

    - k <= 10: use a highly distinguishable discrete palette (tab10)
    - k > 10: sample a continuous palette (turbo) so each cluster gets
      a unique color across the full automatic K-selection range.
    """
    cluster_ids = sorted(name_map)
    n = len(cluster_ids)
    if n == 0:
        return {}

    if n <= 10:
        cmap = plt.get_cmap("tab10", n)
        return {cluster_id: cmap(i) for i, cluster_id in enumerate(cluster_ids)}

    cmap = plt.get_cmap("turbo")
    if n == 1:
        return {cluster_ids[0]: cmap(0.5)}
    return {
        cluster_id: cmap(i / (n - 1))
        for i, cluster_id in enumerate(cluster_ids)
    }


def _lineage_clusters(
    data: dict,
    *,
    clustering_method: str = "kmeans",
    code_embedding_model: str = DEFAULT_CODE_EMBEDDING_MODEL,
    code_embedding_revision: str = DEFAULT_CODE_EMBEDDING_REVISION,
    embedding_device: str = "auto",
    embedding_batch_size: int | None = None,
    embedding_cache: bool = True,
    embedding_cache_path: str | Path | None = None,
    llm_model: str | None = None,
    refresh_cluster_names: bool = False,
    analysis_source_path: str | Path | None = None,
) -> tuple[dict[int, int], dict[int, str], str, str | None]:
    """One clustering of all lineage representative codes.

    Returns (cluster_by_lineage_id, name_by_cluster_id). In Fermi mode a
    lineage's code is fixed for its lifetime (it only changes when the slot
    is re-instantiated, which creates a new lineage_id), so the first code
    observed for a lineage_id is representative.
    """
    code_of: dict[int, str] = {}
    for g in data.get(K_TRAJECTORY, []):
        for a in g.get(F_POPULATION, []):
            lid = a.get(F_LINEAGE_ID)
            if lid is not None and lid not in code_of:
                code_of[lid] = a.get(F_CODE, "")
    for a in data.get(K_FINAL_POPULATION, []):
        lid = a.get(F_LINEAGE_ID)
        if lid is not None and lid not in code_of:
            code_of[lid] = a.get(F_CODE, "")

    if not code_of:
        return {}, {}, "Code embedding", None

    codes = list(code_of.values())
    _, labels, km, unique, names = cluster_codes(
        codes,
        clustering_method=clustering_method,
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
    code_to_cluster = {c: int(l) for c, l in zip(unique, labels)}
    cluster_by_lid = {lid: code_to_cluster[code] for lid, code in code_of.items()}
    return cluster_by_lid, names, "Code embedding", getattr(km, "analysis_run_id", None)


def lineage_survival_plot(data: dict, out_path: Path, *, clusters=None):
    tree = build_lineage_tree(data)
    clus_by_lid, name_map, embedding_label = (clusters or _lineage_clusters(data))[:3]
    color_map = _cluster_color_lookup(name_map)
    lineages = tree["lineages"]

    # order lineages by birth_gen, then root id
    roots = sorted(lineages.values(), key=lambda l: (l["birth_gen"], l["root_lineage_id"]))

    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.18 * len(roots))))
    yticks, ylabels = [], []
    for i, lin in enumerate(roots):
        y = i
        c = clus_by_lid.get(lin["root_lineage_id"], -1)
        color = color_map.get(c, "#7f7f7f") if c >= 0 else "#7f7f7f"
        ax.hlines(y, lin["birth_gen"], lin["death_gen"], color=color, lw=3.5, alpha=0.85)
        # mark independent_init roots with a hollow marker
        if lin["origin"] == "independent_init":
            ax.plot(lin["birth_gen"], y, "o", ms=6, mfc="none", mec=color, mew=1.5)
        else:
            ax.plot(lin["birth_gen"], y, "o", ms=5, color=color)
        yticks.append(y)
        ylabels.append(f"L{lin['root_lineage_id']}")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("generation")
    ax.set_ylabel("lineage (root)")
    ax.set_title(
        f"lineage survival (collapsed; bar = birth→death, color = {embedding_label} cluster)"
    )
    ax.set_xlim(-0.5, max(l["death_gen"] for l in roots) + 0.5)

    # legend for cluster colors
    handles = [
        plt.Line2D([0], [0], color=color_map.get(c, "#7f7f7f"), lw=3,
                   label=f"cluster {c}: {name_map.get(c, '')}")
        for c in sorted(name_map)
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def _layout_tree(tree: dict) -> dict[int, tuple]:
    """Assign (x, y) to each node in the ancestor forest.

    x = birth_gen (time), y = DFS leaf order (so leaves are spread out).
    Returns {lineage_id: (x, y)}.
    """
    events = {e["lineage_id"]: e for e in tree["events"]}
    parent_of = {int(k): v for k, v in tree["parent_of"].items()}
    # children map (parent -> [child lineage_id])
    children: dict[int, list] = {}
    for lid, parent in parent_of.items():
        if parent is not None:
            children.setdefault(parent, []).append(lid)

    # roots = lineage ids with no parent
    roots = sorted([lid for lid, p in parent_of.items() if p is None])

    pos: dict[int, tuple] = {}
    leaf_counter = 0

    def dfs(lid, depth_budget):
        nonlocal leaf_counter
        birth = events[lid]["birth_gen"]
        kids = children.get(lid, [])
        if not kids:
            # leaf: assign next y slot
            y = leaf_counter
            leaf_counter += 1
            pos[lid] = (birth, y)
            return y
        child_ys = [dfs(c, depth_budget - 1) for c in kids]
        # internal node at mean y of children
        y = sum(child_ys) / len(child_ys)
        pos[lid] = (birth, y)
        return y

    for r in roots:
        dfs(r, depth_budget=200)
    return pos


def lineage_backtrack_tree(
    data: dict,
    out_path: Path,
    *,
    clusters=None,
    survivors_only: bool = False,
):
    """Draw the lineage tree.

    ``survivors_only=False`` (default): draw ALL lineages in the forest,
    extinct branches included. ``survivors_only=True``: restrict to the
    ancestry paths of the final-generation agents only.
    """
    tree = build_lineage_tree(data)
    clus_by_lid, name_map, embedding_label = (clusters or _lineage_clusters(data))[:3]
    color_map = _cluster_color_lookup(name_map)
    parent_of = {int(k): v for k, v in tree["parent_of"].items()}

    if survivors_only:
        # collect all nodes on survivor ancestry paths
        nodes = set()
        for s in tree["survivors"]:
            nodes.update(s["path"])
        title = (
            "final-survivor ancestry tree "
            "(leaf = final agent, color = " + embedding_label + " cluster)"
        )
    else:
        # the full forest: every lineage ever recorded (extinct included)
        nodes = set(parent_of.keys())
        title = (
            "lineage tree (all branches, "
            "color = " + embedding_label + " cluster)"
        )

    # assign positions across the full forest (so extinct branches align too)
    pos = _layout_tree(tree)

    fig, ax = plt.subplots(figsize=(13, max(4.5, 0.28 * len(nodes))))
    # edges
    for lid in nodes:
        parent = parent_of.get(lid)
        if parent is not None and parent in nodes:
            x0, y0 = pos[parent]
            x1, y1 = pos[lid]
            ax.plot([x0, x1], [y0, y1], "-", color="#bbbbbb", lw=1.0, zorder=1)

    # nodes
    for lid in nodes:
        x, y = pos[lid]
        c = clus_by_lid.get(lid, -1)
        color = color_map.get(c, "#7f7f7f") if c >= 0 else "#7f7f7f"
        is_survivor = any(s["lineage_id"] == lid for s in tree["survivors"])
        is_root = parent_of.get(lid) is None
        if is_root:
            ax.plot(x, y, "s", ms=9, color=color, zorder=3)
        elif is_survivor:
            ax.plot(x, y, "^", ms=10, color=color, mec="black", mew=0.8, zorder=3)
        else:
            ax.plot(x, y, "o", ms=6, color=color, alpha=0.8, zorder=2)

    ax.set_xlabel("birth generation")
    ax.set_ylabel("lineage (DFS order)")
    ax.set_title(title)
    ax.set_yticks([])

    handles = [
        plt.Line2D([0], [0], marker="o", ls="", color=color_map.get(c, "#7f7f7f"),
                   label=f"cluster {c}: {name_map.get(c, '')}")
        for c in sorted(name_map)
    ]
    handles += [
        plt.Line2D([0], [0], marker="s", ls="", color="k", label="root"),
        plt.Line2D([0], [0], marker="^", ls="", color="k", mec="k", mfc="none", label="final survivor"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=str, required=True, help="path to schema-v4 evolutionary.json")
    ap.add_argument(
        "--survivors-only",
        action="store_true",
        help="only draw the ancestry tree of final-generation survivors "
        "(default: all branches, extinct lineages included)",
    )
    add_clustering_method_args(ap)
    ap.add_argument("--cluster-run-id", type=str, default=None,
                    help="reuse labels and names from a stored clustering run")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="suffix added to output filenames before .png")
    args = ap.parse_args()

    json_path = Path(args.json)
    data = load_evolution_json(json_path)
    if data.get("config", {}).get(F_CONFIG_SCHEMA_VERSION, 0) < 4:
        raise SystemExit("requires schema >= 4 (re-run evolution with updated framework)")

    out_dir = json_path.parent
    if args.cluster_run_id:
        code_labels, names = AnalysisCache(args.embedding_cache_path).get_clustering_run_labels(
            args.cluster_run_id
        )
        code_by_lineage = {}
        for generation in data.get(K_TRAJECTORY, []):
            for agent in generation.get(F_POPULATION, []):
                lineage_id = agent.get(F_LINEAGE_ID)
                if lineage_id is not None and lineage_id not in code_by_lineage:
                    code_by_lineage[lineage_id] = agent.get(F_CODE, "")
        clusters = (
            {lineage_id: code_labels[code] for lineage_id, code in code_by_lineage.items()
             if code in code_labels},
            names,
            "stored Code embedding",
            args.cluster_run_id,
        )
    else:
        clusters = _lineage_clusters(data, **clustering_method_kwargs(args))
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    survival_path = out_dir / f"lineage_survival{suffix}.png"
    lineage_survival_plot(data, survival_path, clusters=clusters)
    lineage_backtrack_tree(
        data,
        out_dir / f"lineage_tree{suffix}.png",
        clusters=clusters,
        survivors_only=args.survivors_only,
    )
    cache = AnalysisCache(args.embedding_cache_path)
    cache.put_artifact(run_id=clusters[3], artifact_type="lineage_survival",
                       path=survival_path, metadata={"format": "png"})
    cache.put_artifact(run_id=clusters[3], artifact_type="lineage_tree",
                       path=out_dir / f"lineage_tree{suffix}.png",
                       metadata={"format": "png", "survivors_only": args.survivors_only})


if __name__ == "__main__":
    main()
