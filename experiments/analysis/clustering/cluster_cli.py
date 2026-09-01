"""Cluster final-population strategy codes by selectable embedding + K-means.

This module is a thin CLI facade over the reusable primitives in
``clustering.pipeline.cluster_codes``: it loads a
final population, prints the per-agent cluster table and cluster sizes.

The 2-D projection figure is drawn by
``experiments.analysis.plot_strategy_clusters``.

Usage:
  uv run python -m experiments.analysis.clustering.cluster_cli --json results/.../evolutionary.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from experiments.evolution_log import (
    F_AGENT_ID, F_CODE, F_FITNESS, K_FINAL_POPULATION, load_evolution_json,
)

from .cli_args import add_clustering_method_args, clustering_method_kwargs
from .pipeline import cluster_codes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=str, required=True, help="path to evolutionary.json")
    ap.add_argument("--k", type=int, default=None, help="fixed K; default = silhouette-best")
    ap.add_argument("--seed", type=int, default=42)
    add_clustering_method_args(ap)
    args = ap.parse_args()

    data = load_evolution_json(Path(args.json))
    pop = sorted(data[K_FINAL_POPULATION], key=lambda a: a[F_AGENT_ID])
    codes = [a[F_CODE] for a in pop]
    print(f"final population: {len(pop)} agents, unique code: {len(set(codes))}")

    _, labels, _, unique, names = cluster_codes(
        codes, k=args.k, seed=args.seed, **clustering_method_kwargs(args)
    )

    # map each unique code -> its cluster, then back to every agent
    code_to_cluster = {c: int(l) for c, l in zip(unique, labels)}
    agent_cluster = {a[F_AGENT_ID]: code_to_cluster[a[F_CODE]] for a in pop}
    print("\n=== per-agent cluster table ===")
    for a in pop:
        c = agent_cluster[a[F_AGENT_ID]]
        print(f"  agent {a[F_AGENT_ID]:>3}: cluster={c} "
              f"({names[c]}) fit={a[F_FITNESS]:>4}")

    print("\n=== cluster sizes ===")
    for c, n in sorted(Counter(agent_cluster.values()).items()):
        print(f"  cluster {c} ({names[c]}): {n} agents")


if __name__ == "__main__":
    main()
