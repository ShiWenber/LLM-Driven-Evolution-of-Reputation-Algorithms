"""Globally cluster multiple experiments and draw directly comparable views."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.evolution_log import F_POPULATION, load_evolution_json

from .clustering.cache import AnalysisCache
from .clustering.cli_args import add_clustering_method_args, clustering_method_kwargs
from .clustering.io import load_generations
from .clustering.pipeline import cluster_strategies
from .plot_strategy_cluster_evolution import _colors, _style_axes, plot_cluster_composition


def _latest_default_logs(root: Path) -> list[Path]:
    selected = []
    for version in ("LLM_v2", "LLM_v3"):
        candidates = sorted(
            root.glob(f"{version}_*/evolutionary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if len(candidates) < 3:
            raise FileNotFoundError(f"Need three {version} logs under {root}")
        selected.extend(candidates[:3])
    return selected


def _experiment_label(path: Path) -> str:
    name = path.parent.name
    version = "LLM_v2" if "LLM_v2" in name else "LLM_v3"
    seed = name.rsplit("seed", 1)[-1] if "seed" in name else name
    return f"{version} · seed {seed}"


def _slice_state(state: dict, generation_indices: list[int]) -> dict:
    global_rows = [row for gi in generation_indices for row in state["gen_rows"][gi]]
    result = dict(state)
    result["X"] = state["X"][global_rows]
    result["Z"] = state["Z"][global_rows]
    result["labels"] = state["labels"][global_rows]
    result["gen_rows"] = []
    offset = 0
    for gi in generation_indices:
        size = len(state["gen_rows"][gi])
        result["gen_rows"].append(list(range(offset, offset + size)))
        offset += size
    return result


def plot_comparison(experiments: list[dict], state: dict, out_path: Path) -> None:
    k = state["km"].n_clusters
    colors = _colors(k)
    ncols = min(3, len(experiments))
    nrows = math.ceil(len(experiments) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, experiment in zip(axes.flat, experiments):
        generations = experiment["generations"]
        substate = _slice_state(state, experiment["generation_indices"])
        counts = np.zeros((len(generations), k), dtype=int)
        for gi, rows in enumerate(substate["gen_rows"]):
            for cluster_id, count in Counter(map(int, substate["labels"][rows])).items():
                counts[gi, cluster_id] = count
        x = [generation["generation"] for generation in generations]
        bottom = np.zeros(len(x))
        for cluster_id in range(k):
            ax.bar(x, counts[:, cluster_id], bottom=bottom, width=0.9,
                   color=colors[cluster_id], edgecolor="white", linewidth=0.2)
            bottom += counts[:, cluster_id]
        _style_axes(ax)
        ax.set_title(experiment["label"], loc="left", fontsize=11, fontweight="semibold")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Agents")
    for ax in axes.flat[len(experiments):]:
        ax.set_visible(False)
    handles = [
        plt.Line2D([0], [0], color=colors[c], lw=7,
                   label=f"{c} · {state['cluster_names'].get(c, '')}")
        for c in range(k)
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.88, 0.5),
               frameon=False, title="Global clusters")
    fig.suptitle(
        f"Cross-experiment strategy-cluster composition · {state['embedding_label']} · K={k}",
        fontsize=16, fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 0.87, 0.95))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_strategy_space(experiments: list[dict], state: dict, out_path: Path) -> None:
    colors = _colors(state["km"].n_clusters)
    versions = [version for version in ("LLM_v2", "LLM_v3")
                if any(version in experiment["label"] for experiment in experiments)]
    fig, axes = plt.subplots(1, len(versions), figsize=(7.5 * len(versions), 6.5),
                             sharex=True, sharey=True, squeeze=False)
    for ax, version in zip(axes.flat, versions):
        seen = set()
        for experiment in experiments:
            if version not in experiment["label"]:
                continue
            for row in [r for gi in experiment["generation_indices"] for r in state["gen_rows"][gi]]:
                code = state["all_codes"][row]
                if code in seen:
                    continue
                seen.add(code)
                cluster_id = int(state["labels"][row])
                ax.scatter(*state["Z"][row], s=12, color=colors[cluster_id], alpha=0.55,
                           linewidths=0)
        _style_axes(ax)
        ax.set_title(f"{version} · {len(seen)} unique strategies", loc="left",
                     fontsize=13, fontweight="semibold")
        ax.set_xlabel(f"{state['projection_label']} component 1")
        ax.set_ylabel(f"{state['projection_label']} component 2")
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", color=colors[c],
                   label=f"{c} · {state['cluster_names'].get(c, '')}")
        for c in range(state["km"].n_clusters)
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.84, 0.5),
               frameon=False, title="Global clusters")
    fig.suptitle("Shared centered-PCA strategy space", fontsize=16, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 0.83, 0.94))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", nargs="*", type=Path,
                        help="logs to compare; default: latest three each for LLM_v2/v3")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/cross_experiment_analysis/LLM_v2_vs_LLM_v3_latest"))
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    add_clustering_method_args(parser)
    args = parser.parse_args()

    paths = args.json or _latest_default_logs(Path("results/quantitative_baseline"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps([str(path.resolve()) for path in paths], indent=2),
                             encoding="utf-8")

    experiments = []
    combined_generations = []
    for path in paths:
        generations = load_generations(load_evolution_json(path))
        label = _experiment_label(path)
        indices = []
        for generation in generations:
            for agent in generation[F_POPULATION]:
                agent["_analysis_experiment_id"] = path.parent.name
                agent["_analysis_source_path"] = str(path.resolve())
            indices.append(len(combined_generations))
            combined_generations.append(generation)
        experiments.append({"path": path, "label": label, "generations": generations,
                            "generation_indices": indices})

    kwargs = clustering_method_kwargs(args)
    kwargs["analysis_source_path"] = manifest_path
    state = cluster_strategies(combined_generations, k=args.k, seed=args.seed, **kwargs)

    comparison_path = args.out_dir / "cross_experiment_cluster_composition.png"
    space_path = args.out_dir / "cross_experiment_strategy_space.png"
    plot_comparison(experiments, state, comparison_path)
    plot_strategy_space(experiments, state, space_path)

    individual_paths = []
    for experiment in experiments:
        path = args.out_dir / f"{experiment['path'].parent.name}_cluster_composition.png"
        plot_cluster_composition(
            experiment["generations"],
            _slice_state(state, experiment["generation_indices"]), path,
        )
        individual_paths.append((experiment, path))

    cache = AnalysisCache(state["embedding_cache_path"])
    cache.put_artifact(run_id=state["analysis_run_id"], artifact_type="cross_composition",
                       path=comparison_path, metadata={"experiments": len(experiments)})
    cache.put_artifact(run_id=state["analysis_run_id"], artifact_type="cross_strategy_space",
                       path=space_path, metadata={"experiments": len(experiments)})
    for experiment, path in individual_paths:
        cache.put_artifact(
            run_id=state["analysis_run_id"], artifact_type="experiment_composition",
            path=path, metadata={"experiment_id": experiment["path"].parent.name},
        )
    print(f"run_id: {state['analysis_run_id']}")
    print(f"output: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
