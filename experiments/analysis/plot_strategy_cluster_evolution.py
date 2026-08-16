"""Cluster all logged agent strategies across generations and visualize them in two ways.

1) strategy-cluster composition per generation: stacked bar plot of cluster counts
   over generation number, similar to the reference plot.
2) generation-by-generation PCA animation: the same embedding + K-means clustering
   projected onto PCA1 x PCA2, with one point per unique strategy code per generation
   and agent IDs annotated (grouped into a list when the same code is shared).

Reusable pipeline logic (data loading, embedding/K-means/projection) lives in the
``clustering`` sub-package; this module only draws figures and exposes the CLI.

K is auto-selected by silhouette (max over K in [2, 20]) unless an explicit
``--k`` is given.

Usage:
  uv run python -m experiments.analysis.plot_strategy_cluster_evolution --json results/.../evolutionary.json
  uv run python -m experiments.analysis.plot_strategy_cluster_evolution --json results/.../evolutionary.json --k 10
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from PIL import Image

from experiments.evolution_log import load_evolution_json

from .clustering.io import load_generations, merge_generations
from .clustering.cli_args import add_clustering_method_args, clustering_method_kwargs
from .clustering.cache import AnalysisCache
from .clustering.pipeline import cluster_strategies, serialize_id_list
from .paths import evolution_json_path


PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]

TRANSITION_STYLES = {
    "independent_init": {"color": "#C2410C", "linestyle": "--",
                         "label": "Independent initialization"},
    "imitate": {"color": "#6D28D9", "linestyle": "-",
                "label": "Imitation"},
}


def _colors(n: int):
    """Return a stable, color-blind-friendly palette for any cluster count."""
    if n <= len(PALETTE):
        return PALETTE[:n]
    return [plt.colormaps["turbo"](x) for x in np.linspace(0.05, 0.95, n)]


def _style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.8, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAB2BF")
    ax.tick_params(colors="#4B5563", labelsize=9)


class FixedPalettePillowWriter(PillowWriter):
    """Quantize every GIF frame against one palette derived from frame one."""

    def finish(self):
        if not self._frames:
            return
        master = self._frames[0].convert("RGB").quantize(
            colors=256,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        frames = [
            frame.convert("RGB").quantize(
                palette=master,
                dither=Image.Dither.NONE,
            )
            for frame in self._frames
        ]
        frames[0].save(
            self.outfile,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / self.fps),
            loop=0,
            optimize=False,
        )


def _find_ffmpeg(explicit_path: str | None = None) -> str:
    """Locate FFmpeg from CLI, PATH, or a standard WinGet installation."""
    if explicit_path:
        path = Path(explicit_path)
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"FFmpeg not found: {path}")
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    winget_root = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    matches = sorted(winget_root.glob("Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe"))
    if matches:
        return str(matches[-1])
    raise FileNotFoundError(
        "FFmpeg is required for MP4 output; add it to PATH or pass --ffmpeg-path."
    )


def plot_cluster_composition(generations, cluster_state, out_path: Path):
    k = cluster_state["km"].n_clusters
    cluster_names = cluster_state["cluster_names"]
    generation_numbers = [gen["generation"] for gen in generations]
    counts = np.zeros((len(generations), k), dtype=int)

    for gi, gen in enumerate(generations):
        row_idx = cluster_state["gen_rows"][gi]
        labels = cluster_state["labels"][row_idx]
        c = Counter(int(v) for v in labels)
        for cid, n in c.items():
            counts[gi, cid] = n

    colors = _colors(k)
    fig, ax = plt.subplots(figsize=(12.5, 6.5), facecolor="white")
    bottom = np.zeros(len(generations))
    for cid in range(k):
        values = counts[:, cid]
        if np.all(values == 0):
            continue
        ax.bar(
            generation_numbers,
            values,
            bottom=bottom,
            width=0.9,
            color=colors[cid],
            alpha=0.95,
            edgecolor="white",
            linewidth=0.35,
            label=f"Cluster {cid} · {cluster_names.get(cid, '')}",
        )
        bottom += values

    _style_axes(ax)
    ax.set_title("Strategy-cluster composition over generations", loc="left",
                 fontsize=16, fontweight="semibold", color="#172033", pad=14)
    ax.text(0, 1.01, f"Global {cluster_state['embedding_label']} + K-means clustering · K = {k}",
            transform=ax.transAxes, fontsize=9.5, color="#667085", va="bottom")
    ax.set_xlabel("Generation", fontsize=10, color="#344054", labelpad=8)
    ax.set_ylabel("Number of agents", fontsize=10, color="#344054", labelpad=8)
    ax.set_xlim(min(generation_numbers) - 0.5, max(generation_numbers) + 0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8,
              frameon=False, title="Strategy clusters", title_fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.79, 1))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_generation_animation(
    generations,
    cluster_state,
    out_path: Path,
    fps: int = 4,
    mutation_arrows: bool = True,
    dpi: int = 140,
    ffmpeg_path: str | None = None,
):
    Z = cluster_state["Z"]
    labels = cluster_state["labels"]
    cluster_names = cluster_state["cluster_names"]
    n_clusters = cluster_state["km"].n_clusters
    colors = _colors(n_clusters)

    x_all = Z[:, 0]
    y_all = Z[:, 1]
    x_min, x_max = np.min(x_all), np.max(x_all)
    y_min, y_max = np.min(y_all), np.max(y_all)
    pad_x = 0.15 * (x_max - x_min + 1e-9)
    pad_y = 0.15 * (y_max - y_min + 1e-9)

    fig, ax = plt.subplots(figsize=(10.5, 6.5), facecolor="white")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=colors[cid],
               markeredgecolor="white", markersize=7,
               label=f"Cluster {cid} · {cluster_names.get(cid, '')}")
        for cid in range(n_clusters)
    ]
    if mutation_arrows:
        for style in TRANSITION_STYLES.values():
            legend_handles.append(
                Line2D([0], [0], color=style["color"], linewidth=1.5,
                       linestyle=style["linestyle"], marker=">", markevery=[1],
                       markersize=6, label=style["label"])
            )

    def agent_state(frame_idx):
        """Map agent id to (code, projected x, projected y) in one generation."""
        gen = generations[frame_idx]
        row_idx = cluster_state["gen_rows"][frame_idx]
        coords = Z[row_idx]
        return {
            int(agent["agent_id"]): (agent["code"], *coords[pos], agent.get("origin"))
            for pos, agent in enumerate(gen["population"])
        }

    def draw_frame(frame_idx):
        ax.clear()
        gen = generations[frame_idx]
        row_idx = cluster_state["gen_rows"][frame_idx]
        gen_labels = labels[row_idx]
        gen_Z = Z[row_idx]

        ax.scatter(x_all, y_all, s=16, c="#CBD2DC", alpha=0.18,
                   edgecolors="none", zorder=1)

        transition_counts = {origin: 0 for origin in TRANSITION_STYLES}
        if mutation_arrows and frame_idx > 0:
            previous = agent_state(frame_idx - 1)
            current = agent_state(frame_idx)
            for agent_id in sorted(previous.keys() & current.keys()):
                old_code, x0, y0, _ = previous[agent_id]
                new_code, x1, y1, origin = current[agent_id]
                if old_code == new_code or np.allclose((x0, y0), (x1, y1)):
                    continue
                if origin not in TRANSITION_STYLES:
                    continue
                transition_counts[origin] += 1
                style = TRANSITION_STYLES[origin]
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": style["color"],
                        "linewidth": 1.35,
                        "linestyle": style["linestyle"],
                        "alpha": 0.78,
                        "shrinkA": 7,
                        "shrinkB": 7,
                        "connectionstyle": "arc3,rad=0.08",
                    },
                    zorder=2,
                )
                ax.annotate(
                    f"A{agent_id}",
                    ((x0 + x1) / 2, (y0 + y1) / 2),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6.5,
                    color=style["color"],
                    bbox={"boxstyle": "round,pad=0.14", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.82},
                    zorder=4,
                )

        # group by exact strategy code in this generation, annotate all agent ids that share it
        code_to_ids = defaultdict(list)
        code_to_first_idx = {}
        for local_pos, agent in enumerate(gen["population"]):
            code = agent["code"]
            code_to_ids[code].append(int(agent["agent_id"]))
            if code not in code_to_first_idx:
                code_to_first_idx[code] = local_pos

        for local_pos, agent in enumerate(gen["population"]):
            code = agent["code"]
            if code not in code_to_first_idx:
                continue
            # keep only one scatter point per unique code in this generation
            if local_pos != code_to_first_idx[code]:
                continue
            cluster_id = int(gen_labels[local_pos])
            x, y = gen_Z[local_pos]
            ax.scatter(
                x,
                y,
                s=105,
                c=[colors[cluster_id]],
                alpha=0.92,
                edgecolors="white",
                linewidths=0.9,
                zorder=3,
            )
            label_text = serialize_id_list(code_to_ids[code])
            ax.annotate(
                label_text,
                (x, y),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=7.5,
                color="#253044",
                fontweight="medium",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.78},
                zorder=4,
            )

        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        _style_axes(ax)
        ax.grid(axis="both", color="#E3E7ED", linewidth=0.7, alpha=0.7)
        projection_label = cluster_state["projection_label"]
        ax.set_xlabel(f"{projection_label} component 1", fontsize=10, color="#344054")
        ax.set_ylabel(f"{projection_label} component 2", fontsize=10, color="#344054")
        ax.set_title(f"Strategy space · Generation {gen['generation']}", loc="left",
                     fontsize=15, fontweight="semibold", color="#172033", pad=12)
        subtitle = f"{len(code_to_ids)} unique strategies · labels show agent IDs"
        if mutation_arrows:
            subtitle += (
                f" · {transition_counts['independent_init']} independent"
                f" / {transition_counts['imitate']} imitation changes"
            )
        ax.text(0, 1.01, subtitle,
                transform=ax.transAxes, fontsize=9, color="#667085", va="bottom")
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1),
                  fontsize=7.5, frameon=False, title="Strategy clusters",
                  title_fontsize=8.5)

    fig.subplots_adjust(right=0.76, left=0.09, top=0.88, bottom=0.11)
    animation = FuncAnimation(fig, draw_frame, frames=len(generations), interval=1000 / fps)
    if out_path.suffix.lower() == ".gif":
        writer = FixedPalettePillowWriter(fps=fps)
    elif out_path.suffix.lower() == ".mp4":
        rcParams["animation.ffmpeg_path"] = _find_ffmpeg(ffmpeg_path)
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=["-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart"],
        )
    else:
        raise ValueError(f"Unsupported animation format: {out_path.suffix}")
    animation.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out_path}")


def _plot_one(generations, cluster_state, out_dir: Path, args) -> None:
    """Render the composition plot + PCA animation for one run."""
    static_path = out_dir / "plot_strategy_cluster_composition_per_generation.png"
    plot_cluster_composition(generations, cluster_state, static_path)
    cache = AnalysisCache(cluster_state["embedding_cache_path"])
    if cluster_state.get("analysis_run_id") is not None:
        cache.put_artifact(
            run_id=cluster_state["analysis_run_id"], artifact_type="cluster_composition",
            path=static_path, metadata={"format": "png"},
        )

    gif_path = out_dir / "plot_strategy_pca_evolution.gif"
    plot_generation_animation(
        generations,
        cluster_state,
        gif_path,
        fps=4,
        mutation_arrows=args.mutation_arrows,
        dpi=args.animation_dpi,
    )
    if cluster_state.get("analysis_run_id") is not None:
        cache.put_artifact(
            run_id=cluster_state["analysis_run_id"], artifact_type="strategy_evolution",
            path=gif_path, metadata={"format": "gif", "fps": 4,
                                    "dpi": args.animation_dpi},
        )
    mp4_path = out_dir / "plot_strategy_pca_evolution.mp4"
    plot_generation_animation(
        generations,
        cluster_state,
        mp4_path,
        fps=4,
        mutation_arrows=args.mutation_arrows,
        dpi=args.animation_dpi,
        ffmpeg_path=args.ffmpeg_path,
    )
    if cluster_state.get("analysis_run_id") is not None:
        cache.put_artifact(
            run_id=cluster_state["analysis_run_id"], artifact_type="strategy_evolution",
            path=mp4_path, metadata={"format": "mp4", "fps": 4,
                                    "dpi": args.animation_dpi},
        )
    print(f"static: {static_path}")
    print(f"gif: {gif_path}")
    print(f"mp4: {mp4_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=str, action="append", default=None,
                    help="path to evolutionary.json (repeatable: multiple files "
                         "are merged by generation and clustered together)")
    ap.add_argument("--k", type=int, default=None,
                    help="global K for K-means clustering (default: auto-select by silhouette)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for K-means and projection")
    add_clustering_method_args(ap)
    ap.add_argument("--out-dir", type=str, default=None, help="directory for generated figures")
    ap.add_argument("--animation-dpi", type=int, default=140,
                    help="animation render resolution (default: 140 dpi)")
    ap.add_argument("--ffmpeg-path", type=str, default=None,
                    help="path to ffmpeg executable; auto-detected by default")
    ap.add_argument(
        "--mutation-arrows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw arrows for per-agent strategy changes between generations (default: true)",
    )
    args = ap.parse_args()

    if not args.json:
        default_path = evolution_json_path(
            Path(__file__).resolve().parents[2] / "results" / "quantitative_baseline",
            "LLM_v3_fermi_z_v3_g100_1000inter", 2,
        )
        if not default_path.exists():
            raise FileNotFoundError(f"No evolutionary.json found at default path: {default_path}\nPass --json explicitly.")
        json_paths = [default_path]
    else:
        json_paths = [Path(p) for p in args.json]

    # Load one or several runs. With multiple runs, a SINGLE global
    # clustering (KMeans + PCA + names) is fit over ALL runs' codes so
    # labels/names are consistent across runs, but each run is plotted
    # separately with its own trajectory (shared_state reuses the model).
    all_gens = []
    for json_path in json_paths:
        data = load_evolution_json(json_path)
        all_gens.append(load_generations(data))

    out_dir = Path(args.out_dir) if args.out_dir else json_paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    method_kwargs = clustering_method_kwargs(args)

    if len(all_gens) == 1:
        # Single run: fit and plot directly.
        cluster_state = cluster_strategies(
            all_gens[0],
            k=args.k,
            seed=args.seed,
            **{**method_kwargs, "analysis_source_path": str(json_paths[0])},
        )
        _plot_one(all_gens[0], cluster_state, out_dir, args)
        print(f"json: {json_paths}")
    else:
        # Multiple runs: fit ONE global clustering over all runs' codes.
        merged = merge_generations(all_gens)  # only to collect codes
        global_state = cluster_strategies(
            merged,
            k=args.k,
            seed=args.seed,
            **{**method_kwargs, "analysis_source_path": str(json_paths[0])},
        )
        print(f"global clustering: K={global_state['km'].n_clusters}, "
              f"{len(global_state['cluster_names'])} named clusters")
        # Then plot each run separately with the shared model.
        for idx, (gens, json_path) in enumerate(zip(all_gens, json_paths)):
            run_out = out_dir / f"seed{idx}"
            run_out.mkdir(parents=True, exist_ok=True)
            shared = dict(global_state)
            shared["analysis_run_id"] = None  # per-run artifacts, not shared run
            run_state = cluster_strategies(
                gens,
                seed=args.seed,
                **{**method_kwargs, "analysis_source_path": str(json_path)},
                shared_state=shared,
            )
            _plot_one(gens, run_state, run_out, args)
            print(f"json: {json_path}")


if __name__ == "__main__":
    main()
