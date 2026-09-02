"""Analyze a multi-seed Fermi run and generate tables, figures, and HTML."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.evolution_log import validate_evolution_results


COLORS = ["#2563eb", "#d97706", "#059669", "#9333ea", "#dc2626"]


def root_lookup(events: list[dict]):
    parents = {event["lineage_id"]: event.get("parent_lineage_id") for event in events}
    cache: dict[int, int] = {}

    def root(lineage_id: int) -> int:
        trail, current = [], lineage_id
        while parents.get(current) is not None:
            if current in cache:
                current = cache[current]
                break
            trail.append(current)
            current = parents[current]
        for item in trail:
            cache[item] = current
        return current

    return root


def safe_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def analyze(files: list[Path], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_rows, update_rows, seed_rows = [], [], []
    runs: list[tuple[int, dict]] = []

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        validate_evolution_results(data)
        seed = int(data["config"]["seed"])
        runs.append((seed, data))
        root_of = root_lookup(data["lineage_events"])
        births = Counter(
            (event.get("birth_gen"), event.get("origin"))
            for event in data["lineage_events"]
            if event.get("origin") != "initial"
        )
        population_size = int(data["config"]["population_size"])

        all_agent_coop, all_agent_fitness = [], []
        for generation in data["trajectory"]:
            population = generation["population"]
            cooperation = [float(agent["cooperation_rate"]) for agent in population]
            fitness = [float(agent["fitness"]) for agent in population]
            self_rep = [float(agent["self_reputation"]) for agent in population]
            roots = Counter(root_of(int(agent["lineage_id"])) for agent in population)
            shares = np.asarray(list(roots.values()), dtype=float) / len(population)
            effective_roots = float(np.exp(-np.sum(shares * np.log(shares))))
            generation_rows.append({
                "seed": seed,
                "generation": int(generation["generation"]),
                "cooperation_mean": float(generation["cooperation_rate_mean"]),
                "cooperation_agent_std": float(np.std(cooperation, ddof=0)),
                "fitness_mean": float(generation["fitness_mean"]),
                "fitness_max": float(generation["fitness_max"]),
                "self_reputation_mean": float(np.mean(self_rep)),
                "unique_strategy_codes": len({
                    hashlib.sha256(agent["code"].encode("utf-8")).hexdigest()
                    for agent in population
                }),
                "root_families": len(roots),
                "effective_root_families": effective_roots,
                "dominant_root_share": max(roots.values()) / len(population),
                "fitness_cooperation_corr": safe_corr(fitness, cooperation),
            })
            all_agent_coop.extend(cooperation)
            all_agent_fitness.extend(fitness)

        for generation in range(1, len(data["trajectory"])):
            imitate = births[(generation, "imitate")]
            independent = births[(generation, "independent_init")]
            accepted = imitate + independent
            update_rows.append({
                "seed": seed,
                "birth_generation": generation,
                "imitate": imitate,
                "independent_init": independent,
                "accepted": accepted,
                "opportunities": population_size,
                "acceptance_rate": accepted / population_size,
            })

        trajectory = data["trajectory"]
        final_population = data["final_population"]
        final_roots = Counter(
            root_of(int(agent["lineage_id"])) for agent in final_population
        )
        accepted_total = sum(
            count for (generation, _origin), count in births.items() if generation is not None
        )
        independent_total = sum(
            count for (_generation, origin), count in births.items()
            if origin == "independent_init"
        )
        seed_rows.append({
            "seed": seed,
            "initial_cooperation": float(trajectory[0]["cooperation_rate_mean"]),
            "final_cooperation": float(trajectory[-1]["cooperation_rate_mean"]),
            "mean_cooperation": float(np.mean([x["cooperation_rate_mean"] for x in trajectory])),
            "min_cooperation": float(min(x["cooperation_rate_mean"] for x in trajectory)),
            "max_cooperation": float(max(x["cooperation_rate_mean"] for x in trajectory)),
            "final_fitness": float(trajectory[-1]["fitness_mean"]),
            "accepted_updates": accepted_total,
            "acceptance_rate": accepted_total / ((len(trajectory) - 1) * population_size),
            "independent_init_updates": independent_total,
            "independent_share_of_accepted": independent_total / accepted_total,
            "final_root_families": len(final_roots),
            "final_dominant_root_share": max(final_roots.values()) / population_size,
            "agent_fitness_cooperation_corr": safe_corr(all_agent_fitness, all_agent_coop),
            "fallback_init": int(data["config"].get("fallback_init_count", 0)),
            "fallback_mutation": int(data["config"].get("fallback_mutation_count", 0)),
        })

    generation_df = pd.DataFrame(generation_rows).sort_values(["seed", "generation"])
    update_df = pd.DataFrame(update_rows).sort_values(["seed", "birth_generation"])
    seed_df = pd.DataFrame(seed_rows).sort_values("seed")
    generation_df.to_csv(output_dir / "generation_metrics.csv", index=False)
    update_df.to_csv(output_dir / "update_metrics.csv", index=False)
    seed_df.to_csv(output_dir / "seed_summary.csv", index=False)

    coop_pivot = generation_df.pivot(index="generation", columns="seed", values="cooperation_mean")
    fitness_pivot = generation_df.pivot(index="generation", columns="seed", values="fitness_mean")
    aggregate = {
        "n_seeds": len(runs),
        "generations": int(generation_df["generation"].max() + 1),
        "population_size": int(runs[0][1]["config"]["population_size"]),
        "updates_per_generation": int(runs[0][1]["config"].get("updates_per_gen", 0)),
        # 旧版框架的 config 可能未记录 llm_concurrency；缺失时按框架默认回退到 population_size。
        "llm_concurrency": int(runs[0][1]["config"].get(
            "llm_concurrency", runs[0][1]["config"]["population_size"]
        )),
        "final_cooperation_mean": float(seed_df["final_cooperation"].mean()),
        "final_cooperation_sd": float(seed_df["final_cooperation"].std(ddof=1)),
        "overall_cooperation_mean": float(generation_df["cooperation_mean"].mean()),
        "peak_cross_seed_mean_cooperation": float(coop_pivot.mean(axis=1).max()),
        "peak_cross_seed_mean_generation": int(coop_pivot.mean(axis=1).idxmax()),
        "final_fitness_mean": float(seed_df["final_fitness"].mean()),
        "final_fitness_sd": float(seed_df["final_fitness"].std(ddof=1)),
        "accepted_updates_total": int(seed_df["accepted_updates"].sum()),
        "acceptance_rate_mean": float(seed_df["acceptance_rate"].mean()),
        "independent_share_mean": float(seed_df["independent_share_of_accepted"].mean()),
        "fallback_total": int(seed_df[["fallback_init", "fallback_mutation"]].to_numpy().sum()),
        "cooperation_between_seed_sd_mean": float(coop_pivot.std(axis=1, ddof=1).mean()),
        "fitness_between_seed_sd_mean": float(fitness_pivot.std(axis=1, ddof=1).mean()),
    }
    payload = {
        "aggregate": aggregate,
        "seeds": seed_df.to_dict(orient="records"),
        "source_files": [str(path.resolve()) for path in files],
    }

    plot_dashboard(generation_df, update_df, seed_df, output_dir, aggregate)
    plot_lineage_summary(runs, output_dir, aggregate)
    plot_relationships(generation_df, output_dir)

    (output_dir / "analysis_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_html(payload, generation_df, update_df, output_dir)
    return payload


def mean_band(ax, frame, metric, ylabel):
    for index, (seed, group) in enumerate(frame.groupby("seed")):
        ax.plot(group["generation"], group[metric], color=COLORS[index], alpha=.55,
                lw=1.5, label=f"seed {seed}")
    pivot = frame.pivot(index="generation", columns="seed", values=metric)
    mean, sd = pivot.mean(axis=1), pivot.std(axis=1, ddof=1)
    ax.fill_between(mean.index, mean - sd, mean + sd, color="#111827", alpha=.12,
                    label="mean ± 1 SD")
    ax.plot(mean.index, mean, color="#111827", lw=2.8, label="cross-seed mean")
    ax.set_ylabel(ylabel); ax.set_xlabel("Generation"); ax.grid(alpha=.22)


def plot_dashboard(generation_df, update_df, seed_df, output_dir, agg):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    n_seeds = agg["n_seeds"]
    generations = agg["generations"]
    population_size = agg["population_size"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 15), constrained_layout=True)
    mean_band(axes[0, 0], generation_df, "cooperation_mean", "Cooperation rate")
    axes[0, 0].set_ylim(0, 1.03); axes[0, 0].set_title("Cooperation trajectories")
    axes[0, 0].legend(ncol=2, fontsize=8)
    mean_band(axes[0, 1], generation_df, "fitness_mean", "Mean fitness")
    axes[0, 1].set_title("Fitness trajectories")

    for index, (seed, group) in enumerate(update_df.groupby("seed")):
        axes[1, 0].plot(group["birth_generation"], group["acceptance_rate"],
                        marker="o", ms=3, color=COLORS[index], label=f"seed {seed}")
    axes[1, 0].axhline(.5, color="#111827", ls="--", lw=1, alpha=.6)
    axes[1, 0].set(title="Accepted Fermi updates", xlabel="Birth generation",
                   ylabel=f"Accepted / {population_size}", ylim=(0, 1)); axes[1, 0].grid(alpha=.22)

    for index, (seed, group) in enumerate(generation_df.groupby("seed")):
        axes[1, 1].plot(group["generation"], group["unique_strategy_codes"],
                        color=COLORS[index], label=f"seed {seed}")
    axes[1, 1].set(title="Exact strategy-code diversity", xlabel="Generation",
                   ylabel="Unique code strings"); axes[1, 1].grid(alpha=.22)

    for index, (seed, group) in enumerate(generation_df.groupby("seed")):
        axes[2, 0].plot(group["generation"], group["dominant_root_share"],
                        color=COLORS[index], label=f"seed {seed}")
    axes[2, 0].set(title="Dominant root-family concentration", xlabel="Generation",
                   ylabel="Largest root-family share", ylim=(0, 1)); axes[2, 0].grid(alpha=.22)

    x = np.arange(len(seed_df)); width = .36
    axes[2, 1].bar(x - width/2, seed_df["final_cooperation"], width,
                   color="#2563eb", label="Final cooperation")
    axes[2, 1].bar(x + width/2, seed_df["mean_cooperation"], width,
                   color="#0891b2", label="20-gen mean")
    axes[2, 1].set_xticks(x, [f"seed {s}" for s in seed_df["seed"]])
    axes[2, 1].set(title="Seed-level cooperation summary", ylabel="Cooperation rate", ylim=(0, 1))
    axes[2, 1].legend(); axes[2, 1].grid(axis="y", alpha=.22)
    fig.suptitle(
        f"Fermi evolution: {n_seeds} seeds × {generations} generations, N={population_size}",
        fontsize=16, weight="bold",
    )
    fig.savefig(output_dir / "evolution_dashboard.png", dpi=180)
    fig.savefig(output_dir / "evolution_dashboard.pdf")
    plt.close(fig)


def plot_lineage_summary(runs, output_dir, agg):
    population_size = agg["population_size"]
    fig, axes = plt.subplots(len(runs), 2, figsize=(15, 4 * len(runs)), constrained_layout=True)
    # 单 seed 时 subplots 返回一维数组，统一成 (n_runs, 2) 以便二维索引。
    axes = np.asarray(axes).reshape(len(runs), 2)
    for row, (seed, data) in enumerate(runs):
        events = [event for event in data["lineage_events"] if event.get("origin") != "initial"]
        gens = range(1, len(data["trajectory"]))
        imitate = [sum(e.get("birth_gen") == g and e.get("origin") == "imitate" for e in events) for g in gens]
        independent = [sum(e.get("birth_gen") == g and e.get("origin") == "independent_init" for e in events) for g in gens]
        axes[row, 0].bar(gens, imitate, color="#2563eb", label="imitate")
        axes[row, 0].bar(gens, independent, bottom=imitate, color="#d97706", label="independent init")
        axes[row, 0].set(title=f"seed {seed}: accepted births by origin", xlabel="Birth generation", ylabel="Births")
        axes[row, 0].set_ylim(0, population_size); axes[row, 0].legend(); axes[row, 0].grid(axis="y", alpha=.2)

        root_of = root_lookup(data["lineage_events"])
        roots = Counter(root_of(int(a["lineage_id"])) for a in data["final_population"])
        counts = sorted(roots.values(), reverse=True)
        labels = [f"family {i+1}" for i in range(len(counts))]
        axes[row, 1].bar(labels, counts, color="#059669")
        axes[row, 1].set(title=f"seed {seed}: final root-family sizes", ylabel="Agents")
        axes[row, 1].tick_params(axis="x", rotation=65); axes[row, 1].grid(axis="y", alpha=.2)
    fig.suptitle("Fermi acceptance and lineage concentration", fontsize=16, weight="bold")
    fig.savefig(output_dir / "lineage_summary.png", dpi=180)
    fig.savefig(output_dir / "lineage_summary.pdf")
    plt.close(fig)


def plot_relationships(generation_df, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    relationships = [
        ("self_reputation_mean", "Mean self-reputation"),
        ("fitness_mean", "Mean fitness"),
        ("cooperation_agent_std", "Within-population cooperation SD"),
        ("effective_root_families", "Effective root families"),
    ]
    for ax, (metric, label) in zip(axes.flat, relationships):
        for index, (seed, group) in enumerate(generation_df.groupby("seed")):
            ax.plot(group[metric], group["cooperation_mean"], "o-", ms=4,
                    lw=1, alpha=.7, color=COLORS[index], label=f"seed {seed}")
        corr = generation_df[metric].corr(generation_df["cooperation_mean"])
        ax.set(xlabel=label, ylabel="Cooperation rate",
               title=f"Cooperation vs {label.lower()}  (r={corr:.3f})")
        ax.set_ylim(0, 1.03); ax.grid(alpha=.22)
    axes[0, 0].legend()
    fig.suptitle("State, payoff, diversity, and cooperation", fontsize=16, weight="bold")
    fig.savefig(output_dir / "state_relationships.png", dpi=180)
    fig.savefig(output_dir / "state_relationships.pdf")
    plt.close(fig)


def write_html(payload, generation_df, update_df, output_dir):
    agg = payload["aggregate"]
    n_seeds = agg["n_seeds"]
    generations = agg["generations"]
    population_size = agg["population_size"]
    llm_concurrency = agg["llm_concurrency"]
    # 每代 learner 更新机会 = 种群规模（与 update_metrics 中 opportunities 口径一致）。
    learner_opportunities = population_size
    total_seed_generation_points = n_seeds * generations
    seed_table = pd.DataFrame(payload["seeds"])[[
        "seed", "initial_cooperation", "final_cooperation", "mean_cooperation",
        "final_fitness", "accepted_updates", "acceptance_rate",
        "independent_share_of_accepted", "final_root_families",
        "final_dominant_root_share", "fallback_init", "fallback_mutation",
    ]].round(4).to_html(index=False, border=0, classes="data")
    content = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{n_seeds} seed × {generations} gen 分析</title><style>
body{{margin:0;background:#f4f7fb;color:#172033;font:15px/1.6 'Segoe UI','Microsoft YaHei',sans-serif}}main{{max-width:1240px;margin:auto;padding:34px 20px}}h1{{margin-bottom:4px}}h2{{margin-top:30px}}.muted{{color:#64748b}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.card,.panel{{background:white;border:1px solid #dbe3ef;border-radius:14px;padding:18px;box-shadow:0 8px 25px #1f29370d}}.value{{font-size:27px;font-weight:750;color:#1d4ed8}}img{{width:100%;display:block;border-radius:10px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #e2e8f0;text-align:right}}th{{background:#eff6ff}}.scroll{{overflow:auto}}@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main><h1>Fermi 演化实验完整分析</h1><p class='muted'>{n_seeds} seeds × {generations} generations · N={population_size} · learner={learner_opportunities}/代（无放回）· LLM concurrency={llm_concurrency}</p>
<section class='cards'><div class='card'>最终合作率均值<div class='value'>{agg['final_cooperation_mean']:.3f}</div><span class='muted'>SD {agg['final_cooperation_sd']:.3f}</span></div><div class='card'>{generations} 代总体合作率<div class='value'>{agg['overall_cooperation_mean']:.3f}</div></div><div class='card'>Fermi 接受率<div class='value'>{agg['acceptance_rate_mean']:.1%}</div><span class='muted'>{agg['accepted_updates_total']} accepted births</span></div><div class='card'>Fallback<div class='value'>{agg['fallback_total']}</div><span class='muted'>{n_seeds} seeds 合计</span></div></section>
<h2>演化总览</h2><div class='panel'><img src='evolution_dashboard.png' alt='演化指标总览'></div><h2>Fermi 与谱系</h2><div class='panel'><img src='lineage_summary.png' alt='Fermi 接受与谱系集中度'></div><h2>状态与结果关系</h2><div class='panel'><img src='state_relationships.png' alt='状态与合作率关系'></div><h2>逐 seed 汇总</h2><div class='panel scroll'>{seed_table}</div><h2>口径说明</h2><div class='panel'><ul><li>阴影为 {n_seeds} 个 seed 的均值 ± 1 个样本标准差，仅用于描述性展示，n={n_seeds} 不适合强统计推断。</li><li>相关系数是 {total_seed_generation_points} 个 seed-generation 点上的描述性 Pearson r，不代表因果关系。</li><li>策略多样性是完整代码字符串的精确去重，不等同于行为多样性。</li><li>root family 由完整 lineage parent 链回溯得到；independent_init 会创建新的 root。</li><li>Fermi 接受率 = 实际出生事件数 / 每代 {learner_opportunities} 次 learner 更新机会。</li></ul></div></main></body></html>"""
    (output_dir / "analysis_report.html").write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True, dest="pattern")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(Path().glob(args.pattern))
    if not files:
        raise SystemExit(f"no files matched {args.pattern!r}")
    result = analyze(files, args.output_dir)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
