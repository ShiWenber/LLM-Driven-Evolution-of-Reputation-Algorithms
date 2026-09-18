"""Compare matched frozen-strategy fixation measurements; no simulation calls."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/synchronous_fixation_comparison_20260917"
OLD = ROOT / "results/manuscript_v2/v4_flash/benchmarks/fixation/seed2/fixation_benchmark.json"
NEW = OUT / "synchronous/fixation_benchmark.json"
FIGURES = ROOT / "paper_zh/figures/manuscript_v2"
DIRECTIONS = ("candidate_invades_probe", "probe_invades_candidate")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rho(differences, beta):
    cumulative = np.cumsum(differences)
    return float(1 / (1 + np.exp(-beta * cumulative).sum()))


def profile_probabilities(curve, beta, reverse=False):
    values = np.array([row["replicate_differences"] for row in curve])
    if reverse:
        values = -values[::-1]
    return [rho(values[:, r], beta) for r in range(values.shape[1])]


def within_replicate_drift(curve, repetitions):
    """Split each repetition separately; archived flag splits concatenated blocks."""
    checks = []
    for row in curve:
        blocks = row["block_payoffs"]
        assert len(blocks) % repetitions == 0
        count = len(blocks) // repetitions
        for r in range(repetitions):
            part = blocks[r*count:(r+1)*count]
            d = [b["mutant"] - b["resident"] for b in part]
            h = len(d) // 2
            checks.append({"k": row["mutant_count"], "replicate": r,
                           "gap": abs(statistics.mean(d[:h]) - statistics.mean(d[h:]))})
    return {"max_gap": max(c["gap"] for c in checks),
            "above_0.10": [c for c in checks if c["gap"] > .10]}


def analyze():
    old, new = load(OLD), load(NEW)
    assert old["candidate"]["code_sha256"] == new["candidate"]["code_sha256"]
    for key in ("population_size", "burn_in_interactions", "measure_interactions", "beta",
                "action_error_probability", "observation_error_probability", "seed", "replicates",
                "reputation_reset_between_rounds"):
        assert old["config"][key] == new["config"][key], key
    assert new["config"]["benefit"] == 2 and new["config"]["cost"] == 1
    assert new["config"]["observation_schedule"] == "synchronous"
    assert set(old["probes"]) == set(new["probes"])
    rows, diagnostics = [], {}
    for probe in new["probes"]:
        diagnostics[probe] = {}
        for schedule, data in (("asynchronous", old), ("synchronous", new)):
            curve = data["results"][probe][DIRECTIONS[0]]["curve"]
            assert [c["mutant_count"] for c in curve] == list(range(1, 20))
            for entry in curve:
                assert len(entry["replicate_differences"]) == 5
                assert math.isclose(statistics.mean(entry["replicate_differences"]), entry["payoff_difference"], abs_tol=1e-14)
            diagnostics[probe][schedule] = within_replicate_drift(curve, 5)
        for reverse, direction in enumerate(DIRECTIONS):
            row = {"probe": probe, "direction": direction}
            for schedule, data in (("asynchronous", old), ("synchronous", new)):
                curve = data["results"][probe][DIRECTIONS[0]]["curve"]
                differences = [c["payoff_difference"] for c in curve]
                if reverse:
                    differences = [-d for d in differences[::-1]]
                value = rho(differences, data["config"]["beta"])
                assert math.isclose(value, data["results"][probe][direction]["rho"], abs_tol=1e-14)
                profiles = profile_probabilities(curve, data["config"]["beta"], bool(reverse))
                row[schedule] = value
                row[schedule + "_profile_sd"] = statistics.stdev(profiles)
                row[schedule + "_profile_min"] = min(profiles)
                row[schedule + "_profile_max"] = max(profiles)
            row["difference_sync_minus_async"] = row["synchronous"] - row["asynchronous"]
            row["relative_change_percent"] = 100 * row["difference_sync_minus_async"] / row["asynchronous"]
            rows.append(row)
    report = {"candidate_sha256": new["candidate"]["code_sha256"], "config": new["config"],
              "sources": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (OLD, NEW)],
              "rows": rows, "within_replicate_drift": diagnostics,
              "profile_uncertainty_note": "Five plug-in values calculated separately from replicate-index payoff profiles; descriptive spread, not confidence intervals or evolutionary replicates."}
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return old, new, report


def build_figure(old, new):
    with plt.rc_context({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(3, 3, figsize=(7.1, 6.1),
                                 gridspec_kw={"width_ratios": [1.65, 1, 1]})
        settings = [("Asynchronous", old, "#247A91", "--", "o"),
                    ("Synchronous", new, "#C96B35", "-", "s")]
        for i, probe in enumerate(("L1", "ALLC", "ALLD")):
            for index, (label, data, color, line, marker) in enumerate(settings):
                curve = data["results"][probe][DIRECTIONS[0]]["curve"]
                x = np.array([c["mutant_count"] for c in curve]) / 20
                y = np.array([c["payoff_difference"] for c in curve])
                sd = np.array([c["payoff_difference_std"] for c in curve])
                axes[i, 0].plot(x, y, color=color, ls=line, marker=marker, markevery=3,
                                ms=3, lw=1.1, label=label)
                axes[i, 0].fill_between(x, y-sd, y+sd, color=color, alpha=.14)
                for reverse, direction in enumerate(DIRECTIONS):
                    ax = axes[i, reverse+1]
                    value = data["results"][probe][direction]["rho"]
                    profiles = profile_probabilities(curve, 1, bool(reverse))
                    ax.scatter(index+np.linspace(-.12, .12, 5), profiles,
                               s=10, facecolors="none", edgecolors=color, alpha=.55)
                    ax.scatter(index, value, marker=marker, s=32, color=color, zorder=4)
                    ax.annotate(f"{value:.6f}", (index, value), xytext=(0, 9),
                                textcoords="offset points", ha="center", fontsize=8)
            axes[i, 0].axhline(0, color=".6", lw=.7, ls=":")
            axes[i, 0].set(title=f"{probe}: payoff difference", ylabel=r"$\pi_{candidate}-\pi_{probe}$",
                            xlabel="Candidate fraction k/N", xlim=(0, 1))
            if probe == "ALLD":
                axes[i, 0].axvspan(.675, .725, color=".6", alpha=.16)
                axes[i, 0].annotate("Drift at k=14", (.7, -.17), xytext=(.10, -.52),
                                    arrowprops={"arrowstyle": "->", "color": ".35", "lw": .8},
                                    fontsize=8, color=".25")
            for col, title in ((1, "Candidate invades"), (2, "Probe invades")):
                ax = axes[i, col]
                ax.axhline(.05, color=".5", ls=":", lw=.8)
                lo, hi = ax.get_ylim()
                span = max(hi-lo, .0004)
                ax.set(ylim=(max(-.005, lo-.12*span), hi+.35*span), xlim=(-.6, 1.6),
                       xticks=[0, 1], xticklabels=["Async", "Sync"],
                       ylabel="Fixation probability", title=title)
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        handles = [Line2D([], [], color=c, ls=ls, marker=m, ms=4, label=label)
                   for label, _, c, ls, m in settings]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(.5, 1.0))
        fig.subplots_adjust(left=.085, right=.985, bottom=.07, top=.91, hspace=.7, wspace=.64)
    return fig


def main():
    old, new, report = analyze()
    fig = build_figure(old, new)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"scheduling_comparison.{ext}", dpi=220)
    plt.close(fig)
    (FIGURES / "scheduling_provenance.json").write_text(json.dumps(report["sources"], indent=2), encoding="utf-8")
    lines = ["# 同步与异步固定概率比较结果", "",
             f"同步实验已完成：950 次组成测量，380,000,000 次配对交互；运行 {new['elapsed_seconds']:.1f} 秒。",
             "", "同一 v4-flash seed-2 冻结代表策略；中性固定概率为 0.05。",
             "", "| 对手 | 候选入侵：异步 | 候选入侵：同步 | 反向入侵：异步 | 反向入侵：同步 |",
             "|---|---:|---:|---:|---:|"]
    for probe in [f"L{i}" for i in range(1, 9)] + ["ALLC", "ALLD"]:
        forward = next(r for r in report["rows"] if r["probe"] == probe and r["direction"] == DIRECTIONS[0])
        reverse = next(r for r in report["rows"] if r["probe"] == probe and r["direction"] == DIRECTIONS[1])
        lines.append(f"| {probe} | {forward['asynchronous']:.6f} | {forward['synchronous']:.6f} | {reverse['asynchronous']:.6f} | {reverse['synchronous']:.6f} |")
    lines += ["", "## 解释与限制", "",
              "两种调度下，候选入侵合作型对手的点估计高于中性值，反向低于中性值；对 ALLD 则方向相反。",
              "同步没有一致降低重复间波动：候选入侵 L1、ALLD 的五条重复曲线所对应的固定概率 SD 下降，ALLC 的 SD 上升。",
              "这是一个冻结策略的外部评估，不能回答同步是否减少整个演化过程中的合作崩溃，也不是两种调度的等价性检验。",
              "", "## 逐重复漂移检查", "",
              "L1–L8 与 ALLC 的每次重复前后半段收益差差异均小于 0.008。",
              "ALLD 的 k=14 组成：异步 r=0 超过 0.10；同步 r=2、3、4 超过 0.10。",
              f"ALLD 最大差异：异步 {report['within_replicate_drift']['ALLD']['asynchronous']['max_gap']:.6f}，同步 {report['within_replicate_drift']['ALLD']['synchronous']['max_gap']:.6f}。",
              "两种调度都通过旧的拼接诊断，但逐重复检查揭示了测量期漂移，因此 ALLD 固定概率是有限预算的代入估计，不能当作收敛后稳定性的证据。",
              "", "配置与复现命令见 [README](README.md)，完整数值见 [comparison.json](comparison.json) 和 [comparison.csv](comparison.csv)。",
              "论文中的阴影表示收益差的样本 SD，空心点表示分别代入五条重复曲线的结果；都不是置信区间。"]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": [r for r in report["rows"] if r["probe"] in ("L1", "ALLC", "ALLD")],
                      "within_replicate_drift": report["within_replicate_drift"]}, indent=2))


if __name__ == "__main__":
    main()
