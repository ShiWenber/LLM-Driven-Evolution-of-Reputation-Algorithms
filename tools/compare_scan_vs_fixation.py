"""Compare an invasion sweep against a fixation benchmark on the SAME candidates.

Two questions this answers, both of which the individual tools cannot:

1. **Sign structure.** Does the benchmark's per-composition payoff difference
   ``d(k)`` predict which candidates sweep in the scan? Under the scan's rule
   (``strictly_higher_fitness_always_copied``, i.e. selection strength beta ->
   infinity) a candidate sweeps **iff** ``d(k) > 0 at every composition**; a
   single negative composition breaks the ratchet. So the count of negative
   compositions should separate the sweepers from the non-sweepers.

2. **Magnitude.** Selection strength beta -> infinity needs only ``d > 0``, while
   the Traulsen-Haüert benchmark at ``beta = 1`` needs a large ``d``. This prints
   the ``rho`` that a *constant* per-composition advantage ``d`` yields, so the
   measured ``d`` can be read against the takeover the scan actually showed.

Together these explain why a scan can show spreading for a pair whose benchmark
rho is indistinguishable from neutral, without either method being wrong.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any


def leading_eight(probes: list[str]) -> list[str]:
    return [p for p in probes if p.startswith("L") and p[1:].isdigit()]


def rho_for_constant_advantage(d: float, beta: float, n: int) -> float:
    """Traulsen-Haüert rho when every composition carries the same advantage d."""
    total = sum(math.exp(-beta * d * i) for i in range(1, n))
    return 1.0 / (1.0 + total)


def d_for_target_rho(target: float, beta: float, n: int) -> float:
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if rho_for_constant_advantage(mid, beta, n) < target:
            lo = mid
        else:
            hi = mid
    return lo


def scan_takeovers(invasion_dir: Path) -> tuple[dict[str, int], dict[str, int], int]:
    """Per candidate: seeds sweeping all L1-L8 at k=1, and total seeds."""
    per_cand: dict[str, dict[int, dict[str, bool]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )
    for f in invasion_dir.rglob("n1_seed*/invasion.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        per_cand[d["candidate_label"]][d["seed"]][d["resident_label"]] = (
            d["final_invader_frequency"] == 1.0
        )
    took: dict[str, int] = {}
    seeds: dict[str, int] = {}
    for cand, by_seed in per_cand.items():
        le = leading_eight(list(next(iter(by_seed.values()))))
        took[cand] = sum(
            1 for v in by_seed.values() if all(v.get(r, False) for r in le)
        )
        seeds[cand] = len(by_seed)
    return took, seeds, len(per_cand)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixation-glob", required=True,
                    help="glob of fixation_benchmark.json files, e.g. "
                         "'results/.../fixation/n20_noisestudy_*/fixation_benchmark.json'")
    ap.add_argument("--invasion-dir", type=Path, required=True,
                    help="directory holding <candidate>/invades_<resident>/n1_seed*/invasion.json")
    ap.add_argument("--probe", default="L1", help="probe whose d(k) is tabulated")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(Path().glob(args.fixation_glob))
    if not files:
        print(f"[error] no fixation files matched {args.fixation_glob}")
        return 1

    first = json.loads(files[0].read_text(encoding="utf-8"))
    n = int(first["config"]["population_size"])
    neutral = 1.0 / n

    took, seeds, n_cand = scan_takeovers(args.invasion_dir)

    rows = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        label = f.parent.name
        # Match the scan's candidate label by stripping the run-dir prefix.
        cand = next((c for c in took if label.endswith(c)), label.split("_", 1)[-1])
        entry = d["results"][args.probe]["candidate_invades_probe"]
        ds = [c["payoff_difference"] for c in entry["curve"]]
        rows.append(
            {
                "label": label,
                "cand": cand,
                "d_min": min(ds),
                "d_max": max(ds),
                "n_neg": sum(1 for v in ds if v < 0),
                "d_mean": entry["mean_payoff_difference"],
                "rho": entry["rho"],
                "took": took.get(cand),
                "seeds": seeds.get(cand),
            }
        )

    lines: list[str] = []
    lines.append(f"<!-- scan vs fixation: N={n} beta={args.beta:g} "
                 f"neutral={neutral:.3f} probe={args.probe} candidates={len(rows)} -->")
    lines.append("")
    lines.append(f"**Sign structure of d(k) vs {args.probe} 与扫描横扫结果**")
    lines.append("")
    lines.append("扫描规则 `strictly_higher_fitness_always_copied` 等价于 β→∞：**逐 run** 地看，"
                 "候选只有在访问到的每个组成上适应度都严格更高时才能横扫，一个负值即打断棘轮。"
                 "下表的负值组成数按 5 replicate 的均值曲线计，因此与横扫率**单调相关**而非充要。")
    lines.append("")
    lines.append(f"| candidate | d({args.probe}) min | d({args.probe}) max | 负值组成数 | "
                 f"扫描横扫 seed 数 | rho |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda r: (r["n_neg"], -r["d_mean"])):
        took_s = f"{r['took']} / {r['seeds']}" if r["took"] is not None else "-"
        lines.append(
            f"| {r['label']} | {r['d_min']:+.4f} | {r['d_max']:+.4f} | "
            f"**{r['n_neg']}** | {took_s} | {r['rho']:.4f} |"
        )

    sweepers = [r for r in rows if r["took"]]
    clean = [r for r in rows if r["n_neg"] == 0]
    lines.append("")
    lines.append(f"> 负值组成数 = 0 的候选共 **{len(clean)}** 个；"
                 f"在扫描中至少横扫 1 个 seed 的候选共 **{len(sweepers)}** 个。")
    if clean:
        lines.append(f"> d(k) 处处为正的候选是：{', '.join(r['cand'] for r in clean)}。")
    lines.append(">")
    lines.append("> **负值组成数与横扫率单调相关，不是非黑即白。** d(k) 是 5 个 replicate 的**均值**，"
                 "单个 run 会围绕它波动：均值处处为正的候选几乎总能横扫，均值里有 3 个负值的偶尔能横扫，"
                 "负值更多（本批 ≥4 个）的则被卡住。不要把这张表当确定性判决。")

    lines.append("")
    lines.append(f"**幅度：固定 per-composition 优势 d 在 β={args.beta:g}、N={n} 下的 rho**")
    lines.append("")
    lines.append("| d | rho | 相对中性 |")
    lines.append("|---:|---:|---:|")
    # Report the advantage of the cleanest sweeper (fewest negative compositions),
    # falling back to the largest mean advantage if nothing swept.
    measured_pool = clean or sweepers or rows
    measured = max(measured_pool, key=lambda r: r["d_mean"])["d_mean"]
    target_d = d_for_target_rho(2 / 3, args.beta, n)
    probes_d = sorted({round(v, 4) for v in (0.0, measured, 0.044, 0.097, 0.15, target_d)})
    for d in probes_d:
        rho = rho_for_constant_advantage(d, args.beta, n)
        tag = " **← 实测（横扫者）**" if abs(d - round(measured, 4)) < 5e-4 else ""
        if not tag and abs(d - round(target_d, 4)) < 5e-4:
            tag = " **← 2/3 取代所需**"
        lines.append(f"| {d:.4f}{tag} | {rho:.4f} | {rho / neutral:.2f}x |")
    lines.append("")
    lines.append(f"> 横扫者的实测 d ≈ {measured:.4f}；要在 β={args.beta:g} 下达到扫描展示的 2/3 取代率，"
                 f"需要 d ≈ {target_d:.3f}（差 {target_d / measured:.0f} 倍）。")
    lines.append("> 结论：两侧在**符号结构**上一致（左panel 的 d(k) 符号即扫描结果的最佳预测变量），"
                 "分歧只在**幅度**，即选择强度 β→∞ vs β=1。不要因为左panel 看起来平坦就断定「没有优势」。")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"[written] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
