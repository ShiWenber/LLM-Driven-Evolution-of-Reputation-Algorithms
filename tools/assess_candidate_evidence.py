"""Assess how much evidence a single candidate's advantage actually carries.

Answers "is this candidate good enough to claim?" from the recorded JSON only.
Five checks, ordered by how easily they overturn an optimistic reading:

1. **Independence.** The invasion scan's rule is
   ``strictly_higher_fitness_always_copied``, i.e. selection strength beta ->
   infinity. Under beta -> infinity a takeover is a *deterministic consequence*
   of ``d(k) > 0`` -- which is exactly what the fixation plot's left panel draws.
   So "the scan and the benchmark agree" is ONE fact seen twice, not two tests.
   The right panel (``rho``, the only quantity with a probabilistic meaning) is
   the one to quote.

2. **Multiple comparisons.** The candidate is normally the best of many, chosen
   after seeing the data. Under the neutral null the probability of a seed pair
   taking over is computed here, then expanded to the number of candidates.

3. **Within-arm replication.** If only one seed of an arm shows the property, it
   is a lucky seed, not a property of that arm's condition. Reported as a
   one-sided Fisher exact test against the other arms pooled.

4. **Sample size.** Exact (Clopper-Pearson) interval on the takeover rate, plus
   the number of seeds needed for 80% power.

5. **Per-composition strength.** How many compositions have ``d > 0``, and how
   many exceed 1 and 2 replicate standard deviations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


def binom_tail_at_least(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if HAVE_SCIPY:
        lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
        hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
        return lo, hi
    z = 1.959964
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def seeds_for_power(target: float, neutral: float, power: float = 0.80,
                    alpha: float = 0.05, cap: int = 500) -> int | None:
    for n in range(3, cap + 1):
        crit = next((k for k in range(n + 1)
                     if binom_tail_at_least(k, n, neutral) <= alpha), None)
        if crit is None:
            continue
        if binom_tail_at_least(crit, n, target) >= power:
            return n
    return None


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(table at least as extreme as [[a,b],[c,d]]) with the row/col margins fixed."""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    total = 0.0
    for x in range(min(r1, c1) + 1):
        px = (math.comb(c1, x) * math.comb(n - c1, r1 - x)) / math.comb(n, r1)
        if x >= a:
            total += px
    return min(1.0, total)


def leading_eight(probes: list[str]) -> list[str]:
    return [p for p in probes if p.startswith("L") and p[1:].isdigit()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--focal", type=Path, required=True,
                    help="fixation_benchmark.json of the candidate under scrutiny")
    ap.add_argument("--fixation-glob", required=True,
                    help="glob of ALL candidates' fixation_benchmark.json")
    ap.add_argument("--invasion-dir", type=Path, required=True)
    ap.add_argument("--arm-prefix", default="seed",
                    help="label marker separating arms, e.g. 'seed'")
    ap.add_argument("--focal-label", default=None,
                    help="label of the focal candidate inside --invasion-dir")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    focal = json.loads(args.focal.read_text(encoding="utf-8"))
    n = int(focal["config"]["population_size"])
    neutral = 1.0 / n
    label = args.focal_label or args.focal.parent.name
    probes = list(focal["results"])
    le = leading_eight(probes)

    lines: list[str] = []
    lines.append(f"<!-- evidence assessment: {label}, N={n}, neutral={neutral:.3f} -->")
    lines.append("")

    # ---- 1. independence -------------------------------------------------
    lines.append("### 1. 入侵扫描与固定概率不是两份独立证据")
    lines.append("")
    lines.append("扫描规则 `strictly_higher_fitness_always_copied` 等价于 **β→∞**；"
                 "在 β→∞ 下，横扫是 `d(k)>0` 的**确定性推论**，而 `d(k)` 正是固定概率图左panel 画的东西。"
                 "所以「两张图都显示优势」= 同一个事实出现两次，**不是两次检验**。")
    lines.append("")
    lines.append("真正有概率意义的是右panel 的 ρ：")
    lines.append("")
    lines.append("| probe | ρ | 相对中性 | advantage | ±1 s.d. 跨越中性线? |")
    lines.append("|---|---:|---:|---:|:--:|")
    for p in probes:
        e = focal["results"][p]["candidate_invades_probe"]
        band = ""
        if "rho_at_minus_1sd" in e:
            spans = e["rho_at_minus_1sd"] <= neutral <= e["rho_at_plus_1sd"]
            band = "是（不可分辨）" if spans else "否"
        lines.append(f"| {p} | {e['rho']:.4f} | {e['rho'] / neutral:.2f}× | "
                     f"{e['advantage']:+.4f} | {band} |")

    # ---- 2. multiple comparisons ----------------------------------------
    lines.append("")
    lines.append("### 2. 多重比较")
    lines.append("")
    files = sorted(Path().glob(args.fixation_glob))
    per_cand: dict[str, dict[int, dict[str, bool]]] = {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        cand = f.parent.name
        by_seed: dict[int, dict[str, bool]] = {}
        for p in le:
            for c in d["results"][p]["candidate_invades_probe"]["curve"]:
                by_seed.setdefault(int(c["seed"]), {})[p] = (
                    c["payoff_difference"] > 0
                )
        per_cand[cand] = by_seed
    clean = [c for c, bs in per_cand.items()
             if all(all(v.values()) for v in bs.values())]
    n_seed_slots = max(len(bs) for bs in per_cand.values())
    p_cand = binom_tail_at_least(2, 3, neutral)
    n_cands = len(files)
    fw = 1 - (1 - p_cand) ** n_cands
    lines.append(f"- 中性零假设下，单个候选「3 个 seed 中 ≥2 个横扫」的概率 = "
                 f"**{p_cand:.5f}**")
    lines.append(f"- 本批共考察 **{n_cands}** 个候选")
    lines.append(f"- **family-wise p = {fw:.4f}**（Bonferroni 阈值 0.05/{n_cands} = "
                 f"{0.05 / n_cands:.4f}）")
    verdict = ("**不显著**，该命中可由「从 15 个候选里挑最优」解释"
               if fw > 0.05 else "显著")
    lines.append(f"- 结论：{verdict}")
    lines.append("")
    lines.append(f"> 全正 d(k) 的候选：{', '.join(clean) if clean else '（无）'} "
                 f"（共 {len(clean)}/{n_cands}）")

    # ---- 3. within-arm replication --------------------------------------
    lines.append("")
    lines.append("### 3. 臂内复现：是「该条件的性质」还是「一个幸运 seed」？")
    lines.append("")
    arms: dict[str, list[str]] = {}
    for c in per_cand:
        arm = c.rsplit(f"_{args.arm_prefix}", 1)[0]
        arms.setdefault(arm, []).append(c)
    lines.append("| arm | 全正 d(k) 的 seed 数 | 总 seed 数 |")
    lines.append("|---|---:|---:|")
    for arm, cs in arms.items():
        ok = sum(1 for c in cs if c in clean)
        lines.append(f"| {arm} | {ok} | {len(cs)} |")

    # The focal label may lack the run-dir prefix the glob labels carry
    # (`ae0p01_seed3` vs `n20_noisestudy_ae0p01_seed3`), so match by suffix.
    focal_key = next((c for c in per_cand
                      if c == label or c.endswith(f"_{label}")), None)
    if focal_key is None:
        lines.append("")
        lines.append(f"> [warn] focal label {label!r} not found among glob candidates "
                     f"{sorted(per_cand)[:3]}...; skipping arm comparison.")
    else:
        focal_arm = focal_key.rsplit(f"_{args.arm_prefix}", 1)[0]
        focal_cands = arms.get(focal_arm, [])
        a = len([c for c in focal_cands if c in clean])
        b = len(focal_cands) - a
        c_oth = len(clean) - a
        d_oth = (n_cands - len(focal_cands)) - c_oth
        p_fisher = fisher_one_sided(a, b, c_oth, d_oth)
        lines.append("")
        lines.append(f"- 焦点臂 `{focal_arm}`：{a}/{len(focal_cands)} 个 seed 全正；"
                     f"其余臂合计 {c_oth}/{n_cands - len(focal_cands)}")
        lines.append(f"- 单侧 Fisher 精确检验 p = **{p_fisher:.3f}**"
                     + ("（**不显著**，不能说这是该噪声条件的性质）"
                        if p_fisher > 0.05 else "（显著）"))
    lines.append("> 因此把它写成「1% 噪声能演化出优势策略」是没有依据的："
                 "这是 15 个候选里的单个 seed。")

    # ---- 4. sample size --------------------------------------------------
    lines.append("")
    lines.append("### 4. 样本量")
    lines.append("")
    lo, hi = clopper_pearson(2, 3)
    lines.append(f"- 观测横扫率 2/3 = 66.7%，95% 精确 CI = **[{lo:.1%}, {hi:.1%}]**"
                 "（区间几乎覆盖整个 [0,1]，没有信息量）")
    lines.append("")
    lines.append("| 真实横扫率 | 80% 功效所需 seed 数（α=0.05，vs 中性） |")
    lines.append("|---:|---:|")
    for target in (0.15, 0.20, 0.30, 0.50):
        need = seeds_for_power(target, neutral)
        lines.append(f"| {target:.2f} | {need if need else '>500'} |")
    lines.append("")
    lines.append("> 这些是**入侵过程的内部 seed**，不是新的演化，因此成本很低。")

    # ---- 5. per-composition strength ------------------------------------
    lines.append("")
    lines.append("### 5. 逐组成强度（焦点候选）")
    lines.append("")
    lines.append("| probe | d>0 的组成数 | d>1 s.d. | d>2 s.d. | d 最小值 | d 最大值 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for p in probes:
        curve = focal["results"][p]["candidate_invades_probe"]["curve"]
        ds = [c["payoff_difference"] for c in curve]
        stds = [(c.get("payoff_difference_std") or 0.0) for c in curve]
        lines.append(
            f"| {p} | {sum(1 for v in ds if v > 0)}/{len(ds)} | "
            f"{sum(1 for v, s in zip(ds, stds) if v > s)}/{len(ds)} | "
            f"{sum(1 for v, s in zip(ds, stds) if v > 2 * s)}/{len(ds)} | "
            f"{min(ds):+.4f} | {max(ds):+.4f} |"
        )
    lines.append("")
    lines.append("> 逐组成的 d 在 k 上**高度相关**，且八个规范给出完全相同的曲线，"
                 "因此这些计数**不能**当作独立检验的样本量。")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"[written] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
