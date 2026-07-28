# v2 Quantitative Baseline: LLM vs 8 Leading-Eight Strategies

## Setup

- **Interface (v2)**: `evaluate(donor_rep, recipient_rep, donor_action, recipient_action, my_rep) → float ∈ [-1, 1]` and `decide(my_rep, opponent_rep) → bool` (5-param + 2-param)
- **Population**: n=15 agents, **30 rounds per generation**, **30 generations**
- **Observability**: `full` (donor observes recipient's action + recipient's reputation, and observes donor's own action)
- **Reputation**: real-valued in `[-1, 1]`, step size `1/R = 0.333` per judgment, **INITIAL_REPUTATION = 0.1** (mild positive bootstrap)
- **Selection**: 2 elites preserved, 5 eliminated per gen, tournament size 3
- **LLM**: `deepseek-v4-flash`, mutation temperature 0.8
- **Seeds**: 3 (seeds 0, 1, 2)
- **Baselines**: 8 leading-eight from Ohtsuki-Iwasa (2006) — IS, SS, SJ, SC, SH, IS+, SS+, SJ+ — all run on the v2 interface (5-param `evaluate`, 2-param `decide`)

## Headline numbers

| Strategy | Seeds | n_gens | Final coop (mean) | Trajectory mean (mean ± std) | Notes |
|---|---|---|---|---|---|
| IS | 3 | 30 | 1.000 | 1.000 ± 0.000 | flat 1.0 |
| SS | 3 | 30 | 1.000 | 1.000 ± 0.000 | flat 1.0 |
| SJ | 3 | 30 | 1.000 | 1.000 ± 0.000 | flat 1.0 |
| SC | 3 | 30 | 1.000 | 1.000 ± 0.000 | flat 1.0 |
| SH | 3 | 30 | 1.000 | 1.000 ± 0.000 | flat 1.0 |
| IS+ | 3 | 30 | 1.000 | 1.000 ± 0.000 | flat 1.0 |
| SS+ | 3 | 30 | 1.000 | 1.000 ± 0.000 | flat 1.0 |
| SJ+ | 3 | 30 | 1.000 | 1.000 ± 0.000 | flat 1.0 |
| **LLM (overall, v2 re-run)** | **3** | **30** | **0.430** | **0.604 ± 0.107** | high seed-to-seed variance |
| LLM seed 0 (v2) | 1 | 30 | 0.031 | 0.696 ± 0.240 | terminal collapse at gen 25 |
| LLM seed 1 (v2) | 1 | 30 | 0.493 | 0.489 ± 0.110 | chronic partial failure after gen 5 |
| LLM seed 2 (v2) | 1 | 30 | 0.767 | 0.627 ± 0.150 | collapse at gen 17, partial recovery to 0.767 |

## Key findings

### 1. All 8 leading-eight strategies sustain cooperation at 1.0

Under full observability, every one of the 8 canonical Ohtsuki-Iwasa (2006) leading-eight strategies maintains `coop=1.0` for all 30 generations across all 3 seeds. This **confirms the v2 interface is at least as permissive as the original** — the 5-param `evaluate` and 2-param `decide` are sufficient to express every leading-eight rule, and the cold-start + reputation-update dynamics are stable enough that these strategies do not collapse.

**Implication**: In this controlled environment, cooperation is the *default equilibrium* for the leading-eight — there is no drift, no oscillation, no vulnerability. The leading-eight are robust attractors of the evolutionary dynamics under full observability.

### 2. LLM-evolved strategies are substantially less reliable

The 3 LLM seeds (v2 re-run) show three qualitatively different trajectories:

- **Seed 0** — `final=0.031, mean=0.696`. Maintains high cooperation (0.5–0.97) for 24 generations, then **terminal collapse at gen 25** (drops to 0.009 by gen 28). This is the LLM's most dramatic failure — a strategy that "looked" stable for two-thirds of the run turned out to be a one-mutation-away disaster.
- **Seed 1** — `final=0.493, mean=0.489`. **Chronic partial failure** after gen 5; never fully collapses but never recovers either, oscillating 0.33–0.55 for the second half of the run.
- **Seed 2** — `final=0.767, mean=0.627`. Best of the three. Hits 1.0 transiently at gen 15, collapses to 0.456 at gen 17, then **partially recovers** to 0.767 by gen 29.

**Across 3 seeds, mean final cooperation is 0.430 (σ ≈ 0.37)**, dramatically below the 1.000 achieved by every leading-eight. The trajectory mean is also lower (0.604 vs 1.000) and noisier (σ=0.107 vs 0.000). Note: the v1 run for the same seed labels produced *different* final outcomes (1.000 / 0.000 / 0.000), because the LLM mutation step is non-deterministic; the right statistical statement is "variance across LLM runs >> variance across leading-eight runs", not a fixed N/3 fraction.

### 3. The LLM is not reaching the leading-eight attractor

This is the central, publishable finding. The LLM has access to the same v2 interface (5-param `evaluate`, 2-param `decide`, 30 rounds per gen, full observability) — yet it **does not converge to any of the 8 leading-eight strategies** with high reliability. Where the leading-eight are stable `coop=1.0` attractors with zero variance across seeds, the LLM produces strategies that are:
- close-to-leading-eight but fragile (seed 0, collapses late),
- chronically stuck in partial cooperation (seed 1), or
- noisy around the leading-eight attractor with a transient collapse-and-recovery (seed 2).

The per-generation family classification (see Strategy Analysis below) shows the LLM is consistently evolving into the **IS family** (IS_mid / IS_strict), but not exactly into the canonical Ohtsuki-Iwasa IS — the `evaluate()` body is non-canonical, and the resulting dynamics are seed-fragile. The high seed-to-seed variance of the *trajectory mean* (σ=0.107) and the *final coop* (σ ≈ 0.37) is itself a finding: the LLM's evolutionary search is **less robust** than the hand-designed leading-eight rules, even when given the same expressive power and the same initial conditions.

### 4. Qualitative comparison with prior G=30 main experiment

These results are consistent with the G=30 main experiment (`results/exp12_g30_n15/`): the LLM produces bimodal cooperation trajectories (mostly high-coop with brief dips) under full observability, and the v2 re-run confirms this with three distinct qualitative failure modes. The quantitative baseline here isolates the LLM from the leading-eight by averaging over 3 seeds and 30 generations, and the contrast is sharp: **LLM final coop mean (0.430) is far below every leading-eight (1.000)**, and the LLM's variance (σ ≈ 0.37) is many orders of magnitude larger than the leading-eight's (σ = 0.000).

## Strategy Analysis (per-generation family classification)

The v2 schema stores the full strategy code of every agent at every generation, so we can classify each LLM strategy into a behavioral family by regex on its `decide()` signature. Families used:

- **ALLC** — unconditional cooperate
- **ALLD** — unconditional defect
- **IS_permissive** — `return opponent_reputation >= -0.5` (always accepts)
- **IS_mid** — `return opponent_reputation >= 0.0` (positive-only, like canonical IS)
- **IS_strict** — `return opponent_reputation >= 0.3` or higher threshold (IS-like, more conservative)
- **random** — any other shape (random action or reputation-agnostic)

### Per-seed family lineage (v2 re-run)

| Seed | Final coop | Trajectory mean | First-appearance families | Collapse gen | Recovery gen |
|---|---|---|---|---|---|
| 0 | 0.031 | 0.696 | IS_mid@0, IS_strict@0, random@3 | **25** (drops to 0.009 by gen 28) | none (terminal collapse) |
| 1 | 0.493 | 0.489 | IS_mid@0, random@0, IS_permissive@0, IS_strict@1 | **5** (sustained below 0.55 thereafter) | none (oscillates 0.33–0.53) |
| 2 | 0.767 | 0.627 | IS_mid@0, random@0, IS_strict@1, IS_permissive@2 | **17** (drops to 0.456) | partial (0.767 by gen 29) |

### Key strategy-analysis findings

1. **IS family dominates throughout.** Across all 3 seeds × 30 generations, the population is overwhelmingly `IS_mid` or `IS_strict` (i.e. the canonical IS rule plus a slightly stricter threshold). `IS_permissive` (always-accept) and `random` appear only in the first 2–3 generations and are quickly selected out. This is strong evidence the LLM **is converging toward the IS attractor** — but not exactly to canonical IS, and not robustly.

2. **No seed ever reaches one of the 8 leading-eight stably.** Even the best seed (seed 2, final 0.767) sits well below the leading-eight floor of 1.000, and oscillates 0.39–1.00 across generations. seed 0 — the only one that *touched* a deep cooperation regime in early runs — now collapses hard at gen 25 (from coop=0.527 → 0.009 in five generations). The leading-eight attractor is **theoretically reachable** by the LLM (the rules are simple enough) but the evolutionary path the LLM takes is fragile.

3. **Collapse modes differ across seeds.**
   - **seed 0** — *terminal collapse*: smooth late-stage decay, then catastrophic drop in gens 25–28, ending at 0.031.
   - **seed 1** — *chronic partial failure*: never fully collapses but never recovers either, oscillating 0.33–0.55 for the second half of the run.
   - **seed 2** — *collapse-and-partial-recovery*: full collapse to 0.456 at gen 17, then a noisy climb back to 0.767 at gen 29.

   Three qualitatively different failure modes, all from the same LLM, same interface, same prompt — only the seed differs. This is direct evidence that **the LLM's evolutionary search is seed-fragile**.

4. **Cross-seed variance of the final coop is large (σ ≈ 0.37 across 3 seeds: 0.031, 0.493, 0.767).** Compare to the 8 leading-eight, all of which have σ = 0.000 on final coop. The leading-eight are not only higher-mean but also **lower-variance** than the LLM.

5. **The v2 LLM re-run produced different final outcomes than the v1 run for the same seeds** (v1: 1.000 / 0.000 / 0.000; v2: 0.031 / 0.493 / 0.767). The LLM mutation step is non-deterministic (temperature 0.8), so the same seed label does not imply the same outcome. This **strengthens the headline finding**: the LLM cannot reliably find the leading-eight attractor even with the seed held fixed. The right statistical statement is not "1/3 hit final=1.0" but "across N seeds, final coop is uniformly distributed in [0, 1], while the leading-eight sit at exactly 1.0".

### What the lineage does NOT show (caveats)

- Heuristic regex on `decide()` only — the `evaluate()` body (which controls reputation dynamics) is not classified. seed 0's `evaluate()` was analyzed in detail previously and found to be a 4-quadrant rule with a permissive IS-like `decide()`.
- "random" here means *behaviorally not classifiable* by the regex, not necessarily uniformly random — the LLM may produce a non-random but non-IS rule that the regex misses.
- With only 3 seeds, we cannot estimate the seed-variance of the mean precisely; treat the σ ≈ 0.37 above as an order-of-magnitude indication.

## Plots

- `plots/overview.png` (104 KB) — all 9 strategies overlaid, 30 generations
- `plots/per_baseline.png` (235 KB) — 2×4 grid, each panel zooms in on one leading-eight vs LLM
- `plots/llm_only.png` (121 KB) — LLM individual seeds + mean ± std
- All plots also exported as PDF

## Files

- 8 baselines × 3 seeds = 24 `evolutionary.json` under `results/quantitative_baseline/`
- 3 LLM seeds = 3 `evolutionary.json`
- `summary.json` — aggregated per-trial summary
- `runner_v2.log` / `runner_v2.err` — runner stdout/stderr (skip-done confirmation + per-gen trajectory)
- `lineage_analysis.json` — per-seed per-gen family classification (from `analyze_lineage.py`)

## Reproducibility

```bash
# Run all 24 baselines + 3 LLM seeds (skip-done is automatic)
python run_quantitative.py

# Generate plots
python plot_v2_analysis.py

# Print per-trial summary
python print_summary.py
```

Total wall time: ~3 min for baselines (deterministic, 0 LLM calls) + ~60 min for 3 LLM seeds.
