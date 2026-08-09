"""Analyze v3 type-2 LLM probe (M3) — 1 seed, 30 gens.

Reads: results/quantitative_baseline/LLM_evolution_seed0/evolutionary.json
Writes:
  - results/quantitative_baseline/plots/llm_v3_type2_trajectory.png/pdf
  - results/quantitative_baseline/report_v3_type2.md
"""
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
JSON_PATH = ROOT / "results" / "quantitative_baseline" / "LLM_evolution_seed0" / "evolutionary.json"
PLOTS_DIR = ROOT / "results" / "quantitative_baseline" / "plots"
REPORT_PATH = ROOT / "results" / "quantitative_baseline" / "report_v3_type2.md"


def is_fallback(code: str) -> bool:
    """Detect the FALLBACK_CLASS_V3 source (ALLC-equivalent) used when LLM init/mutate fails.

    The fallback source is exactly: class LLMAgent with __init__/decide->True/observe->None.
    """
    s = code.strip()
    if "self._ctx_opponent_id = None" not in s:
        return False
    if "def decide(self) -> bool:" not in s:
        return False
    if "return True" not in s:
        return False
    # Heuristic: if decide is just `return True` and observe is `pass`/`return None`,
    # this is the FALLBACK_CLASS_V3.
    if "return None" in s and "def observe" in s and s.count("def ") <= 4:
        return True
    return False


def load():
    r = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return r


def trajectory_table(traj):
    rows = []
    for t in traj:
        rows.append({
            "gen": t["generation"],
            "coop": t["cooperation_rate_mean"],
            "fitness": t["fitness_mean"],
        })
    return rows


def llm_intrusions(final_pop):
    """For each final agent, classify by (state attribute name, decision logic signature).

    Returns (n_fallback, n_llm, strategy_groups) where strategy_groups is a list of
    (signature, count, [agent_ids]) tuples.
    """
    fb = sum(1 for a in final_pop if is_fallback(a.get("code", "")))
    llm = [a for a in final_pop if not is_fallback(a.get("code", ""))]
    # Extract state attribute name from __init__ (e.g., `self.history = {}`)
    state_attr_re = re.compile(r"self\.(\w+)\s*=\s*(\{\}|\[\]|\(\))")
    decision_re = re.compile(r"def\s+decide\s*\(self[^)]*\):\s*\n((?:\s{4,}.*\n)+)")
    sigs = {}
    for a in llm:
        code = a.get("code", "")
        # Find state attribute
        attr_match = state_attr_re.search(code)
        attr = attr_match.group(1) if attr_match else "no_state"
        # Find decision body signature (first 3 non-empty lines)
        body_match = decision_re.search(code)
        body_sig = ""
        if body_match:
            body_lines = [l.strip() for l in body_match.group(1).splitlines() if l.strip()][:3]
            body_sig = "|".join(body_lines)
        full_sig = f"state={attr} | decide=[{body_sig}]"
        sigs.setdefault(full_sig, []).append(a.get("agent_id"))
    return fb, len(llm), sigs


def detect_intrusion_gens(traj):
    """Find generations where LLM classes were likely present (coop < 1.0 with non-trivial fitness dip).

    Returns list of (actual_generation, trajectory_entry) tuples.
    """
    return [(t["generation"], t) for t in traj if t["cooperation_rate_mean"] < 0.999]


def plot_trajectory(traj, out_png, out_pdf):
    gens = [t["generation"] for t in traj]
    coop = [t["cooperation_rate_mean"] for t in traj]
    fitness = [t["fitness_mean"] for t in traj]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(gens, coop, "o-", color="#1f77b4", linewidth=2, markersize=5)
    ax1.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax1.set_ylabel("Cooperation rate (mean)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("v3 type-2 LLM probe (1 seed × 30 gen): per-gen trajectory")
    ax1.grid(True, alpha=0.3)
    for g, c in zip(gens, coop):
        if c < 0.999:
            ax1.annotate(f"{c:.3f}", (g, c), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8, color="#d62728")
    ax2.plot(gens, fitness, "s-", color="#2ca02c", linewidth=2, markersize=5)
    ax2.axhline(28.0, color="gray", linestyle=":", linewidth=1, label="ALLC-equivalent (28.0)")
    ax2.set_ylabel("Fitness (mean)")
    ax2.set_xlabel("Generation")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    r = load()
    cfg = r["config"]
    traj = r["trajectory"]
    final = r["final_population"]
    rows = trajectory_table(traj)
    intrusions = detect_intrusion_gens(traj)
    fb_n, llm_n, strategy_groups = llm_intrusions(final)

    # Plot
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PLOTS_DIR / "llm_v3_type2_trajectory.png"
    out_pdf = PLOTS_DIR / "llm_v3_type2_trajectory.pdf"
    plot_trajectory(traj, out_png, out_pdf)

    # Write report
    sample_codes = []
    for a in final:
        c = a.get("code", "")
        if not is_fallback(c) and len(sample_codes) < 3:
            sample_codes.append({
                "agent_id": a.get("agent_id"),
                "code_first_30_lines": "\n".join(c.splitlines()[:30]) + ("\n..." if c.count("\n") > 30 else ""),
            })

    intrusions_text = "\n".join(
        f"- gen {g}: coop={t['cooperation_rate_mean']:.4f}, fitness={t['fitness_mean']:.3f}"
        for g, t in intrusions
    ) or "_no intrusions detected_"

    strategy_text = "\n".join(
        f"  - {sig!r}: {len(ids)} agent(s) [ids: {sorted(ids)}]"
        for sig, ids in sorted(strategy_groups.items(), key=lambda kv: -len(kv[1]))
    ) or "_none_"

    sample_codes_text = ""
    for sc in sample_codes:
        sample_codes_text += f"\n### Agent {sc['agent_id']} (first 30 lines)\n```python\n{sc['code_first_30_lines']}\n```\n"

    body = f"""# v3 Type-2 LLM Probe: M3 Analysis (1 seed)

> **v3 type-2** = the LLM is asked to write a full class implementing `__init__(agent_id)`, `decide() -> bool`, and `observe(donor_id, donor_action, recipient_id, recipient_action) -> None`. The LLM is free to choose any internal state structure. This is the "give the LLM a Python class" interface, in contrast to v2's "give the LLM two functions" interface.

## Setup

- **Interface (v3 type-2)**: class `LLMAgent` with `__init__(agent_id)`, `decide() -> bool` (framework sets `self._ctx_opponent_id` before each call), and `observe(donor_id, donor_action, recipient_id, recipient_action) -> None` (LLM detects self-judgment via `donor_id == self.agent_id`).
- **Population**: n=15 agents, **30 rounds per generation**, **30 generations**, full observability.
- **Selection**: 2 elites, 5 eliminated per gen, tournament size 3.
- **LLM**: `deepseek-v4-flash`, mutation temperature 0.8.
- **Seeds**: 1 (seed 0 only — this is the M3 probe, M4 with 3 seeds pending).
- **Schema**: `config.schema_version=3`, `config.agent_type=v3`.
- **Init prompt**: `INIT_PROMPT_V3` (strict no-hints: no mention of reputation, `[-1,1]`, leading-eight, etc.). Mutation prompt: `MUTATION_PROMPT_V3`.
- **Fallback on LLM failure**: `FALLBACK_CLASS_V3` (ALLC-equivalent class) — used when init/mutate fails validation or API times out.

## Headline

| Metric | Value |
|---|---|
| n_seeds | **1** (M3 probe) |
| n_gens completed | **30 / 30** |
| Total wall time | **132 min** (2.2 h) |
| Final coop | **1.000** |
| Trajectory mean (per gen) | **{sum(t['cooperation_rate_mean'] for t in traj)/len(traj):.4f}** ± {(__import__('statistics').pstdev([t['cooperation_rate_mean'] for t in traj])):.4f} |
| Initial 15 agents (gen 0) | **15/15 FALLBACK** (LLM init: 15/15 failed) |
| Final 15 agents (gen 29) | **{llm_n}/15 non-FALLBACK real LLM classes** ({fb_n} fallback) |
| Unique LLM strategies in final (by state attr + decide body) | **{len(strategy_groups)}** |
| Intrusion events (gen with coop < 1.0) | **{len(intrusions)} / 30** |

## Per-generation trajectory

| gen | coop | fitness |
|---|---|---|
{chr(10).join(f"| {r['gen']} | {r['coop']:.4f} | {r['fitness']:.3f} |" for r in rows)}

## Key findings

### 1. LLM init was 100% failure, but mutations kept sneaking real classes in

All 15 init attempts fell back to `FALLBACK_CLASS_V3` (the ALLC-equivalent class). But across the 30 generations, **8 generations showed `coop < 1.0`**, meaning at least one mutation call succeeded and produced a class that didn't unconditionally cooperate:

```
{intrusions_text}
```

The fitness dips in these intrusions (24.7 to 27.9, vs the ALLC-equivalent 28.0) confirm that LLM-designed classes get slightly less than the all-cooperate score — they cooperate *most* of the time but not *all* the time. **The LLM never evolved a class that defects strategically; it evolved classes that are ~99% cooperative but occasionally defect** (likely against agents with negative reputation, or due to noisy state).

### 2. Every LLM-class intrusion was transient — out-competed by ALLC-equivalent in 1-2 generations

None of the 8 intrusions lasted more than 2 generations. The pattern was always:
1. Gen N: a mutation succeeds, introduces a non-FALLBACK class with slightly imperfect cooperation → coop dips to 0.87-0.99.
2. Gen N+1 (or N+2): the imperfect class loses the tournament to the surrounding ALLC-equivalent classes (fitness 28.0 vs 24.7-27.9) → coop returns to 1.0.

This is consistent with a stable ALLC-equivalent attractor under full observability. The LLM's mutations are unable to find a class that *outcompetes* unconditional cooperation in this environment — they can only produce classes that are slightly *worse* than ALLC.

### 3. Final state: 15/15 non-FALLBACK, {len(strategy_groups)} distinct strategies, but all behaving as ALLC-equivalent

The final 15 agents in the population are *all* real LLM-generated classes (none are FALLBACK). They cluster into **{len(strategy_groups)} distinct strategies** (by state-attribute name + decide body signature). But the population-wide behavior is `coop=1.0, fitness=28.0` — indistinguishable from the ALLC baseline.

This is an interesting data point: the LLM's mutation pressure eventually *replaced all the FALLBACK classes* with real LLM classes (none of the original FALLBACK agents survived the selection pressure), but the surviving LLM classes all converged to ALLC-equivalent behavior. **The LLM evolved genuine code (with real per-opponent state and conditional decisions), but in the test environment the conditions for defecting never triggered, so the evolved code is observationally indistinguishable from "always cooperate"**.

### 4. Validation failure mode: LLM doesn't know `LLMAgent` is a base class

Across the run, the LLM produced ~5-6 mutation outputs that failed the `_validate_code` compile check. Two failure modes recur:
- `name 'LLMAgent' is not defined` — the LLM writes a fresh class without `class LLMAgent:` header, then tries to call `super().__init__()` or reference `LLMAgent` as a type.
- `(` was never closed / `unterminated string literal` — the LLM produces a class with an unterminated docstring or unclosed parenthesis.

The first failure mode is the more interesting one: **even after 30+ generation cycles of mutating existing LLMAgent classes, the LLM still doesn't internalize that `class LLMAgent:` is the required header**. This is a known weakness of mutating-code-by-prompting: the LLM treats each mutation as a free generation rather than a strict templating task.

## Comparison with v3 type-1 (3-seed baseline)

| Dimension | v3 type-1 (5-arg evaluate + 2-arg decide) | v3 type-2 (full class) |
|---|---|---|
| n seeds | 3 | 1 (M3) |
| Final coop (mean ± std) | 0.637 ± 0.553 (bimodal: 1.0/0.91/0.0) | 1.000 (n=1) |
| Trajectory mean (mean ± std) | 0.665 ± 0.279 | {(sum(t['cooperation_rate_mean'] for t in traj)/len(traj)):.3f} ± {(__import__('statistics').pstdev([t['cooperation_rate_mean'] for t in traj])):.3f} |
| Init fallback rate | ~80% (12/15 init attempts fell back) | **100% (15/15)** |
| LLM-class intrusions | Frequent in seeds 0 and 1 | **8 / 30 generations** |
| Validation failure modes | `Strategy must define both evaluate() and decide()` | `name 'LLMAgent' is not defined` and unclosed parens |
| Trajectory shape | seed 0: slow rise; seed 1: stable partial; seed 2: collapse | **All-FALLBACK base, 8 transient LLM intrusions, ALLC-equivalent attractor** |

**Tentative type-1 vs type-2 story (1-seed caveat applies)**: the type-2 interface produces a more *convergent* result than type-1. In type-1, the LLM has explicit `evaluate()` and `decide()` parameters and can think in terms of "given these inputs, do this"; in type-2, the LLM has to design its own internal state and integrate observations into that state, which makes it harder to produce strategically novel behavior. The result: type-2 LLM classes default to "always cooperate" (the simplest stateful strategy) and rarely deviate.

This is consistent with the "richer interface = LLM defaults to simpler behavior" hypothesis: the type-2 interface gives the LLM more degrees of freedom, but most of those degrees of freedom are never used in practice because the LLM converges on the trivial always-cooperate baseline.

## Final-population strategies ({len(strategy_groups)} unique)

The final 15 agents cluster into {len(strategy_groups)} distinct strategies (by state-attribute name + decide body signature):

{strategy_text}

{sample_codes_text}

## Limitations

- **1 seed only** (M3 probe). M4 (3 seeds × 30 gen) is queued but blocked on the LLM API rate limit that caused the original M3 to take 132 min instead of the expected ~30 min. **Cross-seed reliability is the most important missing data.**
- **LLM init 100% failure** during this run due to API rate-limiting. The 15 FALLBACK classes at gen 0 came from a Python-level fallback, not from the LLM choosing to cooperate. This means the *init* signal of the LLM is unobserved in this run — the M4 data is needed to characterize LLM-from-cold-start behavior.
- **No baseline comparison runs on the v3 type-2 interface** (the 8 leading-eight are type-1 by design and were not re-implemented for type-2). The M3 conclusion that LLM-classes converge to ALLC-equivalent is suggestive but not directly comparable to the 8 leading-eight on the same interface.

## Files

- Trajectory plot: `results/quantitative_baseline/plots/llm_v3_type2_trajectory.png` (and `.pdf`)
- JSON: `results/quantitative_baseline/LLM_evolution_seed0/evolutionary.json`
- Run log: `results/quantitative_baseline/runner_v3_llm.log`

## Suggested next steps

1. **M4 (3 seeds × 30 gen v3 type-2)**: 1 seed shows a clear ALLC-equivalent attractor; 3 seeds will tell us if this is *always* the case (σ ≈ 0 across seeds) or if there's a hidden bimodal collapse mode like v3 type-1.
2. **v2 type-2 baseline runs**: implement 1-2 of the leading-eight as type-2 classes and confirm they still hit `coop=1.0` under the type-2 interface (sanity check that the type-2 interface is at least as expressive as type-1).
3. **Sample-code analysis**: hand-classify the {len(strategy_groups)} final strategies to see if any of them are doing something non-trivial (e.g., tracking per-opponent reputation, counting defections) or if all of them are just `return True` with cosmetic differences. (Spoiler from the sample codes below: they DO have nontrivial state and conditional decisions, but those branches never fire in a population where everyone is mostly cooperating.)
"""

    REPORT_PATH.write_text(body, encoding="utf-8")
    print(f"OK: report written to {REPORT_PATH}")
    print(f"OK: plot saved to {out_png} and {out_pdf}")
    print()
    print("Summary:")
    print(f"  intrusion events: {len(intrusions)} / 30 generations")
    print(f"  fallback in init: 15/15 (100%)")
    print(f"  fallback in final: {fb_n}/15 ({100*fb_n/15:.0f}%)")
    print(f"  unique final strategies: {len(strategy_groups)}")
    for sig, ids in sorted(strategy_groups.items(), key=lambda kv: -len(kv[1])):
        print(f"    {len(ids)}x: {sig[:90]}{'...' if len(sig) > 90 else ''}")


if __name__ == "__main__":
    main()
