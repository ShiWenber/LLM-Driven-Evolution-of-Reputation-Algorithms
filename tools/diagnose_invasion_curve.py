"""Diagnose whether an invasion-curve point above the y=x diagonal means invasion.

Reads the ``summary.json`` from ``run_invasion`` and decomposes every
``(candidate, resident, count)`` cell into the outcome mixture that produces the
plotted mean:

    y(k) = P(fix) * 1 + P(extinct) * 0 + E[plateau] * P(plateau)

The plotted curve is a MEAN over residents and seeds, and the underlying outcome
is bimodal (absorbed at 0, or absorbed at N), so the mean can sit above the
diagonal while the typical run dies out. This tool reports both.

Why the diagonal is the right neutral reference
-----------------------------------------------
The plotter draws ``y = x`` labelled "No frequency change". For a neutral
Moran-type process the fixation probability from ``k`` copies is exactly ``k/N``
(the martingale property), and ``k/N`` is precisely the diagonal. So a curve
sitting above the diagonal does mean ``P(fix) > k/N`` -- *provided* the y value is
an unbiased estimate of ``P(fix)``. It is not, for two reasons this tool measures:

1. **Censoring.** ``--generations`` truncates the walk. With no mutation the only
   absorbing states are 0 and N, so an unabsorbed run contributes its mid-range
   frequency (~0.5) instead of 0 or 1, which pushes the mean UP at low k and DOWN
   at high k.
2. **Mean vs median.** At low k the diagonal is small, so a small minority of
   fixations is enough to lift the mean clear of it.

Also reports the per-candidate fixation rate from k=1 against the neutral ``1/N``,
with an exact binomial tail probability, and warns that the ten residents are not
independent replicates when the leading eight are indistinguishable.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any


def exact_binom_tail(observed: int, n: int, p: float) -> float:
    """P(X >= observed) for X ~ Binomial(n, p)."""
    return sum(
        math.comb(n, i) * p ** i * (1 - p) ** (n - i)
        for i in range(observed, n + 1)
    )


def cells(summary: dict[str, Any]) -> dict[tuple[str, str, int], float]:
    out = {}
    for cand, residents in summary["groups"].items():
        for res, counts in residents.items():
            for k, cell in counts.items():
                out[(cand, res, int(k))] = float(cell["mean_final_invader_frequency"])
    return out


def raw_runs(summary_path: Path) -> list[dict[str, Any]]:
    runs = []
    for f in summary_path.parent.rglob("invasion.json"):
        runs.append(json.loads(f.read_text(encoding="utf-8")))
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    n = int(summary["population_size"])
    neutral = 1.0 / n
    runs = raw_runs(args.summary)
    if not runs:
        print(f"[warn] no invasion.json under {args.summary.parent}")
        return 1

    by_count: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for d in runs:
        by_count[d["initial_invader_count"]].append(d)

    lines: list[str] = []
    lines.append(f"<!-- invasion-curve diagnosis: N={n} neutral=1/N={neutral:.3f} "
                 f"runs={len(runs)} -->")
    lines.append("")
    lines.append("**What each plotted point is made of**")
    lines.append("")
    lines.append("| k | k/N (neutral) | mean (plotted) | median | extinct | fixed | "
                 "plateau (censored) | mean excl. fixations |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in sorted(by_count):
        vals = [d["final_invader_frequency"] for d in by_count[k]]
        m = len(vals)
        ext = sum(1 for v in vals if v == 0.0)
        fix = sum(1 for v in vals if v == 1.0)
        pla = m - ext - fix
        mean = sum(vals) / m
        med = sorted(vals)[m // 2]
        no_fix = [v for v in vals if v != 1.0]
        lines.append(
            f"| {k} | {k / n:.3f} | {mean:.3f} | {med:.3f} | {ext / m:.1%} | "
            f"{fix / m:.1%} | {pla / m:.1%} | "
            f"{(sum(no_fix) / len(no_fix) if no_fix else float('nan')):.3f} |"
        )

    lines.append("")
    lines.append("> A row whose `mean` exceeds `k/N` while its `median` is 0.000 means "
                 "the mean is lifted by a minority of fixations, not by a typical run "
                 "rising. The `mean excl. fixations` column is the same mean with those "
                 "fixations removed.")

    def labels() -> list[str]:
        return sorted({r for _, r, _ in cells(summary)})

    leading_eight = [r for r in labels() if r.startswith("L") and r[1:].isdigit()]

    lines.append("")
    lines.append("**Per-candidate fixation rate from k=1 vs the neutral benchmark**")
    lines.append("")
    lines.append("A seed is counted as fixing against the leading eight only when the "
                 "candidate takes over from **every** L1-L8 at that seed, which is the "
                 "cleanest 'can it invade the leading eight' statement.")
    lines.append("")
    lines.append("| candidate | seeds taking over L1-L8 | P | neutral | exact P(X>=obs) | "
                 "raw runs | seeds taking over ALLD |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    per_cand: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for d in runs:
        if d["initial_invader_count"] == 1:
            per_cand[d["candidate_label"]].append(d)
    for cand in sorted(per_cand):
        rs = per_cand[cand]
        by_seed: dict[int, dict[str, bool]] = collections.defaultdict(dict)
        for d in rs:
            by_seed[d["seed"]][d["resident_label"]] = (
                d["final_invader_frequency"] == 1.0
            )
        n_seeds = len(by_seed)
        le = sum(
            1 for v in by_seed.values()
            if all(v.get(r, False) for r in leading_eight)
        )
        alld = sum(1 for v in by_seed.values() if v.get("ALLD", False))
        p = exact_binom_tail(le, n_seeds, neutral)
        lines.append(
            f"| {cand} | {le} / {n_seeds} | {le / n_seeds:.0%} | "
            f"{neutral:.3f} | {p:.3f} | {len(rs)} | {alld} / {n_seeds} |"
        )
    lines.append("")
    lines.append("> `raw runs` is the number of leaves; the independent unit is the "
                 "**seed**, because when the leading eight are indistinguishable a "
                 "takeover repeats identically across all eight norms. Only candidates "
                 "whose `exact P` is small have evidence of a genuine takeover rate.")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"[written] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
