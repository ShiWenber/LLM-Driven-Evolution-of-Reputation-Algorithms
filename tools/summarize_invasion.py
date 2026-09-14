"""Summarise ``run_invasion`` sweeps into the compact tables INVASION_SOP §6.1 asks for.

INVASION_SOP §6.4 forbids report numbers that only a throwaway ``tmp/`` script can
reproduce, so this lives in ``tools/`` next to ``verify_sop_acceptance.py``. It
reads one or more ``summary.json`` files and prints (and optionally writes)
Markdown tables derived *only* from the recorded ``groups`` cells
``groups[candidate][resident][count] = {runs, fixations, extinctions,
mean_final_invader_frequency}``.

Definitions used by the per-candidate table (fixed here so the report and the
tool cannot drift apart):

``low_mean``
    Mean final invader share over the *low* starting counts (default 1-6) and all
    residents. A near-zero value means the candidate dies out before establishing
    itself; this is the column that separates threshold invaders from coexisters.
``at k``  (k = 1, 8, 19)
    Mean final invader share over all residents when the sweep starts from exactly
    ``k`` invaders out of N.
``fixations`` / ``extinctions``
    Totals over every (resident, count, seed) cell in the sweep.
``swept_from``
    Per resident, the *smallest* starting count at which the candidate reached a
    final share >= ``--fixed-threshold`` (default 0.999). Reporting the count
    matters: a candidate that only reaches 1.00 when it already starts at 19/20
    has merely held on to a majority, which is not an invasion. The listed count
    is therefore the invasion threshold, and ``19`` (or a resident absent from
    the list) means "no invasion observed at any starting count".

The ``l1-l8 spread`` column is a sanity check, not a result: several earlier
sweeps found the eight leading-eight norms indistinguishable, which means they do
not supply eight independent replicates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _counts(summary: dict[str, Any]) -> list[int]:
    return sorted(int(c) for c in next(iter(next(iter(summary["groups"].values())).values())))


def _residents(summary: dict[str, Any]) -> list[str]:
    return list(next(iter(summary["groups"].values())))


def cell_mean(summary: dict[str, Any], cand: str, res: str, count: int) -> float:
    cell = summary["groups"][cand][res].get(str(count))
    return float(cell["mean_final_invader_frequency"]) if cell else float("nan")


def candidate_row(summary: dict[str, Any], cand: str, low_counts: list[int],
                  fixed_threshold: float) -> dict[str, Any]:
    residents = _residents(summary)
    counts = _counts(summary)

    def mean_over(res_list, cnt_list) -> float:
        vals = [
            cell_mean(summary, cand, r, c)
            for r in res_list
            for c in cnt_list
            if cell_mean(summary, cand, r, c) == cell_mean(summary, cand, r, c)
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    low = mean_over(residents, [c for c in counts if c in low_counts])
    at = {k: mean_over(residents, [k]) for k in (1, 8, 19) if k in counts}

    fix = ext = 0
    for r in residents:
        for c in counts:
            cell = summary["groups"][cand][r].get(str(c))
            if cell:
                fix += int(cell["fixations"])
                ext += int(cell["extinctions"])

    swept: dict[str, int] = {}
    for r in residents:
        hit = [c for c in counts
               if cell_mean(summary, cand, r, c) >= fixed_threshold]
        if hit:
            swept[r] = min(hit)

    lead = [r for r in residents if r.startswith("L") and r[1:].isdigit()]
    lead_vals = [mean_over([r], counts) for r in lead]
    spread = max(lead_vals) - min(lead_vals) if lead_vals else float("nan")

    return {
        "candidate": cand,
        "low_mean": low,
        "at_1": at.get(1, float("nan")),
        "at_8": at.get(8, float("nan")),
        "at_19": at.get(19, float("nan")),
        "fixations": fix,
        "extinctions": ext,
        "swept_from": swept,
        "l1_l8_spread": spread,
    }


def render(summary: dict[str, Any], low_counts: list[int], fixed_threshold: float,
           arm_of=None) -> str:
    counts = _counts(summary)
    rows = [candidate_row(summary, c, low_counts, fixed_threshold)
            for c in summary["groups"]]

    lines: list[str] = []
    lines.append(f"<!-- {summary.get('experiment')} | N={summary.get('population_size')} "
                 f"| interactions={summary.get('interactions_per_generation', 'n/a')} "
                 f"| ae={summary.get('action_error_probability')} "
                 f"oe={summary.get('observation_error_probability')} "
                 f"| runs={summary.get('completed_or_cached_runs')} -->")
    lines.append("")
    lines.append("| candidate | low-start mean | from 1/N | from 8/N | from 19/N | fix/ext | swept to 1.00 from count (per resident) | L1-L8 spread |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|")
    for r in rows:
        swept = ", ".join(f"{k}@{v}" for k, v in sorted(r["swept_from"].items())) or "—"
        lines.append(
            f"| {r['candidate']} | {r['low_mean']:.3f} | {r['at_1']:.3f} | "
            f"{r['at_8']:.3f} | {r['at_19']:.3f} | {r['fixations']} / {r['extinctions']} | "
            f"{swept} | {r['l1_l8_spread']:.3f} |"
        )

    if arm_of is not None:
        arms: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            arms.setdefault(arm_of(r["candidate"]), []).append(r)
        lines += ["", "**Per-arm aggregate**", "",
                  "| arm | candidates | mean low-start | mean from 1/N | mean from 19/N | mean fix/ext | residents ever swept (min count seen) |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        for arm, rs in arms.items():
            n = len(rs)
            best: dict[str, int] = {}
            for r in rs:
                for rname, c in r["swept_from"].items():
                    best[rname] = min(best.get(rname, 10 ** 9), c)
            swept_any = [f"{k}@{v}" for k, v in sorted(best.items())]
            lines.append(
                f"| {arm} | {n} | "
                f"{sum(r['low_mean'] for r in rs)/n:.3f} | "
                f"{sum(r['at_1'] for r in rs)/n:.3f} | "
                f"{sum(r['at_19'] for r in rs)/n:.3f} | "
                f"{sum(r['fixations'] for r in rs)/n:.0f} / {sum(r['extinctions'] for r in rs)/n:.0f} | "
                f"{', '.join(swept_any) if swept_any else '—'} |"
            )

    residents = _residents(summary)
    lines += ["", "**Per-resident aggregate (mean final invader share across candidates)**", "",
              "| resident | " + " | ".join(f"{c}/N" for c in counts) + " |",
              "|---|" + "---:|" * len(counts)]
    for res in residents:
        vals = []
        for c in counts:
            v = [cell_mean(summary, cand, res, c) for cand in summary["groups"]]
            v = [x for x in v if x == x]
            vals.append(f"{sum(v)/len(v):.3f}" if v else "—")
        lines.append(f"| {res} | " + " | ".join(vals) + " |")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", type=Path, nargs="+", required=True)
    ap.add_argument("--low-counts", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                    help="starting counts treated as 'low' (default 1-6)")
    ap.add_argument("--fixed-threshold", type=float, default=0.999,
                    help="final share counted as 'swept to 1.00' (default 0.999)")
    ap.add_argument("--arm-prefix", default=None,
                    help="split candidate labels on this prefix to build an arm "
                         "aggregate, e.g. 'seed' turns 'ae0p1_seed3' into arm 'ae0p1'")
    ap.add_argument("--output", type=Path, default=None, help="write Markdown here as well")
    args = ap.parse_args()

    arm_of = None
    if args.arm_prefix:
        pref = args.arm_prefix
        arm_of = lambda c: c.split(pref)[0].rstrip("_") if pref in c else c

    chunks = []
    for path in args.summary:
        summary = load_summary(path)
        chunks.append(f"## `{path}`\n")
        chunks.append(render(summary, args.low_counts, args.fixed_threshold, arm_of))
    text = "\n".join(chunks)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"[written] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
