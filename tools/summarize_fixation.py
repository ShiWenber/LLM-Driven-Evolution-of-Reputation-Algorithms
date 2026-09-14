"""Summarise ``run_fixation_benchmark`` output into INVASION_SOP §6.1-style tables.

INVASION_SOP §6.4 forbids report numbers that only a throwaway ``tmp/`` script can
reproduce, so this lives in ``tools/``. It reads one or more
``fixation_benchmark.json`` files and prints (and optionally writes) Markdown.

Verdict rule -- deliberately identical to ``run_fixation_benchmark._verdict`` so a
number quoted here matches what the resident runner prints on its own stdout:

    rho > neutral + tol   -> "invades >neutral"
    rho < neutral - tol   -> "blocked <neutral"
    otherwise             -> "neutral"      (neutral = 1/N)

``tol`` defaults to the runner's absolute ``0.01``. That value is *absolute*, so it
is a much larger fraction of the neutral line at small N (20 % of ``1/N`` at N=20,
vs 4 % at N=50). Raise ``--tol``-relative use via ``--tol`` if a tighter band is
needed; the report should state which rule was applied.

The stationarity audit is the other half of SOP §4.3: a probe whose
``max_half_to_half_gap`` exceeds ``--stationarity-threshold`` (default 0.10) has not
converged, and its ``rho`` must not be quoted as independent evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verdict(rho: float, neutral: float, tol: float) -> str:
    if rho > neutral + tol:
        return "invades >neutral"
    if rho < neutral - tol:
        return "blocked <neutral"
    return "neutral"


def rows_for(path: Path, tol: float, gap_threshold: float) -> list[dict[str, Any]]:
    data = load(path)
    neutral = float(data["config"]["neutral_fixation_probability"])
    label = path.parent.name
    rows = []
    for probe in data["results"]:
        entry = data["results"][probe]["candidate_invades_probe"]
        rho = float(entry["rho"])
        gap = float(
            data.get("stationarity", {}).get(probe, {}).get("max_half_to_half_gap", float("nan"))
        )
        rows.append(
            {
                "label": label,
                "probe": probe,
                "rho": rho,
                "neutral": neutral,
                "verdict": verdict(rho, neutral, tol),
                "gap": gap,
                "gap_ok": gap <= gap_threshold,
                "n": int(data["config"]["population_size"]),
                "ae": float(data["config"].get("action_error_probability", 0.0)),
                "oe": float(data["config"].get("observation_error_probability", 0.0)),
                "burn": int(data["config"].get("burn_in_interactions", 0)),
                "measure": int(data["config"].get("measure_interactions", 0)),
                "reps": int(data["config"].get("replicates", 0)),
            }
        )
    return rows


def _arm(label: str, arm_prefix: str | None) -> str:
    """Arm = candidate label with the per-seed marker removed.

    ``n20_noisestudy_ae0_seed3`` -> ``n20_noisestudy_ae0`` when arm_prefix="seed".
    """
    if not arm_prefix:
        return label
    marker = f"_{arm_prefix}"
    idx = label.rfind(marker)
    # rfind so an arm name that itself contains the marker still splits at the last one
    return label[:idx] if idx != -1 else label


def render(all_rows: list[dict[str, Any]], tol: float, gap_threshold: float,
           arm_prefix: str | None = None) -> str:
    lines: list[str] = []
    head = all_rows[0]
    lines.append(
        f"<!-- fixation benchmark: N={head['n']} ae={head['ae']} oe={head['oe']} "
        f"burn={head['burn']} measure={head['measure']} replicates={head['reps']} "
        f"| neutral=1/N={head['neutral']:.4f} | verdict tol={tol} "
        f"| stationarity threshold={gap_threshold} -->"
    )
    lines.append("")

    lines.append("| candidate | " + " | ".join(f"{r['probe']}" for r in all_rows if r["label"] == head["label"]) + " |")
    lines.append("|---|" + "---:|" * sum(1 for r in all_rows if r["label"] == head["label"]))
    for label in dict.fromkeys(r["label"] for r in all_rows):
        rs = [r for r in all_rows if r["label"] == label]
        lines.append(f"| {label} | " + " | ".join(f"{r['rho']:.4f}" for r in rs) + " |")

    lines += ["", f"**Verdict counts** (tol = {tol})", "",
              "| candidate | invades >neutral | neutral | blocked <neutral | probes w/ stationarity gap > "
              f"{gap_threshold} |", "|---|---:|---:|---:|---:|"]
    for label in dict.fromkeys(r["label"] for r in all_rows):
        rs = [r for r in all_rows if r["label"] == label]
        lines.append(
            f"| {label} | "
            f"{sum(1 for r in rs if r['verdict'].startswith('invades'))} | "
            f"{sum(1 for r in rs if r['verdict'] == 'neutral')} | "
            f"{sum(1 for r in rs if r['verdict'].startswith('blocked'))} | "
            f"{sum(1 for r in rs if not r['gap_ok'])} |"
        )

    probes = list(dict.fromkeys(r["probe"] for r in all_rows))
    lines += ["", "**Per-probe aggregate across candidates**", "",
              "| probe | mean rho | invades | neutral | blocked |", "|---|---:|---:|---:|---:|"]
    for probe in probes:
        rs = [r for r in all_rows if r["probe"] == probe]
        lines.append(
            f"| {probe} | {sum(r['rho'] for r in rs) / len(rs):.4f} | "
            f"{sum(1 for r in rs if r['verdict'].startswith('invades'))} | "
            f"{sum(1 for r in rs if r['verdict'] == 'neutral')} | "
            f"{sum(1 for r in rs if r['verdict'].startswith('blocked'))} |"
        )

    bad = [r for r in all_rows if not r["gap_ok"]]
    lines += ["", f"**Stationarity audit** — {len(all_rows) - len(bad)}/{len(all_rows)} "
                  f"candidate-probe pairs converge at threshold {gap_threshold}"]
    if bad:
        lines.append("")
        lines.append("| candidate | probe | max_half_to_half_gap |")
        lines.append("|---|---|---:|")
        for r in bad:
            lines.append(f"| {r['label']} | {r['probe']} | {r['gap']:.3f} |")
        lines.append("")
        lines.append("> A flagged pair has not reached a stationary payoff profile, so its "
                     "`rho` must not be quoted as independent evidence (SOP §7.2). "
                     "The `stationary-only` column below recomputes each arm's mean rho "
                     "after dropping every flagged pair.")

    if arm_prefix:
        lines += ["", f"**Per-arm aggregate** (arm marker `_{arm_prefix}<N>` stripped)", "",
                  "| arm | candidates | mean rho (all probes) | mean rho (stationary only) | "
                  "probes flagged | invades >neutral |", "|---|---:|---:|---:|---:|---:|"]
        for arm in dict.fromkeys(_arm(r["label"], arm_prefix) for r in all_rows):
            rs = [r for r in all_rows if _arm(r["label"], arm_prefix) == arm]
            clean = [r for r in rs if r["gap_ok"]]
            m_all = sum(r["rho"] for r in rs) / len(rs)
            m_clean = sum(r["rho"] for r in clean) / len(clean) if clean else float("nan")
            lines.append(
                f"| {arm} | {len({r['label'] for r in rs})} | {m_all:.4f} | {m_clean:.4f} | "
                f"{len(rs) - len(clean)} | "
                f"{sum(1 for r in rs if r['verdict'].startswith('invades'))} |"
            )

        arms = list(dict.fromkeys(_arm(r["label"], arm_prefix) for r in all_rows))
        lines += ["", "**Per-arm x per-probe mean rho**", "",
                  "| arm | " + " | ".join(probes) + " |", "|---|" + "---:|" * len(probes)]
        for arm in arms:
            cells = []
            for probe in probes:
                rs = [r for r in all_rows if _arm(r["label"], arm_prefix) == arm
                      and r["probe"] == probe]
                cells.append(f"{sum(r['rho'] for r in rs) / len(rs):.4f}" if rs else "-")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")

        lines += ["", "**Per-arm x per-probe mean rho (stationary probes only)**", "",
                  "| arm | " + " | ".join(probes) + " |", "|---|" + "---:|" * len(probes)]
        for arm in arms:
            cells, notes = [], []
            for probe in probes:
                rs = [r for r in all_rows if _arm(r["label"], arm_prefix) == arm
                      and r["probe"] == probe and r["gap_ok"]]
                if rs:
                    cells.append(f"{sum(r['rho'] for r in rs) / len(rs):.4f}")
                else:
                    cells.append("n/a")
                    notes.append(probe)
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
            if notes:
                lines.append(f"")
                lines.append(f"> `{arm}`: no stationary candidate for {', '.join(notes)}.")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", type=Path, nargs="+", required=True,
                    help="one or more fixation_benchmark.json files")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="absolute rho tolerance for the verdict (same default as the runner)")
    ap.add_argument("--stationarity-threshold", type=float, default=0.10,
                    help="max_half_to_half_gap above which a probe is flagged (SOP §4.3)")
    ap.add_argument("--arm-prefix", default=None,
                    help="candidate-label marker separating arms, e.g. 'seed' for `..._ae0_seed3`")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.summary:
        rows.extend(rows_for(path, args.tol, args.stationarity_threshold))
    text = render(rows, args.tol, args.stationarity_threshold, args.arm_prefix)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"[written] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
