"""Compare several fixation-benchmark runs of ONE candidate strategy.

Holds the strategy constant and varies the evaluation settings, which isolates
the effect of the varying factor from "the evolutionary process happened to
produce a different strategy". The latter is what confounds a cross-arm
comparison, because each arm's candidate is a different code object. The tool
refuses to compare files whose ``code_sha256`` differ.

Two groupings:

``--group-by noise`` (default)
    Rows are the measurement-noise levels. Use this to ask "how much of the
    advantage is supplied by noise?"

``--group-by dir``
    Rows are the run directories. Use this with several ``--seed`` values at a
    fixed noise to ask "is the plot reproducible, or does it move with the RNG
    seed?" The runner seeds each composition deterministically from
    ``--seed + 1000003*rep + k``, so a fresh ``--seed`` is a different
    realisation of the SAME measurement, not a different experiment.

Note on caching: ``run_fixation_benchmark.cache_matches`` deliberately omits
``seed`` from its key, so re-running into an existing output directory with a
different ``--seed`` silently returns the old numbers. Use a fresh ``--output``
or pass ``--force``.

Reports, per row: rho against every probe with the neutral 1/N reference, the
per-composition d(k) span and negative-composition count, and the stationarity
gap (a drifting profile makes rho unreadable). Reads only
``fixation_benchmark.json``, so every number is traceable to a resident runner.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def stats(path: Path, leading_eight_only: bool) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    cfg = d["config"]
    probes = list(d["results"])
    if leading_eight_only:
        probes = [p for p in probes if p.startswith("L") and p[1:].isdigit()]

    rhos, bands, neg, worst_gap, mn, mx = {}, {}, 0, 0.0, math.inf, -math.inf
    for p in probes:
        entry = d["results"][p]["candidate_invades_probe"]
        rhos[p] = entry["rho"]
        if "rho_at_minus_1sd" in entry and "rho_at_plus_1sd" in entry:
            bands[p] = (float(entry["rho_at_minus_1sd"]),
                        float(entry["rho_at_plus_1sd"]))
        for c in entry["curve"]:
            v = c["payoff_difference"]
            mn, mx = min(mn, v), max(mx, v)
            if v < 0:
                neg += 1
        worst_gap = max(
            worst_gap,
            d.get("stationarity", {}).get(p, {}).get("max_half_to_half_gap", 0.0),
        )

    return {
        "label": path.parent.name,
        "n": int(cfg["population_size"]),
        "ae": float(cfg.get("action_error_probability", 0.0)),
        "oe": float(cfg.get("observation_error_probability", 0.0)),
        "seed": cfg.get("seed"),
        "replicates": cfg.get("replicates"),
        "neutral": float(cfg["neutral_fixation_probability"]),
        "probes": probes,
        "rhos": rhos,
        "bands": bands,
        "neg_compositions": neg,
        "d_min": mn,
        "d_max": mx,
        "worst_gap": worst_gap,
        "candidate": d["candidate"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", type=Path, nargs="+", required=True,
                    help="fixation_benchmark.json files to compare")
    ap.add_argument("--leading-eight-only", action="store_true",
                    help="restrict the d(k) summary to L1-L8")
    ap.add_argument("--group-by", choices=("noise", "dir"), default="noise",
                    help="row label: measurement noise (default) or run directory")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    rows = [stats(p, args.leading_eight_only) for p in args.summary]
    if args.group_by == "noise":
        rows.sort(key=lambda r: (r["ae"], r["oe"]))
        tag = lambda r: f"{r['ae']:.0%} / {r['oe']:.0%}"  # noqa: E731
        head = "measurement noise"
    else:
        rows.sort(key=lambda r: r["label"])
        tag = lambda r: r["label"]  # noqa: E731
        head = "run"
    neutral = rows[0]["neutral"]
    probes = rows[0]["probes"]

    # Every file must refer to the same code object, or the comparison is void.
    cands = {r["candidate"]["code_sha256"] for r in rows}
    L: list[str] = []
    L.append(f"<!-- fixed candidate `{rows[0]['candidate']['label']}` | "
             f"grouped by {head} | probes={len(probes)} | neutral={neutral:.4f} "
             f"| distinct candidate hashes={len(cands)} -->")
    L.append("")
    if len(cands) != 1:
        L.append("> [WARNING] these files do NOT come from the same strategy "
                 f"({len(cands)} distinct code_sha256 values), so the comparison "
                 "confounds strategy identity with measurement noise.")
        L.append("")
    L.append(f"**candidate:** `{rows[0]['candidate']['label']}` "
             f"(`code_sha256 = {rows[0]['candidate']['code_sha256'][:16]}...`)")
    L.append("")

    L.append(f"### rho by {head}")
    L.append("")
    L.append(f"| {head} | " + " | ".join(probes) + " |")
    L.append("|---|" + "---:|" * len(probes))
    for r in rows:
        cell = " | ".join(f"{r['rhos'][p]:.4f}" for p in probes)
        L.append(f"| {tag(r)} | {cell} |")
    L.append(f"| (neutral 1/N) | " + " | ".join(f"{neutral:.4f}" for _ in probes) + " |")
    L.append("")

    L.append("### Ratio to neutral")
    L.append("")
    L.append(f"| {head} | " + " | ".join(probes) + " |")
    L.append("|---|" + "---:|" * len(probes))
    for r in rows:
        cell = " | ".join(f"{r['rhos'][p] / neutral:.2f}x" for p in probes)
        L.append(f"| {tag(r)} | {cell} |")
    L.append("")

    if args.group_by == "dir" and len(rows) > 1:
        L.append("### Spread across runs (is the plot reproducible?)")
        L.append("")
        L.append(f"| probe | min rho | max rho | range | reported-band half-width |")
        L.append("|---|---:|---:|---:|---:|")
        for p in probes:
            vals = [r["rhos"][p] for r in rows]
            bands = [r["bands"].get(p) for r in rows if r["bands"].get(p)]
            half = (max((b[1] - b[0]) / 2 for b in bands) if bands else float("nan"))
            L.append(f"| {p} | {min(vals):.4f} | {max(vals):.4f} | "
                     f"{max(vals) - min(vals):.4f} | {half:.4f} |")
        L.append("")
        L.append("> If the run-to-run range is comparable to the reported band, the plot "
                 "is a draw from the same distribution and only the SUMMARY is "
                 "stable. If the range is much smaller than the band, the point "
                 "estimate is a deterministic function of ``--seed`` and the band "
                 "is the honest uncertainty.")
        L.append("")

    L.append("### d(k) structure and stationarity")
    L.append("")
    L.append(f"| {head} | negative compositions | d min | d max | worst stat. gap |")
    L.append("|---|---:|---:|---:|---:|")
    for r in rows:
        flag = " (!)" if r["worst_gap"] > 0.10 else ""
        L.append(f"| {tag(r)} | {r['neg_compositions']} | "
                 f"{r['d_min']:+.4f} | {r['d_max']:+.4f} | {r['worst_gap']:.4f}{flag} |")
    L.append("")
    L.append("- A negative-composition count of 0 means the candidate out-earns the "
             "probe at every mixture, which under the scan's beta -> infinity rule "
             "would sweep.")
    L.append("- `(!)` marks a probe whose profile has not reached stationarity; its rho "
             "must not be quoted (see INVASION_SOP section 7.2).")
    L.append("- **Caveat:** an evaluation at 0% noise sits in the all-good steady state "
             "that INVASION_SOP section 0.4 warns about, so that row is a control, not a "
             "result.")

    text = "\n".join(L) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"[written] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
