"""Summarise evolved-arm cooperation trajectories from ``evolutionary.json``.

INVASION_SOP §6.4 requires every number in a report to trace to a *resident*
command, so the per-arm cooperation context used in the invasion/fixation report
is produced here rather than by a throwaway ``tmp/`` script.

Reads the ``trajectory`` list of each ``evolutionary.json`` (field
``cooperation_rate_mean``, one entry per generation) and reports the final
generation, the minimum over the run, and the arm-level mean/range.

Note on the metric name: ``config.cooperation_metric`` currently reads
``both_players_per_joint_action_v1`` but the implementation counts a cooperative
*action* per player, so the value is a per-action cooperation rate. The config
name is a known mislabel and is not used here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def arm_stats(dirs: list[Path]) -> list[dict]:
    out = []
    for d in dirs:
        ev = json.loads((d / "evolutionary.json").read_text(encoding="utf-8"))
        coop = [g["cooperation_rate_mean"] for g in ev["trajectory"]]
        out.append(
            {
                "label": d.name,
                "seed": d.name.rsplit("_seed", 1)[-1],
                "n_generations": len(coop),
                "final": coop[-1],
                "min": min(coop),
                "mean": sum(coop) / len(coop),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("results/quantitative_baseline"))
    ap.add_argument("--glob", required=True,
                    help="glob under --root containing a '{arm}' placeholder, e.g. "
                         "'LLM_agent-type1_fermi_g100_*_{arm}_5seed_seed*'")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    lines: list[str] = []
    for arm in args.arms:
        dirs = sorted(args.root.glob(args.glob.format(arm=arm)))
        stats = arm_stats(dirs)
        if not stats:
            lines.append(f"| {arm} | - | - | - | - | (no runs found) |")
            continue
        finals = [s["final"] for s in stats]
        lines.append(
            f"| {arm} | {len(stats)} | {sum(finals) / len(finals):.4f} | "
            f"{min(finals):.4f} - {max(finals):.4f} | "
            f"{min(s['min'] for s in stats):.4f} | "
            + ", ".join(f"seed{s['seed']}={s['final']:.4f}" for s in stats) + " |"
        )

    header = ("| arm | seeds | mean final coop | final range | worst mid-run | per-seed final |\n"
              "|---|---:|---:|---|---:|---|")
    text = header + "\n" + "\n".join(lines) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"[written] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
