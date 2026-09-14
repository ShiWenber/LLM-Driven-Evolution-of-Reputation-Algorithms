"""Characterise an extracted evolved strategy against the leading eight.

Six views, all from the recorded artefacts (no new simulation):

0. **Structural audit.** Dead branches, whether ``decide`` reduces to a minimal
   form, whether the assessment sign ever depends on reputation, and whether the
   clamp binds. Catches LLM-generated logic that does not do what it appears to.
1. **Decision truth table** (my_good, opp_good) -> C/D, beside L1-L8.
2. **Assessment truth table** (action, A_good, B_good) -> G/B, beside L1-L8.
3. **Hamming distance** to each leading-eight norm, so "which norm is this?"
   has a number rather than an impression.
4. **Decision grid** over (my_reputation, opponent_reputation).
5. **Behavioural profile:** mean cooperation rate against each resident, read
   from the existing invasion runs' trajectories.

Output is ASCII-safe so it runs on a GBK console; the surrounding narrative is
kept in English for the same reason. The generated Markdown tables are plain
ASCII too, which keeps them diffable.

Terminology note: an "assessment is first-order" means the *sign* of the
reputation update depends only on the action, not on the actor's or the
recipient's standing. Leading-eight norms are second-order (standing) norms:
their table entries in the GDG/GDB/BDG/BDB columns differ precisely because
justified defection against a bad recipient is judged GOOD.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v2_quantitative.baselines import (  # noqa: E402
    ACTION_TABLES, ASSESSMENT_TABLES, BASELINE_VERSION,
)


def load_strategy(path: Path) -> dict:
    ns: dict = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    return ns


def action_table(ns) -> str:
    """(my_good, opp_good) -> C/D, as a 4-char string in GG, GB, BG, BB order."""
    out = []
    for my_rep in (1.0, -1.0):
        for opp_rep in (1.0, -1.0):
            out.append("C" if ns["decide"](my_rep, opp_rep) else "D")
    return "".join(out)


def raw_delta(ns, action: str, b_good: bool, b_rep_scale: float = 1.0) -> float:
    """Net assessment change, probed at A_rep = 0.

    Probing at 0 matters: a multiplicative decay (``new = (A_rep + delta) * k``)
    would otherwise make a delta of exactly 0 read as BAD, because the decay
    shrinks a positive starting score. That is mean reversion, not an assessment.
    """
    b_rep = b_rep_scale if b_good else -b_rep_scale
    return ns["observe"](0.0, action, b_rep, "cooperate", 0.0) - 0.0


def assessment_table(ns, b_rep_scale: float = 1.0) -> tuple[str, list[float]]:
    """(action, A_good, B_good) -> sign of the assessment, 8-char string.

    Row order matches ``ASSESSMENT_TABLES``: GCG, GCB, BCG, BCB, GDG, GDB, BDG, BDB.
    A_rep is held at zero to isolate the assessment from decay and from the
    starting score; whether the strategy also conditions on A_rep is reported
    separately by ``assess_depends_on_a_rep``.
    """
    out, deltas = [], []
    for action in ("cooperate", "defect"):
        for _a_good in (True, False):
            for b_good in (True, False):
                delta = raw_delta(ns, action, b_good, b_rep_scale)
                deltas.append(delta)
                out.append("G" if delta > 1e-12 else ("B" if delta < -1e-12 else "0"))
    return "".join(out), deltas


def assess_order(ns) -> dict:
    """Classify the assessment in the standard first-/second-order sense.

    FIRST-order (image scoring): the sign of the update depends only on the
    action. SECOND-order: it also depends on the RECIPIENT's standing, so
    defection against a bad recipient can be judged GOOD. That is the defining
    test, and it is measured over B_rep here.

    ``A_rep`` dependence is reported separately because it is a different
    property (self-consistency / self-image), not the first-/second-order axis.
    """
    b_rep_flips, a_rep_flips = [], []
    for a_action in ("cooperate", "defect"):
        for b_action in ("cooperate", "defect"):
            signs = set()
            for i in range(-10, 11):
                d = ns["observe"](0.0, a_action, i / 10, b_action, 0.0)
                signs.add("+" if d > 1e-12 else ("-" if d < -1e-12 else "0"))
            if len(signs) > 1:
                b_rep_flips.append((a_action, b_action, sorted(signs)))
            for b_rep in (-0.3, 0.3):
                lo = ns["observe"](-0.9, a_action, b_rep, b_action, 0.0) - (-0.9)
                hi = ns["observe"](0.9, a_action, b_rep, b_action, 0.0) - 0.9
                if (lo > 1e-12) != (hi > 1e-12):
                    a_rep_flips.append((a_action, b_action, b_rep))
    return {
        "second_order": bool(b_rep_flips),
        "b_rep_flipping_cells": b_rep_flips,
        "a_rep_dependence": bool(a_rep_flips),
    }


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def decision_map(ns, lo: float, hi: float, steps: int = 41):
    grid = []
    for i in range(steps):
        my = lo + (hi - lo) * i / (steps - 1)
        row = []
        for j in range(steps):
            opp = lo + (hi - lo) * j / (steps - 1)
            row.append("C" if ns["decide"](my, opp) else ".")
        grid.append((my, row))
    return grid


def structural_audit(ns) -> dict:
    """Probe the code for dead branches, sign stability and binding clamps."""
    decide = ns["decide"]
    grid = [i / 100 for i in range(-100, 101)]

    minimal_mismatch = 0
    for my in grid:
        for opp in grid:
            expected = True if opp >= 0.02 else (False if opp <= -0.08 else my >= -0.20)
            if decide(my, opp) != expected:
                minimal_mismatch += 1

    order = assess_order(ns)

    # Does the assessment use the recipient's ACTION, which the leading-eight
    # tables do not parameterise?
    uses_b_action = False
    for a_action in ("cooperate", "defect"):
        for b_rep in (-0.3, 0.3):
            v1 = ns["observe"](0.0, a_action, b_rep, "cooperate", 0.0)
            v2 = ns["observe"](0.0, a_action, b_rep, "defect", 0.0)
            if abs(v1 - v2) > 1e-12:
                uses_b_action = True

    return {
        "minimal_form_mismatches": minimal_mismatch,
        "minimal_form_grid": len(grid) ** 2,
        "second_order": order["second_order"],
        "b_rep_flipping_cells": order["b_rep_flipping_cells"],
        "a_rep_dependence": order["a_rep_dependence"],
        "uses_b_action": uses_b_action,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strategy", type=Path, required=True)
    ap.add_argument("--invasion-dir", type=Path, default=None,
                    help="invasion sweep dir; adds the empirical cooperation profile")
    ap.add_argument("--candidate-label", default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    ns = load_strategy(args.strategy)
    L: list[str] = []
    L.append(f"<!-- strategy analysis: {args.strategy.name} | norms {BASELINE_VERSION} -->")
    L.append("")

    audit = structural_audit(ns)
    at = action_table(ns)
    asmt, deltas = assessment_table(ns)

    L.append("### 0. Structural audit")
    L.append("")
    L.append(f"- Norm order (B_rep flips the assessment sign?): "
             f"**{'SECOND-ORDER' if audit['second_order'] else 'first-order'}**")
    if audit["b_rep_flipping_cells"]:
        L.append(f"  - cells where B_rep flips the sign: "
                 f"`{audit['b_rep_flipping_cells']}`")
    L.append(f"- Assessment uses the recipient's ACTION (B_action): "
             f"**{audit['uses_b_action']}**")
    L.append(f"- Assessment depends on A_rep (the actor's own standing): "
             f"{audit['a_rep_dependence']}")
    L.append(f"- `decide` reducible to the 3-line form: "
             f"**{'yes' if audit['minimal_form_mismatches'] == 0 else 'no'}** "
             f"({audit['minimal_form_mismatches']} / "
             f"{audit['minimal_form_grid']} grid points differ)")
    L.append("- Minimal equivalent `decide`: "
             "`opp >= 0.02 -> C; opp <= -0.08 -> D; else my >= -0.20`")
    L.append("")
    L.append("> The leading-eight tables are parameterised by "
             "`(action, A_good, B_good)`. If the strategy also uses `B_action`, "
             "it lives in a LARGER space and the Hamming distance to those tables "
             "is indicative only.")
    L.append("")

    L.append("### 1. Decision table (my_good, opp_good) -> C/D")
    L.append("")
    L.append("| strategy | GG | GB | BG | BB |")
    L.append("|---|---|---|---|---|")
    L.append("| **extracted** | " + " | ".join(at) + " |")
    for name in sorted(ACTION_TABLES):
        L.append(f"| {name} | " + " | ".join(ACTION_TABLES[name]) + " |")
    L.append("")

    L.append("### 2. Assessment table (action, A_good, B_good) -> G/B")
    L.append("")
    L.append("Row order as in `ASSESSMENT_TABLES`: GCG, GCB, BCG, BCB, GDG, GDB, BDG, BDB.")
    L.append("")
    L.append("| strategy | GCG | GCB | BCG | BCB | GDG | GDB | BDG | BDB |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    L.append("| **extracted** | " + " | ".join(asmt) + " |")
    for name in sorted(ASSESSMENT_TABLES):
        L.append(f"| {name} | " + " | ".join(ASSESSMENT_TABLES[name]) + " |")
    L.append("")
    L.append("Raw delta magnitude (A_rep = 0, |B_rep| = 0.5):")
    L.append("")
    L.append("| action | B_good | delta |")
    L.append("|---|---|---:|")
    idx = 0
    for action in ("cooperate", "defect"):
        for _ in (True, False):
            for b_good in (True, False):
                L.append(f"| {action} | {b_good} | {deltas[idx]:+.4f} |")
                idx += 1
    L.append("")

    L.append("### 3. Hamming distance to each norm (smaller = more similar)")
    L.append("")
    L.append("| norm | decision distance | assessment distance | total |")
    L.append("|---|---:|---:|---:|")
    ranked = []
    for name in sorted(ACTION_TABLES):
        da = hamming(at, ACTION_TABLES[name])
        dd = hamming(asmt, ASSESSMENT_TABLES[name])
        ranked.append((da + dd, da, dd, name))
    for tot, da, dd, name in sorted(ranked):
        L.append(f"| {name} | {da} / 4 | {dd} / 8 | **{tot}** |")
    best = sorted(ranked)[0]
    L.append("")
    L.append(f"> Nearest: **{best[3]}** (total {best[0]}). The strategy uses continuous "
             "thresholds and a decaying score, so it is not the same parametrisation "
             "as the binary good/bad tables; treat the distance as indicative only.")
    L.append("")

    L.append("### 4. Decision grid (C = cooperate)")
    L.append("")
    L.append("Rows = my_reputation, columns = opponent_reputation, both -1 to +1.")
    L.append("")
    L.append("```")
    for my, row in decision_map(ns, -1.0, 1.0, 41):
        L.append(f"my={my:+.2f} " + "".join(row))
    L.append("```")

    if args.invasion_dir and args.candidate_label:
        coop: dict[str, list[float]] = collections.defaultdict(list)
        for f in args.invasion_dir.rglob("invasion.json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            if d["candidate_label"] != args.candidate_label:
                continue
            cr = [g["cooperation_rate"] for g in d["trajectory"]
                  if g.get("cooperation_rate") is not None]
            if cr:
                coop[d["resident_label"]].append(sum(cr) / len(cr))
        if coop:
            L.append("")
            L.append("### 5. Behavioural profile: mean cooperation vs each resident")
            L.append("")
            L.append("| resident | cooperation | runs |")
            L.append("|---|---:|---:|")
            for res in sorted(coop):
                vals = coop[res]
                L.append(f"| {res} | {sum(vals) / len(vals):.4f} | {len(vals)} |")
            allv = [v for vs in coop.values() for v in vs]
            L.append(f"| **all** | **{sum(allv) / len(allv):.4f}** | {len(allv)} |")

    text = "\n".join(L) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"[written] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
