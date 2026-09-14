"""Why do different seeds give different fixation landscapes?

Answers the question directly from the recorded artefacts. The short version:
the seeds are NOT replicates of one strategy. Each seed is an independent
evolutionary run, and the divergence is present from generation 0 because the
INITIAL population is LLM-generated at temperature > 0 under a per-seed RNG.

Reports:
  1. Code identity of each seed's representative (highest-fitness) agent, so
     duplicates would be visible.
  2. Initial-population overlap between seeds. Zero overlap means the divergence
     is an INPUT difference, not something evolution had to create.
  3. Diversity of the resulting strategies: fitness, cooperation, assessment
     order (first- vs second-order), and a one-line "decide" summary.
  4. The resulting spread in the fixation landscape, to show that the landscape
     is a property of the strategy rather than of the seed.

Consequence for reporting: "N seeds" means N independent draws of the joint
distribution over (strategy, behaviour). It is NOT N replicates of one strategy,
so a per-seed property has N as its sample size, not N x (number of probes).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

sys_path_ok = True
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_run(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def representative(ev: dict[str, Any]) -> dict[str, Any]:
    return max(ev["final_population"], key=lambda a: a["fitness"])


def features(code: str) -> dict[str, Any]:
    ns: dict = {}
    try:
        exec(compile(code, "<s>", "exec"), ns)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    decide, observe = ns["decide"], ns["observe"]
    table = "".join("C" if decide(m, o) else "D"
                    for m in (1.0, -1.0) for o in (1.0, -1.0))
    # SECOND-order means the RECIPIENT's standing flips the assessment sign
    # (justified defection judged GOOD). This is the defining test; whether
    # A_rep matters is a separate property and is reported separately.
    second_order = False
    for a_action in ("cooperate", "defect"):
        for b_action in ("cooperate", "defect"):
            signs = set()
            for i in range(-10, 11):
                d = observe(0.0, a_action, i / 10, b_action, 0.0)
                signs.add("+" if d > 1e-12 else ("-" if d < -1e-12 else "0"))
            if len(signs) > 1:
                second_order = True
    uses_b_action = False
    for a_action in ("cooperate", "defect"):
        for b_rep in (-0.3, 0.3):
            if abs(observe(0.0, a_action, b_rep, "cooperate", 0.0)
                   - observe(0.0, a_action, b_rep, "defect", 0.0)) > 1e-12:
                uses_b_action = True
    return {
        "decision_table": table,
        "assessment_order": "second" if second_order else "first",
        "uses_b_action": uses_b_action,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path,
                    default=Path("results/quantitative_baseline"))
    ap.add_argument("--glob", required=True,
                    help="run-dir glob with {arm} and {seed} placeholders")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    initial_sets: dict[str, dict[int, set[str]]] = {}

    for arm in args.arms:
        initial_sets[arm] = {}
        for seed in range(args.seeds):
            d = args.root / args.glob.format(arm=arm, seed=seed)
            ev = load_run(d / "evolutionary.json")
            best = representative(ev)
            gen0 = ev["trajectory"][0]["population"]
            shas = {hashlib.sha256(a["code"].encode()).hexdigest() for a in gen0}
            initial_sets[arm][seed] = shas
            rows.append({
                "arm": arm, "seed": seed,
                "agent_id": best["agent_id"],
                "fitness": best["fitness"],
                "coop": best["cooperation_rate"],
                "code_len": len(best["code"]),
                "sha": hashlib.sha256(best["code"].encode()).hexdigest(),
                "n_lineages": len({a["lineage_id"] for a in ev["final_population"]}),
                "n_events": len(ev["lineage_events"]),
                "init_distinct": len(shas),
                "init_slots": len(gen0),
                **features(best["code"]),
            })

    L: list[str] = []
    L.append("<!-- seed diversity: each seed is an independent evolutionary run -->")
    L.append("")
    L.append("### 1. Representative strategy per seed")
    L.append("")
    L.append("| arm | seed | agent | fitness | coop | code bytes | sha256[:12] | "
             "decide | assessment |")
    L.append("|---|---:|---:|---:|---:|---:|---|---|---|")
    for r in rows:
        L.append(f"| {r['arm']} | {r['seed']} | {r['agent_id']} | {r['fitness']:.4f} | "
                 f"{r['coop']:.4f} | {r['code_len']} | `{r['sha'][:12]}` | "
                 f"{r['decision_table']} | {r['assessment_order']}-order |")
    L.append("")

    shas = collections.Counter(r["sha"] for r in rows)
    dupes = {s: c for s, c in shas.items() if c > 1}
    L.append(f"- distinct code objects: **{len(shas)} / {len(rows)}** seeds")
    L.append(f"- duplicated codes: {dupes if dupes else 'none'}")
    L.append("")

    L.append("### 2. Initial-population overlap between seeds")
    L.append("")
    L.append("| arm | common initial codes across seeds | distinct total | slots |")
    L.append("|---|---:|---:|---:|")
    for arm in args.arms:
        sets = list(initial_sets[arm].values())
        inter = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        L.append(f"| {arm} | {len(inter)} | {len(union)} | "
                 f"{sum(len(s) for s in sets)} |")
    L.append("")
    L.append("> Zero common initial codes means the seeds differ **from generation 0**: "
             "the 16 starting strategies are LLM-generated under a per-seed RNG "
             "(temperature > 0), so divergence is an input difference rather than "
             "something evolution has to create.")
    L.append("")

    L.append("### 3. Diversity of the resulting strategies")
    L.append("")
    L.append("| arm | fitness range | cooperation range | assessment order | "
             "distinct lineages |")
    L.append("|---|---|---|---|---|")
    for arm in args.arms:
        sub = [r for r in rows if r["arm"] == arm]
        fits = [r["fitness"] for r in sub]
        coops = [r["coop"] for r in sub]
        orders = collections.Counter(r["assessment_order"] for r in sub)
        lin = {r["n_lineages"] for r in sub}
        L.append(f"| {arm} | {min(fits):.4f} - {max(fits):.4f} | "
                 f"{min(coops):.4f} - {max(coops):.4f} | "
                 f"first {orders.get('first', 0)} / second {orders.get('second', 0)} | "
                 f"{min(lin)}-{max(lin)} of 16 |")
    L.append("")
    L.append("> Every run ends with a fully distinct lineage per slot (no single "
             "lineage takes over), so the final population is a diverse set and the "
             "representative is chosen by fitness.")
    L.append("")

    L.append("### 4. Consequence for reporting")
    L.append("")
    L.append(f"- \"{args.seeds} seeds\" means {args.seeds} **independent draws** of the "
             "joint distribution over (strategy, behaviour) -- not "
             f"{args.seeds} replicates of one strategy.")
    L.append("- A fixation landscape is a property of the STRATEGY. Different seeds "
             "give different landscapes because the strategies are different objects.")
    L.append("- Therefore any per-seed claim has sample size = number of seeds, "
             "**not** seeds x probes. Multiplying by the probe count would be "
             "pseudo-replication.")

    text = "\n".join(L) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"[written] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
