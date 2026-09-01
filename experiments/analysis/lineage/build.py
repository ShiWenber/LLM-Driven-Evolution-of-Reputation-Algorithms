"""Build evolutionary-tree data from the framework's recorded lineage events.

Reads an evolution-log record (``experiments.evolution_log``, schema v4:
``lineage_events`` plus per-agent ``lineage_id`` / ``parent_id`` /
``parent_lineage_id`` / ``origin`` / ``birth_gen`` fields) and derives the
two representations needed for visualization:

  1. Full birth-event tree (definition A): every birth is a node, edges are
     `parent_lineage_id -> lineage_id`.
  2. Collapsed lineage forest (definition B): a "lineage" is a maximal chain
     of `imitate` births rooted at an `initial` or `independent_init` node.
     Consecutive small mutations collapse into one persistent lineage; only a
     fully-rewritten strategy (independent_init) starts a new lineage.

Also derives each lineage's death generation (last generation it was observed
in any slot) and the ancestry path of every final survivor back to its root.

Usage:
  uv run python -m experiments.analysis.lineage.build --json results/.../evolutionary.json --out results/.../lineage.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from experiments.evolution_log import (
    F_AGENT_ID, F_BIRTH_GEN, F_CONFIG_SCHEMA_VERSION, F_GENERATION,
    F_LINEAGE_ID, F_ORIGIN, F_PARENT_LINEAGE_ID, F_POPULATION,
    K_FINAL_POPULATION, K_LINEAGE_EVENTS, K_TRAJECTORY, load_evolution_json,
)


def build_lineage_tree(data: dict) -> dict:
    """Derive full-tree + collapsed-forest + survival data from a log record."""
    events: List[dict] = data.get(K_LINEAGE_EVENTS, [])
    trajectory: List[dict] = data.get(K_TRAJECTORY, [])
    final_population: List[dict] = data.get(K_FINAL_POPULATION, [])

    # --- root lookup: lineage_id -> root lineage_id (collapse imitate chains)
    parent_of: Dict[int, Optional[int]] = {}
    root_of: Dict[int, int] = {}
    for e in events:
        parent_of[e[F_LINEAGE_ID]] = e.get(F_PARENT_LINEAGE_ID)

    def _root(lid: int) -> int:
        # iterative upward walk to the origin (memoized per call site)
        seen = set()
        cur = lid
        while parent_of.get(cur) is not None and cur not in seen:
            seen.add(cur)
            cur = parent_of[cur]
        return cur

    for e in events:
        root_of[e[F_LINEAGE_ID]] = _root(e[F_LINEAGE_ID])

    # --- last-seen generation for each lineage_id (scan trajectory)
    last_seen: Dict[int, int] = {}
    for g in trajectory:
        gen = g[F_GENERATION]
        for a in g.get(F_POPULATION, []):
            lid = a.get(F_LINEAGE_ID)
            if lid is not None:
                last_seen[lid] = max(last_seen.get(lid, gen), gen)
    # final population is the last generation; ensure covered
    final_gen = trajectory[-1][F_GENERATION] if trajectory else 0
    for a in final_population:
        lid = a.get(F_LINEAGE_ID)
        if lid is not None:
            last_seen[lid] = max(last_seen.get(lid, final_gen), final_gen)

    # --- collapse to persistent lineages (definition B)
    lineages: Dict[int, dict] = {}
    for e in events:
        lid = e[F_LINEAGE_ID]
        root = root_of[lid]
        lin = lineages.setdefault(root, {
            "root_lineage_id": root,
            "birth_gen": None,
            "death_gen": None,
            "origin": None,
            "members": [],
            "member_origins": [],
        })
        if lin["birth_gen"] is None or e[F_BIRTH_GEN] < lin["birth_gen"]:
            lin["birth_gen"] = e[F_BIRTH_GEN]
        if lin["origin"] is None:
            # origin of the root event (initial / independent_init)
            lin["origin"] = e[F_ORIGIN]
        lin["members"].append(lid)
        lin["member_origins"].append(e[F_ORIGIN])

    for root, lin in lineages.items():
        member_last = [last_seen.get(m, lin["birth_gen"]) for m in lin["members"]]
        lin["death_gen"] = max(member_last) if member_last else lin["birth_gen"]
        lin["n_members"] = len(lin["members"])

    # --- ancestry path of each final survivor (back to root)
    survivors: List[dict] = []
    for a in final_population:
        lid = a.get(F_LINEAGE_ID)
        path = []
        cur = lid
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            path.append(cur)
            cur = parent_of.get(cur)
        survivors.append({
            F_AGENT_ID: a.get(F_AGENT_ID),
            F_LINEAGE_ID: lid,
            "root_lineage_id": root_of.get(lid),
            "path": path,  # leaf -> root order
        })

    return {
        "events": events,
        "parent_of": {str(k): v for k, v in parent_of.items()},
        "root_of": {str(k): v for k, v in root_of.items()},
        "lineages": {str(k): v for k, v in lineages.items()},
        "survivors": survivors,
        "n_events": len(events),
        "n_lineages": len(lineages),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=str, default=None, help="path to schema-v4 evolutionary.json")
    ap.add_argument("--out", type=str, default=None, help="output path for lineage.json")
    args = ap.parse_args()

    if not args.json:
        raise SystemExit("provide --json")

    json_path = Path(args.json)
    data = load_evolution_json(json_path)
    if data.get("config", {}).get(F_CONFIG_SCHEMA_VERSION, 0) < 4:
        raise SystemExit(
            f"{json_path} is schema {data.get('config', {}).get(F_CONFIG_SCHEMA_VERSION)}, "
            f"but lineage fields require schema >= 4. Re-run the evolution with "
            f"the updated framework."
        )
    tree = build_lineage_tree(data)
    out = Path(args.out) if args.out else json_path.with_name("lineage.json")
    out.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"n_events={tree['n_events']}, n_lineages={tree['n_lineages']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
