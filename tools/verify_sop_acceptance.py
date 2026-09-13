"""SOP §8 acceptance check for a completed Fermi analysis output set.

Verifies (see ``ANALYSIS_SOP.md`` §8):

  1. each seed directory contains *exactly* the 10 expected items (8 figures + 2 data files)
  2. every produced file is non-empty
  3. ``lineage.json`` is self-consistent with ``evolutionary.json``
     (schema v4, ``n_events`` == len(``lineage_events``), ``n_lineages`` > 0)
  4. the analysis summary directory holds ``README.md`` + evolution curves + ``joint_*`` output
  5. the clustering cache can be queried by ``run_id`` for cluster names and final composition

Usage::

    uv run python tools/verify_sop_acceptance.py
    uv run python tools/verify_sop_acceptance.py --run-label LLM_..._seed --seeds 0 1 2 3 4 \
        --analysis-dir results/quantitative_baseline/analysis_...

Exit code is 0 when all checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

# Expected per-seed artefacts (SOP §8.1).
EXPECTED_PER_SEED = {
    "plot_strategy_cluster_composition_per_generation.png",
    "plot_strategy_pca_evolution.gif",
    "plot_strategy_pca_evolution.mp4",
    "final_strategy_clusters_pca.png",
    "strategy_dendrogram.png",
    "lineage_survival.png",
    "full_birth_event_tree.png",
    "final_survivor_ancestry_tree.png",
    "lineage.json",
    "evolutionary.json",
}

DEFAULT_ROOT = Path("results/quantitative_baseline")
DEFAULT_CACHE = Path("results/.analysis_cache/strategy_analysis.sqlite3")


class Checker:
    """Accumulates pass/fail results and prints a report."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, msg: str) -> None:
        print(("  [OK]   " if ok else "  [FAIL] ") + msg)
        if not ok:
            self.failures.append(msg)

    def section(self, title: str) -> None:
        print(f"\n== {title} ==")


def check_seed_dirs(chk: Checker, root: Path, label: str, seeds: list[int]) -> None:
    chk.section("§8.1  per-seed dir: exactly the 10 expected items")
    for s in seeds:
        d = root / f"{label}_seed{s}"
        if not d.is_dir():
            chk.check(False, f"seed{s}: directory missing ({d})")
            continue
        items = {p.name for p in d.iterdir()}
        extra = sorted(items - EXPECTED_PER_SEED)
        missing = sorted(EXPECTED_PER_SEED - items)
        chk.check(
            not extra and not missing,
            f"seed{s}: {len(items)} items"
            + (f"  MISSING={missing}" if missing else "")
            + (f"  EXTRA={extra}" if extra else ""),
        )


def check_files_nonempty(chk: Checker, root: Path, label: str, seeds: list[int]) -> None:
    chk.section("§8.2  all produced files are non-empty")
    for s in seeds:
        d = root / f"{label}_seed{s}"
        if not d.is_dir():
            continue
        empty = [p.name for p in d.iterdir() if p.is_file() and p.stat().st_size == 0]
        chk.check(not empty, f"seed{s}: no zero-byte files" + (f"  EMPTY={empty}" if empty else ""))


def check_lineage(chk: Checker, root: Path, label: str, seeds: list[int]) -> None:
    chk.section("§8.3  lineage.json self-consistent with evolutionary.json")
    for s in seeds:
        d = root / f"{label}_seed{s}"
        if not d.is_dir():
            continue
        try:
            lin = json.loads((d / "lineage.json").read_text(encoding="utf-8"))
            evo = json.loads((d / "evolutionary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            chk.check(False, f"seed{s}: read/parse error: {exc}")
            continue
        n_events = len(evo.get("lineage_events", []))
        schema = evo.get("config", {}).get("schema_version")
        chk.check(schema == 4, f"seed{s}: schema_version={schema}")
        chk.check(n_events > 0, f"seed{s}: lineage_events={n_events} (non-empty)")
        chk.check(
            lin.get("n_events") == n_events,
            f"seed{s}: lineage.json n_events={lin.get('n_events')} == events={n_events}",
        )
        chk.check(
            isinstance(lin.get("n_lineages"), int) and lin["n_lineages"] > 0,
            f"seed{s}: n_lineages={lin.get('n_lineages')}",
        )


def check_analysis_dir(chk: Checker, analysis_dir: Path, seeds: list[int]) -> None:
    chk.section("§8.4  analysis dir: README + evolution curves + joint_*")
    chk.check((analysis_dir / "README.md").is_file(), "README.md exists")
    curves = sorted(p.name for p in analysis_dir.glob("evolution_curves*"))
    chk.check(bool(curves), f"evolution curves: {curves}")

    joints = sorted(p for p in analysis_dir.glob("joint_*") if p.is_dir())
    chk.check(bool(joints), f"joint dirs: {[p.name for p in joints]}")
    if not joints:
        return

    jd = joints[0]
    jitems = sorted(p.name for p in jd.iterdir())
    for needed in ("cross_experiment_cluster_composition.png",
                   "cross_experiment_strategy_space.png",
                   "source_manifest.json"):
        chk.check(needed in jitems, f"{jd.name}/{needed} present")
    n_seed_comp = len([n for n in jitems if re.match(r".+_seed\d+_cluster_composition\.png$", n)])
    chk.check(n_seed_comp == len(seeds), f"per-seed composition pngs = {n_seed_comp} (expected {len(seeds)})")


def check_cache(chk: Checker, cache: Path, experiment_id: str, population_size: int) -> None:
    chk.section("§8.5  cache lookup by experiment_id / run_id")
    chk.check(cache.is_file(), f"cache db exists: {cache}")
    if not cache.is_file():
        return

    con = sqlite3.connect(str(cache))
    try:
        row = con.execute(
            "SELECT run_id, cluster_count, cluster_names_json, created_at "
            "FROM clustering_runs WHERE experiment_id=? "
            "ORDER BY created_at DESC LIMIT 1",
            (experiment_id,),
        ).fetchone()
        chk.check(row is not None, f"{experiment_id} clustering run found in cache")
        if row is None:
            return

        rid, k, names_json, created = row
        print(f"      run_id={rid}  K={k}  created={created}")

        try:
            names = json.loads(names_json) if names_json else None
        except json.JSONDecodeError as exc:
            names = None
            print(f"      cluster_names_json parse error: {exc}")

        ok_names = (
            isinstance(names, dict)
            and len(names) == k
            and {int(x) for x in names if str(x).lstrip("-").isdigit()} == set(range(k))
            and all(isinstance(v, str) and v for v in names.values())
        )
        chk.check(ok_names, f"cluster_names_json is a {{str(cluster_id) -> name}} map of size K={k}")
        if isinstance(names, dict):
            for cid in sorted(names, key=lambda x: int(x)):
                print(f"      cluster {cid}: {names[cid]}")

        gen = con.execute(
            "SELECT MAX(generation) FROM cluster_assignments WHERE run_id=?", (rid,)
        ).fetchone()[0]
        rows = con.execute(
            "SELECT experiment_id, cluster_id, COUNT(*) FROM cluster_assignments "
            "WHERE run_id=? AND generation=? GROUP BY experiment_id, cluster_id "
            "ORDER BY experiment_id, cluster_id",
            (rid, gen),
        ).fetchall()
        chk.check(bool(rows), f"gen{gen} composition rows found ({len(rows)} rows)")

        by_exp: dict[str, dict[int, int]] = {}
        for exp, cid, cnt in rows:
            by_exp.setdefault(exp, {})[cid] = cnt
        for exp in sorted(by_exp):
            counts = [by_exp[exp].get(c, 0) for c in range(k)]
            chk.check(
                sum(counts) == population_size,
                f"  {exp}: {counts} sum={sum(counts)} (expected {population_size})",
            )
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="results root holding the seed dirs")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="strategy analysis sqlite cache")
    ap.add_argument(
        "--run-label",
        default="LLM_agent-type1_fermi_z_v3_g100_10000inter_N16_genreset_upd4_5seed",
        help="run label; seed dirs are '<label>_seed<N>'",
    )
    ap.add_argument(
        "--analysis-dir",
        type=Path,
        default=DEFAULT_ROOT / "analysis_agent-type1_g100_10000inter_N16_upd4_deliberate_5seed",
        help="summary analysis directory",
    )
    ap.add_argument("--experiment-id", default="joint_5seed", help="experiment_id used for joint clustering")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--population-size", type=int, default=16, help="expected ints per generation per seed")
    args = ap.parse_args()

    chk = Checker()
    check_seed_dirs(chk, args.root, args.run_label, args.seeds)
    check_files_nonempty(chk, args.root, args.run_label, args.seeds)
    check_lineage(chk, args.root, args.run_label, args.seeds)
    check_analysis_dir(chk, args.analysis_dir, args.seeds)
    check_cache(chk, args.cache, args.experiment_id, args.population_size)

    print("\n== SUMMARY ==")
    if chk.failures:
        print(f"FAILED {len(chk.failures)} check(s):")
        for f in chk.failures:
            print(f"  - {f}")
        return 1
    print("ALL ACCEPTANCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
