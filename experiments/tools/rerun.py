"""Rerun orchestration for LLM-reputation experiments.

Provides:
- `rerun_donor_experiments`: re-run all four donor-game experiments
  with consistent seed management, after the mutation-prompt fix
- `rerun_ipd_baseline`: re-run the IPD baseline (Willis comparison)
- `audit_existing_results`: print a coverage report of what's already
  in experiments/results/ so we know which seeds to run

Seed plan (after ISSUES.md mutation-prompt fix):
    Experiment 1: PRIVATE, FULL × 5 seeds (=10 runs, was 3)
    Experiment 2: 9 observability levels × 5 seeds (=45 runs, was 3)
    Experiment 3: PRIVATE, p=0.3, FULL × 3 seeds (=9 runs)
    Experiment 4: PRIVATE, p=0.3, FULL × 3 seeds (=9 seeds)
    Experiment 5 (IPD): 5 populations × 3 seeds (=15 runs)
    Total: ~88 trials, est. 3-12 hours wall-clock depending on API latency.

Usage:
    python -m experiments.tools.rerun --experiments 1 2 --seeds 0 1 2
    python -m experiments.tools.rerun --audit
    python -m experiments.tools.rerun --experiments 5 --seeds 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

from ..config.load_env import get_api_key

REPO_ROOT = Path(__file__).resolve().parents[2]


def audit_existing_results(results_dir: str = "results") -> Dict[str, Any]:
    """Report which experimental trials already have results on disk."""
    out_path = Path(results_dir)
    audit = {"experiments": {}}
    if not out_path.exists():
        return audit

    for exp_dir in out_path.iterdir():
        if not exp_dir.is_dir():
            continue
        # Each exp dir contains seed-tagged JSON files
        trial_files = sorted(exp_dir.glob("*.json"))
        audit["experiments"][exp_dir.name] = {
            "num_trials": len(trial_files),
            "files": [f.name for f in trial_files],
        }
    return audit


def rerun_donor_experiments(
    experiments: List[int],
    seeds: List[int],
    output_dir: str = "results",
    dry_run: bool = False,
):
    """Re-run donor-game experiments with given seed plan.

    Maps experiment number to its corresponding main.py invocation.
    """
    plan = []
    if 1 in experiments:
        # Experiment 1: Observability contrast (PRIVATE, FULL)
        for seed in seeds:
            plan.append((
                ["python", "-m", "experiments.main", "--run", "evolutionary",
                 "--observability", "private,full",
                 "--seeds", "1", "--seed-offset", str(seed),
                 "--output", f"{output_dir}/exp1_obs_contrast"],
                f"exp1 seed={seed}",
            ))
    if 2 in experiments:
        # Experiment 2: Critical threshold scan
        for seed in seeds:
            plan.append((
                ["python", "-m", "experiments.main", "--run", "threshold",
                 "--p-values", "0,0.05,0.10,0.15,0.20,0.30,0.50,0.70,1.0",
                 "--seeds", "1", "--seed-offset", str(seed),
                 "--output", f"{output_dir}/exp2_threshold_scan"],
                f"exp2 seed={seed}",
            ))
    if 3 in experiments:
        # Experiment 3: Static control
        for seed in seeds:
            plan.append((
                ["python", "-m", "experiments.main", "--run", "static",
                 "--observability", "private,partial_0.3,full",
                 "--seeds", "1", "--seed-offset", str(seed),
                 "--output", f"{output_dir}/exp3_static_control"],
                f"exp3 seed={seed}",
            ))
    if 4 in experiments:
        # Experiment 4: Random mutation control
        for seed in seeds:
            plan.append((
                ["python", "-m", "experiments.main", "--run", "random-mutation",
                 "--observability", "private,partial_0.3,full",
                 "--seeds", "1", "--seed-offset", str(seed),
                 "--output", f"{output_dir}/exp4_random_mutation"],
                f"exp4 seed={seed}",
            ))

    return _run_plan(plan, dry_run=dry_run)


def rerun_ipd_baseline(seeds: List[int], output_dir: str = "results", dry_run: bool = False):
    """Run the IPD baseline (Willis comparison) with given seeds."""
    plan = []
    for seed in seeds:
        plan.append((
            ["python", "-m", "experiments.evolution.ipd_evolution",
             "--seed", str(seed),
             "--output", f"{output_dir}/ipd_baseline"],
            f"ipd seed={seed}",
        ))
    return _run_plan(plan, dry_run=dry_run)


def _run_plan(plan: List[tuple], dry_run: bool = False) -> List[Dict[str, Any]]:
    """Execute a plan of (cmd, label) tuples, reporting per-trial timing."""
    if not get_api_key("deepseek"):
        print("[rerun] No DEEPSEEK_API_KEY configured. Aborting.")
        print("       Add it to .env then re-run.")
        return []

    print(f"[rerun] Executing {len(plan)} trial(s)...")
    results = []
    for i, (cmd, label) in enumerate(plan, 1):
        print(f"\n[rerun] ({i}/{len(plan)}) {label}")
        print(f"        $ {' '.join(cmd)}")
        if dry_run:
            results.append({"label": label, "status": "dry-run", "cmd": cmd})
            continue
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour per trial max
            )
            elapsed = time.time() - t0
            ok = proc.returncode == 0
            results.append({
                "label": label,
                "status": "ok" if ok else "fail",
                "elapsed_sec": round(elapsed, 1),
                "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
            })
            print(f"        status={'OK' if ok else 'FAIL'}, elapsed={elapsed:.1f}s")
        except subprocess.TimeoutExpired:
            results.append({
                "label": label, "status": "timeout",
                "elapsed_sec": 3600,
            })
            print(f"        status=TIMEOUT (>1h)")
    return results


def main():
    p = argparse.ArgumentParser(description="Rerun experiments with seed management")
    p.add_argument("--experiments", type=int, nargs="*", default=None,
                   help="Donor-game experiment numbers to run (1-5; 5 = IPD baseline)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Random seeds to run")
    p.add_argument("--output", type=str, default="results",
                   help="Output directory (default: results)")
    p.add_argument("--audit", action="store_true",
                   help="Audit existing results and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--ipd-only", action="store_true",
                   help="Only run IPD baseline (Experiment 5)")
    args = p.parse_args()

    if args.audit:
        audit = audit_existing_results(args.output)
        print(json.dumps(audit, indent=2))
        return 0

    if args.ipd_only or (args.experiments and 5 in args.experiments):
        rerun_ipd_baseline(args.seeds, output_dir=args.output, dry_run=args.dry_run)

    if args.experiments and not args.ipd_only:
        donor_exps = [e for e in args.experiments if e in (1, 2, 3, 4)]
        if donor_exps:
            rerun_donor_experiments(
                donor_exps, args.seeds,
                output_dir=args.output, dry_run=args.dry_run
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
