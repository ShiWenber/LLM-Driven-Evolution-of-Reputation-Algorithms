"""Legacy production wrapper for the canonical multiprocess Fermi CLI.

This keeps the historical positional calling conventions while delegating all
execution to ``experiments.run_fermi_v3``. Independent seeds therefore use
separate processes from both supported entry points.
"""
from __future__ import annotations

import sys

from experiments.run_fermi_v3 import main as run_fermi_main


def _arg_value(argv: list[str], flag: str, default: str) -> str:
    if flag in argv:
        index = argv.index(flag)
        if index + 1 < len(argv):
            return argv[index + 1]
    return default


def _agent_type(argv: list[str]) -> str:
    aliases = {
        "agent-type1": "agent-type1",
        "agent-type2": "agent-type2",
        "v2": "agent-type1",
        "v3": "agent-type2",
    }
    for token in argv:
        if token.lower() in aliases:
            return aliases[token.lower()]
    return "agent-type2"


def _seeds(argv: list[str]) -> list[int]:
    if "--seeds" in argv:
        values = []
        for token in argv[argv.index("--seeds") + 1:]:
            if token.startswith("--"):
                break
            values.append(int(token))
        return values or [0]
    if argv:
        try:
            return [int(argv[0])]
        except ValueError:
            pass
    return [0, 1, 2]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    agent_type = _agent_type(argv)
    seeds = _seeds(argv)
    observability = _arg_value(argv, "--observability", "full")
    observability_p = _arg_value(argv, "--observability-p", "1.0")
    imitation = _arg_value(argv, "--imitation-learning", "random")
    llm_concurrency = _arg_value(argv, "--llm-concurrency", "16")
    seed_workers = _arg_value(argv, "--seed-workers", str(len(seeds)))

    label = f"LLM_{agent_type}_fermi_z_v3_g100_1000inter_N16_genreset"
    if observability != "full":
        label += f"_{observability}{observability_p}".replace(".", "p")
    label += f"_learn-{imitation}"

    canonical_args = [
        "--agent-type", agent_type,
        "--seeds", *(str(seed) for seed in seeds),
        "--seed-workers", seed_workers,
        "--gens", "100",
        "--target-interactions", "1000",
        "--population-size", "16",
        "--updates-per-gen", "15",
        "--llm-concurrency", llm_concurrency,
        "--provider", "deepseek",
        "--model", "deepseek-v4-flash",
        "--observability", observability,
        "--observability-p", observability_p,
        "--imitation-learning", imitation,
        "--label", label,
    ]
    if "--dry-run" in argv:
        canonical_args.append("--dry-run")
    return run_fermi_main(canonical_args)


if __name__ == "__main__":
    raise SystemExit(main())
