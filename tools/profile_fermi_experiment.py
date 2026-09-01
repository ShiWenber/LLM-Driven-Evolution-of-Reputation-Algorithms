"""Profile the Fermi experiment and emit machine-readable timing data.

This wrapper deliberately avoids changing the experiment implementation.  It
monkey-patches a small set of phase boundaries, runs the normal CLI entry point,
and writes JSON plus cProfile text/pstats files for later visualization.
"""
from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import threading
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from experiments import run_fermi_v3
from experiments.v2_quantitative.game import V2DonorGame
from experiments.v2_quantitative.population import V2EvolutionaryPopulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gens", type=int, default=5)
    parser.add_argument("--target-interactions", type=int, default=200)
    parser.add_argument("--population-size", type=int, default=15)
    parser.add_argument("--updates-per-gen", type=int, default=15)
    parser.add_argument("--agent-type", default="agent-type1")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_output = output_dir / "experiment-output"

    calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lock = threading.Lock()
    originals: list[tuple[type, str, Callable[..., Any]]] = []

    def instrument(cls: type, method_name: str, phase: str) -> None:
        original = getattr(cls, method_name)
        originals.append((cls, method_name, original))

        def wrapped(*method_args: Any, **method_kwargs: Any) -> Any:
            wall0 = time.perf_counter()
            cpu0 = time.process_time()
            ok = False
            try:
                result = original(*method_args, **method_kwargs)
                ok = True
                return result
            finally:
                event = {
                    "phase": phase,
                    "wall_sec": time.perf_counter() - wall0,
                    "cpu_sec": time.process_time() - cpu0,
                    "ok": ok,
                    "finished_at_sec": time.perf_counter() - overall_wall0,
                }
                with lock:
                    calls[phase].append(event)

        wrapped.__name__ = getattr(original, "__name__", method_name)
        wrapped.__doc__ = getattr(original, "__doc__", None)
        setattr(cls, method_name, wrapped)

    overall_wall0 = time.perf_counter()
    overall_cpu0 = time.process_time()
    instrument(V2EvolutionaryPopulation, "_call_llm", "llm_api")
    instrument(V2EvolutionaryPopulation, "_init_population_llm", "llm_initialization")
    instrument(V2EvolutionaryPopulation, "_run_one_generation", "game_generation")
    instrument(V2EvolutionaryPopulation, "_select_and_reproduce_fermi", "fermi_reproduction")
    instrument(V2EvolutionaryPopulation, "_validate_code", "strategy_validation")
    instrument(V2EvolutionaryPopulation, "_make_agent", "strategy_compile_instantiate")
    instrument(V2DonorGame, "play_round", "game_round")
    instrument(V2DonorGame, "distribute_observations_and_self_judgments", "observation_distribution")

    cli_args = [
        "--seed", str(args.seed),
        "--gens", str(args.gens),
        "--target-interactions", str(args.target_interactions),
        "--population-size", str(args.population_size),
        "--updates-per-gen", str(args.updates_per_gen),
        "--agent-type", args.agent_type,
        "--provider", args.provider,
        "--label", f"profile_{args.agent_type}_g{args.gens}_i{args.target_interactions}",
        "--output-root", str(run_output),
    ]
    if args.model:
        cli_args += ["--model", args.model]

    profiler = cProfile.Profile()
    tracemalloc.start()
    exit_code = 1
    error: str | None = None
    try:
        profiler.enable()
        exit_code = run_fermi_v3.main(cli_args)
    except BaseException as exc:  # preserve diagnostics even on CLI failure
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        profiler.disable()
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        wall_sec = time.perf_counter() - overall_wall0
        cpu_sec = time.process_time() - overall_cpu0
        for cls, method_name, original in reversed(originals):
            setattr(cls, method_name, original)

        pstats_path = output_dir / "cpu_profile.pstats"
        profiler.dump_stats(str(pstats_path))
        stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
        with (output_dir / "cpu_profile.txt").open("w", encoding="utf-8") as handle:
            stats.stream = handle
            stats.print_stats(80)

        functions = []
        for (filename, line, funcname), (cc, nc, tt, ct, _callers) in stats.stats.items():
            functions.append({
                "function": f"{filename}:{line}({funcname})",
                "primitive_calls": cc,
                "total_calls": nc,
                "self_sec": tt,
                "cumulative_sec": ct,
            })
        functions.sort(key=lambda row: row["cumulative_sec"], reverse=True)

        payload = {
            "config": vars(args) | {"output_dir": str(output_dir)},
            "command_args": cli_args,
            "exit_code": exit_code,
            "error": error,
            "overall": {
                "wall_sec": wall_sec,
                "cpu_sec": cpu_sec,
                "cpu_to_wall_ratio": cpu_sec / wall_sec if wall_sec else 0.0,
                "python_current_mib": current_bytes / 1024 / 1024,
                "python_peak_mib": peak_bytes / 1024 / 1024,
            },
            "events": dict(calls),
            "top_functions": functions[:120],
        }
        with (output_dir / "profile.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
