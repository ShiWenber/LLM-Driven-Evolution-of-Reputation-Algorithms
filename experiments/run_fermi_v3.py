"""CLI entry point for the v3 Fermi-style evolutionary run.

Usage examples:
    python -m experiments.run_fermi_v3 --seeds 0 1 2
    python -m experiments.run_fermi_v3 --seed 0 --gens 20 --target-interactions 200
    python -m experiments.run_fermi_v3 --provider deepseek --model deepseek-v4-flash --output-root results/quantitative_baseline
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import time
import traceback
from pathlib import Path

from experiments.config.load_env import get_api_key, get_base_url, get_model
from experiments.evolution_log import (
    evolution_json_path, load_evolution_json, run_dir, write_evolution_json,
)
from experiments.v2_quantitative.population import V2EvolutionaryPopulation

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "quantitative_baseline"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the v3 Fermi-style LLM evolutionary experiment with configurable CLI args.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Run a single seed. Overrides --seeds when set.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2],
                        help="Seed list to run. Defaults to [0, 1, 2].")
    parser.add_argument(
        "--seed-workers",
        type=int,
        default=None,
        help=(
            "Independent seed processes. Defaults to the number of seeds; "
            "use 1 to disable cross-seed parallelism."
        ),
    )
    parser.add_argument("--gens", "--num-generations", type=int, default=100,
                        help="Number of generations to run.")
    parser.add_argument(
        "--resume-json",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more evolutionary.json logs to continue. Experiment "
            "parameters are inherited from each log; seeds run in parallel."
        ),
    )
    parser.add_argument(
        "--additional-gens",
        type=int,
        default=None,
        help="Number of new generations to append to every --resume-json log.",
    )
    parser.add_argument("--target-interactions", type=int, default=1000,
                        help="Target PD interactions per generation.")
    parser.add_argument("--population-size", type=int, default=15,
                        help="Population size.")
    parser.add_argument("--updates-per-gen", type=int, default=None,
                        help="Distinct Fermi learners per generation; defaults to population size.")
    parser.add_argument(
        "--learning-method",
        choices=("fermi", "tournament"),
        default="fermi",
        help="Population learning/selection rule. Defaults to fermi.",
    )
    parser.add_argument("--llm-concurrency", type=int, default=None,
                        help="Concurrent LLM requests; defaults to population size.")
    parser.add_argument("--fermi-beta", type=float, default=5.0,
                        help="Fermi beta parameter.")
    parser.add_argument("--mutation-rate", type=float, default=0.1,
                        help="Mutation probability on adoption.")
    parser.add_argument("--mutation-temperature", type=float, default=0.8,
                        help="LLM mutation temperature.")
    parser.add_argument(
        "--imitation-learning",
        choices=("random", "deliberate"),
        default="random",
        help="How the LLM creates a child after imitation.",
    )
    parser.add_argument("--benefit", type=float, default=2.0,
                        help="PD cooperation benefit.")
    parser.add_argument("--cost", type=float, default=1.0,
                        help="PD cooperation cost.")
    parser.add_argument("--observability", type=str, default="full",
                        help="Observability mode passed to V2EvolutionaryPopulation.")
    parser.add_argument("--observability-p", type=float, default=1.0,
                        help="Probability of observability in the selected mode.")
    parser.add_argument("--elite-count", type=int, default=2,
                        help="Elite count for selection logic.")
    parser.add_argument("--num-eliminate", type=int, default=5,
                        help="Number of individuals eliminated per generation.")
    parser.add_argument("--tournament-size", type=int, default=3,
                        help="Tournament size used when selection is tournament-based.")
    parser.add_argument("--provider", type=str, default="deepseek",
                        help="LLM provider name. Used for API key and base URL lookup.")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model name. Defaults to the platform default for the selected provider.")
    parser.add_argument("--llm-thinking", action="store_true",
                        help="Enable thinking mode for the LLM API call.")
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Run label. By default it includes the imitation-learning mode.",
    )
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT),
                        help="Root directory where per-seed result folders are created.")
    parser.add_argument("--agent-type", type=str, default="agent-type2",
                        choices=["agent-type1", "agent-type2", "v2", "v3"],
                        help="Agent family to evolve: 'agent-type1' (legacy 'v2', type-1 "
                             "functions) or 'agent-type2' (legacy 'v3', full LLMAgent class).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate arguments and print the seed plan without running.")
    return parser


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seed is not None:
        return [int(args.seed)]
    if not args.seeds:
        return [0]
    return [int(s) for s in args.seeds]


def run_one_seed(args: argparse.Namespace, seed: int, label: str, out_root: Path) -> dict:
    seed_dir = run_dir(out_root, label, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    out_path = evolution_json_path(out_root, label, seed)
    if out_path.exists():
        out_path.unlink()
        print(f"  [seed {seed}] removed existing {out_path}", flush=True)

    api_key = get_api_key(args.provider)
    base_url = get_base_url(args.provider)
    model = get_model(args.provider, args.model)

    print(f"=== seed {seed} start @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    t0 = time.time()
    summary = {"seed": seed, "completed": False, "error": None}

    try:
        pop = V2EvolutionaryPopulation(
            population_size=args.population_size,
            target_interactions_per_gen=args.target_interactions,
            benefit=args.benefit,
            cost=args.cost,
            observability=args.observability,
            observability_p=args.observability_p,
            elite_count=args.elite_count,
            num_eliminate=args.num_eliminate,
            tournament_size=args.tournament_size,
            llm_model=model,
            api_key=api_key,
            api_base_url=base_url,
            mutation_temperature=args.mutation_temperature,
            seed=seed,
            results_dir=str(out_root),
            use_baseline=None,
            agent_type=args.agent_type,
            llm_thinking=args.llm_thinking,
            learning_method=args.learning_method,
            fermi_beta=args.fermi_beta,
            mutation_rate_on_adoption=args.mutation_rate,
            imitation_learning_mode=args.imitation_learning,
            updates_per_gen=args.updates_per_gen,
            llm_concurrency=args.llm_concurrency,
        )
        result = pop.run_evolution(num_generations=args.gens)
        elapsed = time.time() - t0

        write_evolution_json(out_path, result)

        last = result["trajectory"][-1]
        summary.update({
            "completed": True,
            "elapsed_sec": elapsed,
            "elapsed_min": elapsed / 60,
            "final_coop": last["cooperation_rate_mean"],
            "final_fitness": last["fitness_mean"],
            "fallback_init": result["config"]["fallback_init_count"],
            "fallback_mutation": result["config"]["fallback_mutation_count"],
            "gen0_coop": result["trajectory"][0]["cooperation_rate_mean"],
        })
        print(f"=== seed {seed} done in {elapsed/60:.1f} min ===", flush=True)
        print(f"  gen 0 coop: {summary['gen0_coop']:.3f}", flush=True)
        print(f"  final coop: {summary['final_coop']:.3f}", flush=True)
        print(f"  final fitness: {summary['final_fitness']:.1f}", flush=True)
        print(
            f"  FALLBACK: init={summary['fallback_init']}/{args.population_size}, "
            f"mutation={summary['fallback_mutation']}/{max(1, (args.gens - 1) * args.updates_per_gen)}",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - CLI wrapper around long-running experiment
        elapsed = time.time() - t0
        summary.update({
            "completed": False,
            "elapsed_sec": elapsed,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        print(f"=== seed {seed} FAILED after {elapsed/60:.1f} min ===", flush=True)
        print(f"  {type(exc).__name__}: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)

    return summary


def _resume_source_path(value: str | Path) -> Path:
    path = Path(value).resolve()
    return path / "evolutionary.json" if path.is_dir() else path


def run_one_resume(
    args: argparse.Namespace,
    source: str | Path,
    label: str,
    out_root: Path,
) -> dict:
    """Continue one saved trajectory without modifying its source file."""
    source_path = _resume_source_path(source)
    previous = load_evolution_json(source_path)
    cfg = previous["config"]
    seed = int(cfg["seed"])
    out_path = evolution_json_path(out_root, label, seed)
    if out_path.resolve() == source_path.resolve():
        raise ValueError("resume output must not overwrite its source log")
    if out_path.exists():
        raise FileExistsError(
            f"resume output already exists: {out_path}; choose a new --label"
        )

    api_key = get_api_key(args.provider)
    base_url = get_base_url(args.provider)
    model = get_model(args.provider, args.model or cfg.get("llm_model"))
    llm_concurrency = (
        args.llm_concurrency
        if args.llm_concurrency is not None
        else cfg.get("llm_concurrency")
    )
    agent_type = cfg.get("agent_type", "agent-type1")
    if agent_type == "v2":
        agent_type = "agent-type1"
    elif agent_type == "v3":
        agent_type = "agent-type2"

    print(
        f"=== seed {seed} resume {len(previous['trajectory'])} "
        f"+ {args.additional_gens} generations ===",
        flush=True,
    )
    t0 = time.time()
    summary = {
        "seed": seed,
        "source": str(source_path),
        "completed": False,
        "error": None,
    }
    try:
        pop = V2EvolutionaryPopulation(
            population_size=int(cfg["population_size"]),
            num_rounds_per_gen=int(cfg.get("num_rounds_per_gen", 30)),
            target_interactions_per_gen=cfg.get("target_interactions_per_gen"),
            fitness_window_interactions=cfg.get("fitness_window_interactions", 200),
            benefit=float(cfg.get("benefit", 2.0)),
            cost=float(cfg.get("cost", 1.0)),
            num_generations=len(previous["trajectory"]) + args.additional_gens,
            observability=cfg.get("observability", "full"),
            observability_p=float(cfg.get("observability_p", 1.0)),
            elite_count=int(cfg.get("elite_count", 2)),
            num_eliminate=int(cfg.get("num_eliminate", 5)),
            tournament_size=int(cfg.get("tournament_size", 3)),
            llm_model=model,
            api_key=api_key,
            api_base_url=base_url,
            mutation_temperature=float(cfg.get("mutation_temperature", 0.8)),
            seed=seed,
            results_dir=str(out_root),
            use_baseline=cfg.get("use_baseline"),
            agent_type=agent_type,
            llm_thinking=bool(cfg.get("llm_thinking", False) or args.llm_thinking),
            use_fermi=bool(cfg.get("use_fermi", False)),
            learning_method=cfg.get("learning_method"),
            fermi_beta=float(cfg.get("fermi_beta", 5.0)),
            mutation_rate_on_adoption=float(
                cfg.get("mutation_rate_on_adoption", 0.1)
            ),
            imitation_learning_mode=cfg.get("imitation_learning_mode", "random"),
            updates_per_gen=int(
                cfg.get("updates_per_gen", cfg["population_size"])
            ),
            llm_concurrency=llm_concurrency,
        )
        result = pop.resume_evolution(
            previous,
            args.additional_gens,
            source_path=str(source_path),
        )
        elapsed = time.time() - t0
        write_evolution_json(out_path, result)
        last = result["trajectory"][-1]
        summary.update({
            "completed": True,
            "output": str(out_path),
            "elapsed_sec": elapsed,
            "elapsed_min": elapsed / 60,
            "total_generations": len(result["trajectory"]),
            "final_coop": last["cooperation_rate_mean"],
            "final_fitness": last["fitness_mean"],
            "rng_mode": result["config"]["resume"]["rng_mode"],
        })
        print(
            f"=== seed {seed} resume done: total={len(result['trajectory'])}, "
            f"{elapsed/60:.1f} min ===",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - long-running CLI wrapper
        elapsed = time.time() - t0
        summary.update({
            "elapsed_sec": elapsed,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        print(f"=== seed {seed} resume FAILED: {type(exc).__name__}: {exc} ===")
        print(traceback.format_exc(), flush=True)
    return summary


def run_resume_batch(
    args: argparse.Namespace,
    sources: list[str | Path],
    label: str,
    out_root: Path,
    *,
    on_result=None,
    executor_cls=ProcessPoolExecutor,
) -> list[dict]:
    """Continue independent trajectory logs in separate processes."""
    if not sources:
        return []
    worker_limit = args.seed_workers or len(sources)
    max_workers = min(worker_limit, len(sources))
    if len(sources) == 1:
        result = run_one_resume(args, sources[0], label, out_root)
        if on_result is not None:
            on_result([result])
        return [result]

    completed: dict[int, dict] = {}
    with executor_cls(max_workers=max_workers) as pool:
        future_to_index = {
            pool.submit(run_one_resume, args, source, label, out_root): index
            for index, source in enumerate(sources)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                completed[index] = future.result()
            except Exception as exc:
                completed[index] = {
                    "source": str(sources[index]),
                    "completed": False,
                    "error": f"worker {type(exc).__name__}: {exc}",
                }
            if on_result is not None:
                on_result([completed[i] for i in range(len(sources)) if i in completed])
    return [completed[i] for i in range(len(sources))]


def run_seed_batch(
    args: argparse.Namespace,
    seeds: list[int],
    label: str,
    out_root: Path,
    *,
    on_result=None,
    executor_cls=ProcessPoolExecutor,
) -> list[dict]:
    """Run independent seeds in separate processes and return seed-list order.

    The callback, when provided, runs in the parent process after each seed
    finishes.  Keeping aggregation in the parent prevents concurrent writes to
    the combined summary file.
    """
    if not seeds:
        return []

    worker_limit = args.seed_workers or len(seeds)
    max_workers = min(worker_limit, len(seeds))
    if len(seeds) == 1:
        result = run_one_seed(args, seeds[0], label, out_root)
        if on_result is not None:
            on_result([result])
        return [result]

    completed: dict[int, dict] = {}
    with executor_cls(max_workers=max_workers) as pool:
        future_to_seed = {
            pool.submit(run_one_seed, args, seed, label, out_root): seed
            for seed in seeds
        }
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                completed[seed] = future.result()
            except Exception as exc:  # a worker may exit before run_one_seed catches it
                completed[seed] = {
                    "seed": seed,
                    "completed": False,
                    "error": f"worker {type(exc).__name__}: {exc}",
                }
            ordered_partial = [completed[s] for s in seeds if s in completed]
            if on_result is not None:
                on_result(ordered_partial)

    return [completed[seed] for seed in seeds]


def run_resume_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Validate and dispatch --resume-json mode."""
    if args.additional_gens is None or args.additional_gens < 1:
        parser.error("--resume-json requires --additional-gens >= 1")
    sources = [_resume_source_path(value) for value in args.resume_json]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        parser.error(f"resume log not found: {', '.join(missing)}")

    previews = [load_evolution_json(path) for path in sources]
    seeds = [int(data["config"]["seed"]) for data in previews]
    if len(set(seeds)) != len(seeds):
        parser.error(
            "--resume-json inputs must have distinct seeds because outputs "
            "use one directory per seed"
        )
    for path, data in zip(sources, previews):
        cfg = data["config"]
        if cfg.get("use_baseline") or not cfg.get("use_fermi"):
            parser.error(f"resume currently requires a Fermi LLM log: {path}")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    label = args.label or f"continued_plus{args.additional_gens}gen"
    effective_workers = min(args.seed_workers or len(sources), len(sources))
    print(
        f"=== resume {len(sources)} trajectories "
        f"(+{args.additional_gens} generations, processes={effective_workers}) ==="
    )
    for path, data in zip(sources, previews):
        cfg = data["config"]
        rng_mode = "checkpoint" if cfg.get("rng_state") is not None else "derived_branch"
        print(
            f"  seed={cfg['seed']}: {len(data['trajectory'])} -> "
            f"{len(data['trajectory']) + args.additional_gens} gens, "
            f"rng={rng_mode}, source={path}"
        )
    print(f"  output label: {label} (source logs are never overwritten)")

    if args.dry_run:
        return 0
    api_key = get_api_key(args.provider)
    if not api_key:
        print(
            f"[run_fermi_v3] no API key found for provider '{args.provider}'.",
            flush=True,
        )
        return 2

    overall_t0 = time.time()

    def write_summary(partial: list[dict]) -> None:
        summary_path = out_root / f"{label}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as stream:
            json.dump({
                "label": label,
                "mode": "resume",
                "additional_generations": args.additional_gens,
                "execution": "multiprocess_by_trajectory",
                "seed_workers": effective_workers,
                "sources": [str(path) for path in sources],
                "runs": partial,
                "overall_elapsed_sec": time.time() - overall_t0,
            }, stream, indent=2, ensure_ascii=False)

    summary = run_resume_batch(
        args,
        sources,
        label,
        out_root,
        on_result=write_summary,
    )
    n_done = sum(1 for row in summary if row.get("completed"))
    print(
        f"=== RESUME DONE: {n_done}/{len(summary)} completed in "
        f"{(time.time() - overall_t0)/60:.1f} min ==="
    )
    for row in summary:
        if row.get("completed"):
            print(
                f"  seed {row['seed']}: total={row['total_generations']}, "
                f"final_coop={row['final_coop']:.3f}, rng={row['rng_mode']}"
            )
        else:
            print(f"  {row.get('source', '?')}: FAILED ({row.get('error', '?')})")
    return 0 if n_done == len(summary) else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Normalize legacy agent_type aliases to canonical values.
    if args.agent_type == "v2":
        args.agent_type = "agent-type1"
    elif args.agent_type == "v3":
        args.agent_type = "agent-type2"

    provider = args.provider.lower()
    if args.llm_concurrency is not None and args.llm_concurrency < 1:
        parser.error("--llm-concurrency must be >= 1")
    if args.seed_workers is not None and args.seed_workers < 1:
        parser.error("--seed-workers must be >= 1")
    if args.resume_json is not None:
        return run_resume_cli(args, parser)
    if args.additional_gens is not None:
        parser.error("--additional-gens requires --resume-json")
    if args.updates_per_gen is None:
        args.updates_per_gen = args.population_size
    if args.updates_per_gen < 0 or args.updates_per_gen > args.population_size:
        parser.error(
            "--updates-per-gen must be between 0 and --population-size "
            "for without-replacement learner sampling"
        )
    seeds = resolve_seeds(args)
    if len(set(seeds)) != len(seeds):
        parser.error("--seeds must not contain duplicates")
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    label = args.label or (
        f"LLM_v3_{args.learning_method}_v3_g100_1000inter_learn-{args.imitation_learning}"
    )

    effective_seed_workers = min(args.seed_workers or len(seeds), len(seeds))
    print(
        f"=== {label} seeds={seeds} "
        f"(processes={effective_seed_workers}) ===",
        flush=True,
    )
    print(f"  provider: {provider}, model: {get_model(provider, args.model)}", flush=True)
    print(f"  num_gens: {args.gens}, target_interactions: {args.target_interactions}", flush=True)
    print(f"  learning_method: {args.learning_method}", flush=True)
    if args.learning_method == "fermi":
        print(f"  Z-like: mu={args.mutation_rate}, beta={args.fermi_beta}, updates_per_gen={args.updates_per_gen}", flush=True)
    print(f"  llm_concurrency: {args.llm_concurrency or args.population_size}", flush=True)
    print(f"  seed_workers: {effective_seed_workers}", flush=True)
    print(
        "  max aggregate LLM concurrency: "
        f"{effective_seed_workers * (args.llm_concurrency or args.population_size)}",
        flush=True,
    )
    print(f"  imitation_learning: {args.imitation_learning}", flush=True)
    print(f"  prompts: v3 / minimal Fermi, agent_type={args.agent_type}", flush=True)
    print(f"  output_root: {out_root}", flush=True)
    print(flush=True)

    if args.dry_run:
        api_key = get_api_key(provider)
        if api_key:
            print(f"  api_key: {api_key[:8]}...{api_key[-4:]}", flush=True)
        else:
            print("  api_key: (not configured; dry-run only)", flush=True)
        print("[dry-run] would execute:")
        for seed in seeds:
            print(f"  seed={seed} -> {out_root / f'{label}_seed{seed}'}")
        return 0

    api_key = get_api_key(provider)
    if not api_key:
        print(f"[run_fermi_v3] no API key found for provider '{provider}'. Set {provider.upper()}_API_KEY or use .env.", flush=True)
        return 2

    print(f"  api_key: {api_key[:8]}...{api_key[-4:]}", flush=True)

    overall_t0 = time.time()
    def write_summary(partial_summary: list[dict]) -> None:
        summary_path = out_root / f"{label}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "label": label,
                "num_gens": args.gens,
                "target_interactions_per_gen": args.target_interactions,
                "scheme": args.learning_method,
                "provider": provider,
                "model": get_model(provider, args.model),
                "execution": "multiprocess_by_seed",
                "seed_workers": effective_seed_workers,
                "llm_concurrency_per_seed": args.llm_concurrency or args.population_size,
                "seeds": partial_summary,
                "overall_elapsed_sec": time.time() - overall_t0,
            }, f, indent=2)

    summary = run_seed_batch(
        args,
        seeds,
        label,
        out_root,
        on_result=write_summary,
    )

    total_elapsed = time.time() - overall_t0
    n_done = sum(1 for s in summary if s["completed"])
    print(f"\n=== ALL SEEDS DONE in {total_elapsed/3600:.2f} h ({total_elapsed/60:.0f} min) ===", flush=True)
    print(f"  {n_done}/{len(seeds)} seeds completed successfully", flush=True)
    for s in summary:
        if s["completed"]:
            print(
                f"  seed {s['seed']}: final_coop={s['final_coop']:.3f}, "
                f"final_fitness={s['final_fitness']:.1f}, "
                f"min={s['elapsed_min']:.1f}, "
                f"FALLBACK init/mut={s['fallback_init']}/{s['fallback_mutation']}"
            )
        else:
            print(f"  seed {s['seed']}: FAILED ({s.get('error', '?')})")

    return 0 if n_done == len(seeds) else 1


if __name__ == "__main__":
    sys.exit(main())
