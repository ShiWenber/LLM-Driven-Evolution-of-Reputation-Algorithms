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
    evolution_json_path, run_dir, write_evolution_json,
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
    parser.add_argument("--target-interactions", type=int, default=1000,
                        help="Target PD interactions per generation.")
    parser.add_argument("--population-size", type=int, default=15,
                        help="Population size.")
    parser.add_argument("--updates-per-gen", type=int, default=None,
                        help="Distinct Fermi learners per generation; defaults to population size.")
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
            use_fermi=True,
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
        f"LLM_v3_fermi_z_v3_g100_1000inter_learn-{args.imitation_learning}"
    )

    effective_seed_workers = min(args.seed_workers or len(seeds), len(seeds))
    print(
        f"=== {label} seeds={seeds} "
        f"(processes={effective_seed_workers}) ===",
        flush=True,
    )
    print(f"  provider: {provider}, model: {get_model(provider, args.model)}", flush=True)
    print(f"  num_gens: {args.gens}, target_interactions: {args.target_interactions}", flush=True)
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
                "scheme": "fermi_z_like",
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
