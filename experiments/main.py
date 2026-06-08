"""Main entry point for evolutionary indirect reciprocity experiments.

Architecture: LLM generates Python strategy code → sandbox executes
→ evolutionary selection → LLM mutates surviving code → repeat.

Supports:
- Evolutionary emergence (RUN 1)
- Critical threshold scan (RUN 2)
- No-evolution static control (RUN 3)
- Random mutation control (RUN 4)
"""

import argparse
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.evolution.population import EvolutionaryPopulation
from experiments.evolution.mutation import RandomMutationOperator, MutationOperator
from experiments.evolution.selection import compute_fitness_stats
from experiments.agents.code_agent import CodeAgent
from experiments.game.donor_game import DonorGame
from experiments.config.load_env import (
    get_api_key, get_base_url, get_model
)


# Model configurations.
# API key + base URL are loaded at runtime from .env via load_env.
MODELS = {
    "gpt-4o": {
        "name": "gpt-4o",
        "provider": "openai",
        "key":  lambda: get_api_key("openai"),
        "url":  lambda: get_base_url("openai"),
    },
    "claude-3-5-sonnet": {
        "name": "claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "key":  lambda: get_api_key("anthropic"),
        "url":  lambda: get_base_url("anthropic"),
    },
    "deepseek-v4-flash": {
        "name": "deepseek-v4-flash",
        "provider": "openai",  # OpenAI-compatible
        "key":  lambda: get_api_key("deepseek"),
        "url":  lambda: get_base_url("deepseek"),
    },
    "deepseek-coder": {
        "name": "deepseek-coder",
        "provider": "openai",  # OpenAI-compatible
        "key":  lambda: get_api_key("deepseek"),
        "url":  lambda: get_base_url("deepseek"),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolutionary Indirect Reciprocity Experiments"
    )

    # Run type
    parser.add_argument(
        "--run", type=str, required=True,
        choices=["evolutionary", "threshold", "static", "random-mutation"],
        help="Experiment type"
    )

    # Observability
    parser.add_argument(
        "--observability", type=str, default="full",
        help="Observability: private, partial_0.3, partial_0.7, full, "
             "or comma-separated list"
    )
    parser.add_argument(
        "--p-values", type=str, default=None,
        help="Comma-separated p values for threshold scan"
    )

    # Population
    parser.add_argument(
        "--population", type=int, default=20,
        help="Population size"
    )
    parser.add_argument(
        "--generations", type=int, default=10,
        help="Number of generations"
    )
    parser.add_argument(
        "--rounds", type=int, default=30,
        help="Rounds per generation"
    )

    # Selection
    parser.add_argument(
        "--eliminate", type=int, default=5,
        help="Number of agents eliminated per generation"
    )
    parser.add_argument(
        "--elitism", type=int, default=2,
        help="Number of elite agents (survive directly)"
    )
    parser.add_argument(
        "--tournament", type=int, default=3,
        help="Tournament size for selection"
    )

    # Models
    parser.add_argument(
        "--models", type=str, default="gpt-4o",
        help="Comma-separated model keys: gpt-4o,claude-3-5-sonnet"
    )

    # LLM parameters
    parser.add_argument(
        "--mutation-temperature", type=float, default=0.8,
        help="LLM temperature for mutation"
    )

    # Execution
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument(
        "--benefit", type=int, default=2,
        help="Benefit to recipient"
    )
    parser.add_argument(
        "--cost", type=int, default=1,
        help="Cost to donor"
    )

    return parser.parse_args()


def get_models(model_arg: str) -> List[Dict[str, str]]:
    """Parse model argument. Resolves lazy key/url lambdas now."""
    models = []
    for key in model_arg.split(","):
        key = key.strip()
        if key in MODELS:
            entry = MODELS[key]
            # Resolve the lazy key/url fields into plain strings
            resolved = {
                "name":       entry["name"],
                "provider":   entry["provider"],
                "api_key":    entry["key"]() if callable(entry.get("key")) else entry.get("api_key", ""),
                "api_base_url": entry["url"]() if callable(entry.get("url")) else entry.get("api_base_url", ""),
            }
            if not resolved["api_key"]:
                print(f"Warning: no API key configured for '{key}'. "
                      f"Set DEEPSEEK_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY in .env")
            models.append(resolved)
        else:
            print(f"Warning: unknown model '{key}'")
    return models if models else [_resolve(MODELS["gpt-4o"])]


def _resolve(entry: Dict) -> Dict[str, str]:
    """Resolve a MODELS entry's lazy key/url fields."""
    return {
        "name":         entry["name"],
        "provider":     entry["provider"],
        "api_key":      entry["key"]() if callable(entry.get("key")) else entry.get("api_key", ""),
        "api_base_url": entry["url"]() if callable(entry.get("url")) else entry.get("api_base_url", ""),
    }


def get_observabilities(args) -> List[str]:
    """Parse observability conditions."""
    if args.run == "threshold" and args.p_values:
        p_vals = [float(x) for x in args.p_values.split(",")]
        result = []
        for p in p_vals:
            if p == 0:
                result.append("private")
            elif p >= 1.0:
                result.append("full")
            else:
                result.append(f"partial_{p}")
        return result
    else:
        return args.observability.split(",")


def run_evolutionary(
    models: List[Dict],
    observabilities: List[str],
    args,
    use_random_mutation: bool = False
) -> Dict[str, Any]:
    """Run full evolutionary experiments."""
    all_results = {
        "run_type": "evolutionary"
        if not use_random_mutation else "random_mutation",
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "trials": []
    }

    total = len(models) * len(observabilities) * args.seeds
    trial_num = 0
    start_time = time.time()

    for model_info in models:
        for obs in observabilities:
            for seed in range(args.seeds):
                trial_num += 1
                print(f"\n{'='*60}")
                print(f"[{trial_num}/{total}] "
                      f"Model={model_info['name']}, "
                      f"Obs={obs}, Seed={seed}")
                print(f"{'='*60}")

                # Parse observability for partial conditions
                obs_p = 0.3
                if obs.startswith("partial_"):
                    obs_p = float(obs.split("_")[1])

                pop = EvolutionaryPopulation(
                    population_size=args.population,
                    num_rounds_per_gen=args.rounds,
                    benefit=args.benefit,
                    cost=args.cost,
                    observability=obs,
                    observability_p=obs_p,
                    elite_count=args.elitism,
                    num_eliminate=args.eliminate,
                    tournament_size=args.tournament,
                    llm_provider=model_info["provider"],
                    llm_model=model_info["name"],
                    api_key=model_info.get("api_key", ""),
                    api_base_url=model_info.get("api_base_url", ""),
                    mutation_temperature=args.mutation_temperature,
                    seed=seed,
                    results_dir=args.output
                )

                # Override mutation operator for random mutation control
                if use_random_mutation:
                    pop.mutation_op = RandomMutationOperator()

                results = pop.run_evolution(num_generations=args.generations)
                results["model"] = model_info["name"]
                results["provider"] = model_info["provider"]
                results["observability"] = obs
                results["seed"] = seed

                all_results["trials"].append(results)

                elapsed = time.time() - start_time
                final_gen = results["trajectory"][-1] if results["trajectory"] else {}
                print(f"  Final: coop={final_gen.get('cooperation_rate_mean', 'N/A')}, "
                      f"elapsed={elapsed:.0f}s")

    # Save aggregate
    _save_aggregate(all_results, args.output, "evolutionary")

    return all_results


def run_static_control(
    models: List[Dict],
    observabilities: List[str],
    args
) -> Dict[str, Any]:
    """Run static (no-evolution) control experiment.

    Generates initial population once, runs all rounds without selection/mutation.
    """
    all_results = {
        "run_type": "static_control",
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "trials": []
    }

    total_rounds = args.generations * args.rounds  # Same total rounds
    total = len(models) * len(observabilities) * args.seeds
    trial_num = 0

    for model_info in models:
        for obs in observabilities:
            for seed in range(args.seeds):
                trial_num += 1
                print(f"\n{'='*60}")
                print(f"[{trial_num}/{total}] STATIC "
                      f"Model={model_info['name']}, "
                      f"Obs={obs}, Seed={seed}")
                print(f"{'='*60}")

                obs_p = 0.3
                if obs.startswith("partial_"):
                    obs_p = float(obs.split("_")[1])

                pop = EvolutionaryPopulation(
                    population_size=args.population,
                    num_rounds_per_gen=args.rounds,
                    benefit=args.benefit,
                    cost=args.cost,
                    observability=obs,
                    observability_p=obs_p,
                    elite_count=args.elitism,
                    num_eliminate=0,  # No elimination
                    tournament_size=args.tournament,
                    llm_provider=model_info["provider"],
                    llm_model=model_info["name"],
                    api_key=model_info.get("api_key", ""),
                    api_base_url=model_info.get("api_base_url", ""),
                    seed=seed,
                    results_dir=args.output
                )

                # Initialize once
                pop.initialize_population()

                # Run all rounds without evolution
                trajectory = []
                for gen in range(args.generations):
                    stats = pop.run_generation(gen)
                    trajectory.append(stats)
                    print(f"  Gen {gen}: coop={stats['cooperation_rate_mean']:.3f}")

                results = pop._collect_results()
                results["model"] = model_info["name"]
                results["observability"] = obs
                results["seed"] = seed
                all_results["trials"].append(results)

    _save_aggregate(all_results, args.output, "static_control")
    return all_results


def run_threshold_scan(
    models: List[Dict],
    observabilities: List[str],
    args
) -> Dict[str, Any]:
    """Run critical threshold scan (dense p sweep)."""
    # For threshold scan, use fewer generations for speed
    args.generations = min(args.generations, 5)
    return run_evolutionary(models, observabilities, args, use_random_mutation=False)


def _save_aggregate(
    results: Dict[str, Any],
    output_dir: str,
    run_name: str
):
    """Save aggregate results."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = out_path / f"{run_name}_{timestamp}.json"

    # Strip code strings for compact storage (keep in individual trial files)
    saveable = {"run_type": results["run_type"],
                "timestamp": results["timestamp"],
                "config": results["config"]}
    trials_summary = []
    for trial in results.get("trials", []):
        summary = {
            "model": trial.get("model", ""),
            "observability": trial.get("observability", ""),
            "seed": trial.get("seed", ""),
            "trajectory": trial.get("trajectory", []),
        }
        # Include final generation stats
        if trial.get("final_population"):
            final = trial["final_population"]
            summary["final_mean_fitness"] = sum(
                a["fitness"] for a in final
            ) / len(final)
            summary["final_mean_cooperation"] = sum(
                a["cooperation_rate"] for a in final
            ) / len(final)
            summary["num_valid"] = sum(
                1 for a in final if len(a.get("code", "")) > 20
            )
        trials_summary.append(summary)

    saveable["trials_summary"] = trials_summary

    with open(filepath, 'w') as f:
        json.dump(saveable, f, indent=2, default=str)

    print(f"\nAggregate results saved to: {filepath}")


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"Evolutionary Indirect Reciprocity Experiments")
    print(f"Run type: {args.run}")
    print(f"{'='*60}")

    models = get_models(args.models)
    observabilities = get_observabilities(args)

    print(f"Models: {[m['name'] for m in models]}")
    print(f"Observability conditions: {observabilities}")
    print(f"Population: {args.population}, "
          f"Generations: {args.generations}, "
          f"Rounds/gen: {args.rounds}")
    print(f"Seeds: {args.seeds}")
    print(f"Selection: eliminate {args.eliminate}, "
          f"elitism {args.elitism}, tournament {args.tournament}")

    if args.run == "evolutionary":
        run_evolutionary(models, observabilities, args)
    elif args.run == "threshold":
        run_threshold_scan(models, observabilities, args)
    elif args.run == "static":
        run_static_control(models, observabilities, args)
    elif args.run == "random-mutation":
        run_evolutionary(models, observabilities, args, use_random_mutation=True)

    print(f"\n{'='*60}")
    print("All experiments complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
