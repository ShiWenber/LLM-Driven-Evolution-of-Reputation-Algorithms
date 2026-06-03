"""Comprehensive analysis of evolutionary experiment results.

Key comparison: PRIVATE vs FULL observability under evaluate/decide architecture.
"""

import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = RESULTS_DIR


def load_aggregate_results():
    """Load the most recent aggregate results."""
    files = sorted(RESULTS_DIR.glob("evolutionary_*.json"), key=os.path.getmtime, reverse=True)
    if not files:
        raise FileNotFoundError("No aggregate results found")
    with open(files[0]) as f:
        return json.load(f)


def load_trial_results(trial_info):
    """Load a specific trial result file."""
    model = trial_info["model"]
    obs = trial_info["observability"]
    seed = trial_info["seed"]

    # Find matching file
    for f in sorted(RESULTS_DIR.glob(f"evo_{obs}_{model}_*.json"), key=os.path.getmtime, reverse=True):
        # Check if this file matches by reading it
        with open(f) as fh:
            data = json.load(fh)
            if data.get("config", {}).get("seed") == seed:
                return data
    return None


def make_cooperation_trajectory_plot(data):
    """Plot cooperation trajectories for all trials, grouped by condition."""
    trials = data["trials_summary"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = ['#2196F3', '#4CAF50', '#FF5722']
    conditions = ["private", "full"]
    titles = ["PRIVATE Observability", "FULL Observability"]

    for ax, cond, title in zip(axes, conditions, titles):
        cond_trials = [t for t in trials if t["observability"] == cond]

        for i, trial in enumerate(cond_trials):
            gens = [g["generation"] for g in trial["trajectory"]]
            coops = [g["cooperation_rate_mean"] for g in trial["trajectory"]]
            label = f"Seed {trial['seed']}"
            ax.plot(gens, coops, 'o-', color=colors[i], label=label,
                   linewidth=2, markersize=6, alpha=0.85)

        # Add mean line across seeds
        if len(cond_trials) > 1:
            all_gens = list(range(10))
            all_coops = np.zeros((len(cond_trials), 10))
            for i, trial in enumerate(cond_trials):
                for g in trial["trajectory"]:
                    all_coops[i, g["generation"]] = g["cooperation_rate_mean"]
            mean_coop = np.mean(all_coops, axis=0)
            ax.plot(all_gens, mean_coop, 'k--', linewidth=2.5, label='Mean', alpha=0.7)

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Cooperation Rate", fontsize=12)
        ax.set_title(title + f" (N=15, T=40)", fontsize=13, fontweight='bold')
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig_cooperation_trajectories.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def make_comparison_figure(data):
    """Make a summary comparison figure: PRIVATE vs FULL."""
    trials = data["trials_summary"]

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    colors_private = ['#90CAF9', '#A5D6A7', '#FFAB91']
    colors_full = ['#1565C0', '#2E7D32', '#BF360C']

    # Panel A: Cooperation trajectories with mean ± std
    ax_a = fig.add_subplot(gs[0, :])

    # Compute per-generation stats
    for cond, label, color, ls in [
        ("private", "PRIVATE", "#E53935", "--"),
        ("full", "FULL", "#1565C0", "-")
    ]:
        cond_trials = [t for t in trials if t["observability"] == cond]
        gens = np.arange(10)
        all_coops = np.zeros((len(cond_trials), 10))
        for i, trial in enumerate(cond_trials):
            for g in trial["trajectory"]:
                all_coops[i, g["generation"]] = g["cooperation_rate_mean"]

        mean = np.mean(all_coops, axis=0)
        std = np.std(all_coops, axis=0)

        ax_a.fill_between(gens, mean - std, mean + std, alpha=0.15, color=color)
        ax_a.plot(gens, mean, ls, color=color, linewidth=2.5, label=f"{label} (n=3 seeds)")

    ax_a.set_xlabel("Generation", fontsize=12)
    ax_a.set_ylabel("Mean Cooperation Rate ± 1σ", fontsize=12)
    ax_a.set_title("A. Cooperation Trajectories: PRIVATE vs FULL Observability",
                   fontsize=13, fontweight='bold')
    ax_a.legend(fontsize=11, loc='center left')
    ax_a.set_ylim(-0.05, 1.15)
    ax_a.grid(True, alpha=0.3)

    # Panel B: Final cooperation rates by seed
    ax_b = fig.add_subplot(gs[1, 0])

    private_finals = []
    full_finals = []
    for trial in trials:
        final_coop = trial["trajectory"][-1]["cooperation_rate_mean"]
        best = max(g["cooperation_rate_mean"] for g in trial["trajectory"])
        if trial["observability"] == "private":
            private_finals.append((final_coop, best))
        else:
            full_finals.append((final_coop, best))

    width = 0.6
    labels_cond = ["PRIVATE", "FULL"]
    all_finals = [private_finals, full_finals]
    colors_list = ["#E53935", "#1565C0"]

    for i, (label, finals, color) in enumerate(zip(labels_cond, all_finals, colors_list)):
        final_vals = [f[0] for f in finals]
        best_vals = [f[1] for f in finals]

        ax_b.bar(i, np.mean(final_vals), width, color=color, alpha=0.8,
                yerr=np.std(final_vals), capsize=5)

        # Scatter individual seeds
        jitter = np.linspace(-0.1, 0.1, len(final_vals))
        ax_b.scatter([i] * 3 + jitter, final_vals,
                    color=color, marker='o', s=60, zorder=5, edgecolor='white', linewidth=1)

    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(['PRIVATE', 'FULL'], fontsize=12)
    ax_b.set_ylabel('Cooperation Rate', fontsize=12)
    ax_b.set_title('B. Final Cooperation (bar=mean, dots=seeds)', fontsize=12, fontweight='bold')
    ax_b.set_ylim(-0.05, 1.15)
    ax_b.grid(True, alpha=0.3, axis='y')

    # Panel C: Statistical summary table-like bar chart
    ax_c = fig.add_subplot(gs[1, 1])

    # Compute stats
    stats = {}
    for cond in ["private", "full"]:
        cond_trials = [t for t in trials if t["observability"] == cond]
        finals = [t["trajectory"][-1]["cooperation_rate_mean"] for t in cond_trials]
        bests = [max(g["cooperation_rate_mean"] for g in t["trajectory"]) for t in cond_trials]
        fitness = [t["trajectory"][-1]["fitness_mean"] for t in cond_trials]
        best_fitness = [max(g["fitness_max"] for g in t["trajectory"]) for t in cond_trials]

        stats[cond] = {
            "final_mean": np.mean(finals),
            "final_std": np.std(finals),
            "best_mean": np.mean(bests),
            "best_std": np.std(bests),
            "fitness_mean": np.mean(fitness),
            "fitness_std": np.std(fitness),
            "best_fitness": np.mean(best_fitness),
        }

    # Simple bar chart comparison
    metrics = ["Final\nCooperation", "Best\nCooperation", "Final\nFitness", "Best\nFitness"]
    private_vals = [
        stats["private"]["final_mean"],
        stats["private"]["best_mean"],
        stats["private"]["fitness_mean"],
        stats["private"]["best_fitness"],
    ]
    full_vals = [
        stats["full"]["final_mean"],
        stats["full"]["best_mean"],
        stats["full"]["fitness_mean"],
        stats["full"]["best_fitness"],
    ]
    private_errs = [
        stats["private"]["final_std"],
        stats["private"]["best_std"],
        stats["private"]["fitness_std"],
        np.std([max(g["fitness_max"] for g in t["trajectory"])
                for t in trials if t["observability"] == "private"]),
    ]
    full_errs = [
        stats["full"]["final_std"],
        stats["full"]["best_std"],
        stats["full"]["fitness_std"],
        np.std([max(g["fitness_max"] for g in t["trajectory"])
                for t in trials if t["observability"] == "full"]),
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax_c.bar(x - width/2, private_vals, width, color="#E53935", alpha=0.8,
            label="PRIVATE", yerr=private_errs, capsize=5)
    ax_c.bar(x + width/2, full_vals, width, color="#1565C0", alpha=0.8,
            label="FULL", yerr=full_errs, capsize=5)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(metrics, fontsize=11)
    ax_c.set_title("C. Summary Statistics (mean ± 1σ)", fontsize=13, fontweight='bold')
    ax_c.legend(fontsize=10)
    ax_c.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = OUTPUT_DIR / "fig_comparison_summary.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def make_fitness_trajectory_plot(data):
    """Plot fitness trajectories."""
    trials = data["trials_summary"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, cond, title in zip(axes, ["private", "full"],
                               ["PRIVATE Observability", "FULL Observability"]):
        cond_trials = [t for t in trials if t["observability"] == cond]

        for trial in cond_trials:
            gens = [g["generation"] for g in trial["trajectory"]]
            fitness = [g["fitness_mean"] for g in trial["trajectory"]]
            ax.plot(gens, fitness, 'o-', linewidth=2, markersize=6, alpha=0.85,
                   label=f"Seed {trial['seed']}")

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Mean Fitness", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig_fitness_trajectories.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def print_strategy_analysis(data):
    """Print detailed strategy analysis."""
    trials = data["trials_summary"]

    print("\n" + "="*60)
    print("STRATEGY ANALYSIS")
    print("="*60)

    for cond in ["full", "private"]:
        print(f"\n--- {cond.upper()} Observability ---")
        cond_trials = [t for t in trials if t["observability"] == cond]

        for trial in cond_trials:
            print(f"\nSeed {trial['seed']}:")
            print(f"  Final coop: {trial['trajectory'][-1]['cooperation_rate_mean']:.3f}")
            print(f"  Final fitness: {trial['trajectory'][-1]['fitness_mean']:.1f}")
            print(f"  Max coop ever: {max(g['cooperation_rate_mean'] for g in trial['trajectory']):.3f}")
            print(f"  Num survivors with >0 coop: {trial['final_mean_cooperation']:.3f}")


def print_summary_table(data):
    """Print a clean summary table."""
    trials = data["trials_summary"]

    print("\n" + "="*70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("Architecture: evaluate/decide pair with private reputation stores")
    print(f"Config: N=15, G=10, T=40, b/c=3/1, eliminate 4, elite 2")
    print("="*70)

    print(f"\n{'Condition':<12} {'Seed':<6} {'Gen0':<8} {'Final':<8} {'Max':<8} {'Fitness':<8} {'Result':<10}")
    print("-"*70)

    for cond in ["private", "full"]:
        cond_trials = [t for t in trials if t["observability"] == cond]

        for trial in cond_trials:
            gen0 = trial["trajectory"][0]["cooperation_rate_mean"]
            final = trial["trajectory"][-1]["cooperation_rate_mean"]
            max_coop = max(g["cooperation_rate_mean"] for g in trial["trajectory"])
            fitness = trial["trajectory"][-1]["fitness_mean"]

            if final > 0.5:
                result = "EMERGED"
            elif final < 0.1:
                result = "COLLAPSED"
            else:
                result = "PARTIAL"

            print(f"{cond.upper():<12} {trial['seed']:<6} {gen0:<8.3f} {final:<8.3f} {max_coop:<8.3f} {fitness:<8.1f} {result:<10}")

        # Mean
        finals = [t["trajectory"][-1]["cooperation_rate_mean"] for t in cond_trials]
        maxs = [max(g["cooperation_rate_mean"] for g in t["trajectory"]) for t in cond_trials]
        fit_vals = [t["trajectory"][-1]["fitness_mean"] for t in cond_trials]
        print(f"{'':<12} {'MEAN':<6} {'':<8} {np.mean(finals):<8.3f} {np.mean(maxs):<8.3f} {np.mean(fit_vals):<8.1f}")
        print()


def main():
    print("Loading aggregate results...")
    data = load_aggregate_results()

    print_summary_table(data)
    print_strategy_analysis(data)

    print("\nGenerating figures...")
    make_cooperation_trajectory_plot(data)
    make_fitness_trajectory_plot(data)
    make_comparison_figure(data)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
