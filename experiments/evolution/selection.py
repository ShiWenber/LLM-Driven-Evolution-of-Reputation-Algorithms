"""Selection operators for evolutionary strategy populations.

Implements:
- Tournament selection
- Elitism
- Fitness-proportional ranking
"""

import random
import numpy as np
from typing import List, Tuple

from ..agents.code_agent import CodeAgent


def rank_by_fitness(agents: List[CodeAgent]) -> List[CodeAgent]:
    """Sort agents by fitness in descending order."""
    return sorted(agents, key=lambda a: a.fitness, reverse=True)


def tournament_select(
    agents: List[CodeAgent],
    tournament_size: int = 3,
    num_to_select: int = 1
) -> List[CodeAgent]:
    """
    Select agents using binary tournament selection.
    Higher fitness = higher probability of winning tournament.

    Args:
        agents: Candidate agents
        tournament_size: Number of agents in each tournament
        num_to_select: Number of agents to select

    Returns:
        Selected agents
    """
    if len(agents) == 0:
        return []

    selected = []
    for _ in range(num_to_select):
        candidates = random.sample(
            agents,
            min(tournament_size, len(agents))
        )
        winner = max(candidates, key=lambda a: a.fitness)
        selected.append(winner)

    return selected


def select_survivors(
    agents: List[CodeAgent],
    elite_count: int = 2,
    num_to_eliminate: int = 5,
    tournament_size: int = 3
) -> Tuple[List[CodeAgent], List[CodeAgent], List[CodeAgent]]:
    """
    Select which agents survive, which are eliminated, and which are parents.

    Args:
        agents: Current population
        elite_count: Number of top agents that survive directly (elitism)
        num_to_eliminate: Number of worst agents to eliminate
        tournament_size: Tournament size for selection

    Returns:
        (survivors, eliminated, parents) where:
        - survivors: agents that survive to next generation
        - eliminated: agents that are removed
        - parents: top agents eligible for reproduction (elites + tournament winners)
    """
    n = len(agents)
    ranked = rank_by_fitness(agents)

    # Elites survive directly
    elites = ranked[:elite_count]

    # Bottom agents are eliminated
    eliminated = ranked[-num_to_eliminate:] if num_to_eliminate > 0 else []

    # Middle agents + elites survive
    middle = ranked[elite_count:n - num_to_eliminate] if num_to_eliminate > 0 else ranked[elite_count:]
    survivors = elites + middle

    # Parents are drawn from survivors (for reproduction)
    num_needed = num_to_eliminate  # Need to replace eliminated agents
    parents = tournament_select(survivors, tournament_size, num_needed)

    return survivors, eliminated, parents


def compute_fitness_stats(agents: List[CodeAgent]) -> dict:
    """Compute fitness statistics for the population."""
    fitnesses = [a.fitness for a in agents]
    coop_rates = [a.cooperation_rate for a in agents]
    valid_count = sum(1 for a in agents if a.is_valid)

    return {
        "fitness_mean": float(np.mean(fitnesses)),
        "fitness_std": float(np.std(fitnesses)),
        "fitness_max": float(np.max(fitnesses)),
        "fitness_min": float(np.min(fitnesses)),
        "cooperation_rate_mean": float(np.mean(coop_rates)),
        "cooperation_rate_std": float(np.std(coop_rates)),
        "valid_agents": valid_count,
        "total_agents": len(agents),
    }
