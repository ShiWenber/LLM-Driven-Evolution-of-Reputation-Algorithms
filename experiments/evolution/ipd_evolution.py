"""IPD evolutionary runner — Willis 2025 baseline.

Implements Moran-process-style evolutionary dynamics in a 2-player IPD
with full information. Uses the same LLM-driven mutation operator as
the donor game experiments to enable direct comparison.

This is PAPER_DRAFT.md Experiment 5: a direct sanity check that the
LLM-driven evolutionary protocol reproduces / differs from the
behaviours reported by Willis et al. (2025).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from .ipd_game import IPDGame, ipd_tournament_to_fitness
from ..agents.code_agent import CodeAgent
from ..agents.prompts import build_init_prompt
from ..evolution.mutation import MutationOperator
from ..sandbox.validator import validate_strategy_code, CodeValidationError


@dataclass
class IPDTrialConfig:
    """Configuration for a single IPD evolutionary trial."""
    population_size: int = 12  # Moran population, matches Willis et al.
    num_generations: int = 10
    ipd_rounds_per_match: int = 1000  # Standard IPD match length
    mutation_rate: float = 0.05  # Probability an agent gets mutated each gen
    seed: int = 0
    model: str = "deepseek-v4-flash"
    noise: float = 0.0
    use_random_mutation: bool = False
    output_dir: str = "results"


@dataclass
class IPDTrialResult:
    """Result of one IPD evolutionary trial."""
    config: IPDTrialConfig
    initial_strategies: List[str] = field(default_factory=list)
    final_strategies: List[str] = field(default_factory=list)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    # trajectory[i] = {generation, mean_cooperation, mean_payoff,
    #                  cooperation_rates_per_agent}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "initial_strategies": self.initial_strategies,
            "final_strategies": self.final_strategies,
            "trajectory": self.trajectory,
        }


class IPDEvolutionaryRunner:
    """Run LLM-driven evolution on IPD with Moran-style selection.

    Procedure per generation:
        1. Tournament: every agent plays every other agent in IPD
        2. Fitness: per-agent mean payoff (shifted positive)
        3. Replacement: with prob `mutation_rate`, replace a random
           agent with a mutated copy of another. Otherwise do nothing
           (Moran process has drift via birth-death, not deterministic
           replacement).
    """

    def __init__(
        self,
        config: IPDTrialConfig,
        mutation_op: MutationOperator,
    ):
        self.config = config
        self.mutation_op = mutation_op
        self.rng = random.Random(config.seed)
        np.random.seed(config.seed)

        # Initial strategy pool: LLM generates diverse strategy pairs
        # We re-use one LLM call to get the initial population.
        self._initial_strategies: List[str] = []
        self._agents: List[CodeAgent] = []

        self.game = IPDGame(
            num_rounds=config.ipd_rounds_per_match,
            noise=config.noise,
        )

    def _initialize_population(self):
        """Use LLM to generate initial diverse strategies for IPD."""
        # Use the existing build_init_prompt infrastructure
        prompt = build_init_prompt(
            num_strategies=self.config.population_size,
            population_size=2,  # Two-player IPD
            num_rounds=self.config.ipd_rounds_per_match,
            compact=True,
        )
        # Note: the original prompt is donor-game-flavoured; for IPD we
        # would adapt it. For the Willis-baseline experiment we accept
        # the existing prompt as a "general iterated game" prompt; the
        # LLM is general enough to produce IPD-suitable strategies.
        from experiments.config.load_env import get_api_key, get_base_url
        from openai import OpenAI
        client = OpenAI(
            api_key=get_api_key("deepseek"),
            base_url=get_base_url("deepseek"),
        )
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": (
                    "You are a Python programmer. "
                    "Write a code block containing an `evaluate` and a "
                    "`decide` function. In Iterated Prisoner's Dilemma "
                    "(IPD), only `decide` is called. The strategy should "
                    "decide True (cooperate) or False (defect) based on "
                    "the history of past rounds (my_history, with each "
                    "entry containing 'action' and 'partner_action')."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=8000,
        )
        content = response.choices[0].message.content or ""

        # Split on "# ---" markers as in donor-game initialisation
        from ..sandbox.validator import clean_code
        chunks = [c.strip() for c in content.split("# ---") if c.strip()]
        strategies: List[str] = []
        for chunk in chunks:
            cleaned = clean_code(chunk)
            try:
                validate_strategy_code(cleaned)
                strategies.append(cleaned)
            except CodeValidationError:
                continue
        # Pad if LLM produced fewer than needed
        while len(strategies) < self.config.population_size:
            strategies.append(_FALLBACK_STRATEGY)
        strategies = strategies[:self.config.population_size]
        self._initial_strategies = strategies

    def _make_agent(self, code: str) -> CodeAgent:
        return CodeAgent(agent_id=-1, code=code)

    def _build_agents(self, strategies: List[str]) -> List[CodeAgent]:
        agents = []
        for i, code in enumerate(strategies):
            a = self._make_agent(code)
            a.agent_id = i
            agents.append(a)
        return agents

    def _run_tournament(self) -> Dict[str, Any]:
        return self.game.all_play_all(self._agents, seed=self.config.seed)

    def _mutate_one(self, parent: CodeAgent) -> Optional[CodeAgent]:
        """Mutate a parent strategy, return child CodeAgent (or None on failure)."""
        new_code = self.mutation_op.mutate(
            parent_code=parent.code,
            parent_fitness=parent.fitness,
            population_size=self.config.population_size,
        )
        if new_code is None:
            return None
        child = self._make_agent(new_code)
        return child

    def _replacement_step(self, tournament_result: Dict[str, Any]) -> List[str]:
        """Apply one Moran-style replacement step. Return new strategy list."""
        fitness = ipd_tournament_to_fitness(
            tournament_result, self.config.population_size
        )
        fitness_sum = fitness.sum()
        if fitness_sum <= 0:
            return [a.code for a in self._agents]

        # Sample parent (clone candidate) with probability proportional to fitness
        parent_idx = self.rng.choices(
            range(self.config.population_size), weights=fitness / fitness_sum
        )[0]

        # Decide whether to mutate
        if self.rng.random() < self.config.mutation_rate:
            child = self._mutate_one(self._agents[parent_idx])
            if child is not None:
                # Replace a random agent (not the parent)
                victim_idx = self.rng.randrange(self.config.population_size)
                while victim_idx == parent_idx:
                    victim_idx = self.rng.randrange(self.config.population_size)
                self._agents[victim_idx] = child
                self._agents[victim_idx].agent_id = victim_idx

        return [a.code for a in self._agents]

    def run(self) -> IPDTrialResult:
        """Run the full evolutionary trial."""
        self._initialize_population()
        self._agents = self._build_agents(self._initial_strategies)

        trajectory: List[Dict[str, Any]] = []

        for gen in range(self.config.num_generations):
            # Update agent fitnesses
            for a in self._agents:
                a.fitness = 0.0
            tournament = self._run_tournament()
            for i, stats in enumerate(tournament["per_agent"]):
                self._agents[i].fitness = stats["mean_payoff"]

            trajectory.append({
                "generation": gen,
                "mean_cooperation": tournament["tournament_mean_cooperation"],
                "mean_payoff": tournament["tournament_mean_payoff"],
                "cooperation_rates_per_agent": [
                    stats["cooperation_rate"] for stats in tournament["per_agent"]
                ],
                "payoffs_per_agent": [
                    stats["mean_payoff"] for stats in tournament["per_agent"]
                ],
            })

            if gen < self.config.num_generations - 1:
                self._replacement_step(tournament)

        return IPDTrialResult(
            config=self.config,
            initial_strategies=self._initial_strategies,
            final_strategies=[a.code for a in self._agents],
            trajectory=trajectory,
        )


# A simple baseline strategy used as fallback if LLM initialisation fails
_FALLBACK_STRATEGY = '''
def evaluate(current_reputation, observation, my_history, round_num):
    return 0.0

def decide(recipient_reputation, round_num, my_history):
    # Tit-for-tat
    if not my_history:
        return True
    return my_history[-1].get("partner_action") == "donate"
'''


def run_ipd_trial_cli(
    seed: int = 0,
    population_size: int = 12,
    num_generations: int = 10,
    model: str = "deepseek-v4-flash",
    noise: float = 0.0,
    use_random_mutation: bool = False,
    output_dir: str = "results",
):
    """CLI entry point for a single IPD evolutionary trial."""
    from ..evolution.mutation import MutationOperator, RandomMutationOperator

    config = IPDTrialConfig(
        population_size=population_size,
        num_generations=num_generations,
        seed=seed,
        model=model,
        noise=noise,
        use_random_mutation=use_random_mutation,
        output_dir=output_dir,
    )

    if use_random_mutation:
        mutation_op = RandomMutationOperator()
    else:
        mutation_op = MutationOperator(model=model)

    runner = IPDEvolutionaryRunner(config, mutation_op)
    result = runner.run()

    # Save to disk
    out = Path(output_dir) / "ipd_baseline"
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"trial_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"[IPD trial seed={seed}] final cooperation = "
          f"{result.trajectory[-1]['mean_cooperation']:.3f}, "
          f"mean payoff = {result.trajectory[-1]['mean_payoff']:.3f}")
    print(f"  saved to {out_path}")
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--population", type=int, default=12)
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--model", type=str, default="deepseek-v4-flash")
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--random-mutation", action="store_true")
    p.add_argument("--output", type=str, default="results")
    args = p.parse_args()
    run_ipd_trial_cli(
        seed=args.seed,
        population_size=args.population,
        num_generations=args.generations,
        model=args.model,
        noise=args.noise,
        use_random_mutation=args.random_mutation,
        output_dir=args.output,
    )
