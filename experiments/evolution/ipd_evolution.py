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

from ..game.ipd_game import (
    IPDGame,
    IPDStrategy,
    ipd_tournament_to_fitness,
    load_ipd_strategy_from_code,
)
from ..sandbox.validator import clean_code, validate_strategy_code, CodeValidationError


@dataclass
class IPDTrialConfig:
    """Configuration for a single IPD evolutionary trial."""
    population_size: int = 12
    num_generations: int = 10
    ipd_rounds_per_match: int = 1000
    mutation_rate: float = 0.05
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "initial_strategies": self.initial_strategies,
            "final_strategies": self.final_strategies,
            "trajectory": self.trajectory,
        }


class IPDEvolutionaryRunner:
    """Run LLM-driven evolution on IPD with Moran-style selection.

    Per generation:
        1. Tournament: every agent plays every other agent in IPD
        2. Fitness: per-agent mean payoff (shifted positive)
        3. Replacement: with prob `mutation_rate`, replace a random
           agent with a mutated copy of another. Otherwise no change.
    """

    def __init__(
        self,
        config: IPDTrialConfig,
        mutation_op,
    ):
        self.config = config
        self.mutation_op = mutation_op
        self.rng = random.Random(config.seed)
        np.random.seed(config.seed)

        self._initial_strategies: List[str] = []
        self._agent_codes: List[str] = []
        self._agents: List[IPDStrategy] = []

        self.game = IPDGame(
            num_rounds=config.ipd_rounds_per_match,
            noise=config.noise,
        )

    def _initialize_population(self):
        """LLM generates diverse IPD strategies (or fallback if LLM unavailable)."""
        from experiments.config.load_env import get_api_key
        from openai import OpenAI
        from ..agents.prompts import build_init_prompt

        if not get_api_key("deepseek"):
            print("[ipd] no API key — using fallback strategies")
            self._initial_strategies = [_FALLBACK_TIT_FOR_TAT] * self.config.population_size
            return

        prompt = build_init_prompt(
            num_strategies=self.config.population_size,
            population_size=2,
            num_rounds=self.config.ipd_rounds_per_match,
            compact=True,
        )
        client = OpenAI(
            api_key=get_api_key("deepseek"),
            base_url=__import__("experiments.config.load_env", fromlist=["get_base_url"]).get_base_url("deepseek"),
        )
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": (
                    "You are a Python programmer. "
                    "Write a code block defining a function `decide(my_history, round_num) -> bool` "
                    "for the Iterated Prisoner's Dilemma. "
                    "my_history is a list of dicts each with keys 'action' and 'partner_action'. "
                    "Return True to cooperate, False to defect. "
                    "Be creative — write diverse strategies."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=8000,
        )
        content = response.choices[0].message.content or ""
        chunks = [c.strip() for c in content.split("# ---") if c.strip()]
        strategies: List[str] = []
        for chunk in chunks:
            cleaned = clean_code(chunk)
            try:
                validate_strategy_code(cleaned)
                if load_ipd_strategy_from_code(cleaned) is not None:
                    strategies.append(cleaned)
            except CodeValidationError:
                continue
        while len(strategies) < self.config.population_size:
            strategies.append(_FALLBACK_TIT_FOR_TAT)
        strategies = strategies[:self.config.population_size]
        self._initial_strategies = strategies

    def _build_agents(self) -> List[IPDStrategy]:
        agents: List[IPDStrategy] = []
        for code in self._agent_codes:
            strategy = load_ipd_strategy_from_code(code)
            if strategy is None:
                strategy = load_ipd_strategy_from_code(_FALLBACK_TIT_FOR_TAT)
                self._agent_codes[self._agent_codes.index(code)] = _FALLBACK_TIT_FOR_TAT
            agents.append(strategy)
        return agents

    def _run_tournament(self) -> Dict[str, Any]:
        return self.game.all_play_all(self._agents, seed=self.config.seed)

    def _mutate_one(self, parent_code: str) -> Optional[str]:
        """Mutate a parent code string, return child code (or None on failure)."""
        return self.mutation_op.mutate(
            parent_code=parent_code,
            parent_fitness=0.0,  # fitness tracked by selection, not passed here
            population_size=self.config.population_size,
        )

    def _replacement_step(self, tournament_result: Dict[str, Any]):
        """Apply one Moran-style replacement step in-place."""
        fitness = ipd_tournament_to_fitness(
            tournament_result, self.config.population_size
        )
        fitness_sum = fitness.sum()
        if fitness_sum <= 0:
            return
        probs = fitness / fitness_sum

        if self.rng.random() < self.config.mutation_rate:
            parent_idx = self.rng.choices(
                range(self.config.population_size), weights=probs
            )[0]
            new_code = self._mutate_one(self._agent_codes[parent_idx])
            if new_code is not None and load_ipd_strategy_from_code(new_code) is not None:
                victim_idx = self.rng.randrange(self.config.population_size)
                while victim_idx == parent_idx:
                    victim_idx = self.rng.randrange(self.config.population_size)
                self._agent_codes[victim_idx] = new_code
                self._agents = self._build_agents()

    def run(self) -> IPDTrialResult:
        self._initialize_population()
        self._agent_codes = list(self._initial_strategies)
        self._agents = self._build_agents()

        trajectory: List[Dict[str, Any]] = []

        for gen in range(self.config.num_generations):
            tournament = self._run_tournament()
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
            initial_strategies=list(self._initial_strategies),
            final_strategies=list(self._agent_codes),
            trajectory=trajectory,
        )


# A simple baseline strategy used as fallback if LLM initialisation fails
_FALLBACK_TIT_FOR_TAT = '''
def evaluate(current_reputation, observation, my_history, round_num):
    return 0.0

def decide(recipient_reputation, round_num, my_history):
    """Tit-for-tat: cooperate on first move, then mirror opponent."""
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
