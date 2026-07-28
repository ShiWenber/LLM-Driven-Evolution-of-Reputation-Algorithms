"""Population + LLM-driven evolution for the v2 quantitative interface.

Mirrors the v1 EvolutionaryPopulation but uses the v2 QuantitativeAgent
and V2DonorGame.
"""
from __future__ import annotations
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

from .agent import QuantitativeAgent
from .executor import V2StrategyExecutor
from .game import V2DonorGame
from .prompts import INIT_PROMPT_V2, MUTATION_PROMPT_V2
from .baselines import get_baseline


# Fallback strategies when LLM fails
FALLBACK_STRATEGIES = [
    # Always cooperate
    '''
def evaluate(donor_reputation, recipient_reputation, donor_action, recipient_action, my_reputation):
    return donor_reputation
def decide(my_reputation, opponent_reputation):
    return True
''',
    # Always defect
    '''
def evaluate(donor_reputation, recipient_reputation, donor_action, recipient_action, my_reputation):
    return donor_reputation
def decide(my_reputation, opponent_reputation):
    return False
''',
]


def _extract_code_from_response(text: str) -> Optional[str]:
    """Strip markdown fences if any, return the Python code body."""
    text = text.strip()
    if text.startswith("```"):
        # Drop the first fence line
        lines = text.split("\n")
        # Remove first and last ``` lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip() or None


class V2EvolutionaryPopulation:
    """Manage a population of v2 QuantitativeAgent across generations."""

    def __init__(
        self,
        population_size: int = 15,
        num_rounds_per_gen: int = 30,
        benefit: float = 2.0,
        cost: float = 1.0,
        observability: str = "full",
        observability_p: float = 1.0,
        elite_count: int = 2,
        num_eliminate: int = 5,
        tournament_size: int = 3,
        llm_provider: str = "openai",
        llm_model: str = "deepseek-v4-flash",
        api_key: str = "",
        api_base_url: str = "",
        mutation_temperature: float = 0.8,
        seed: int = 42,
        results_dir: str = "results",
        use_baseline: Optional[str] = None,
        # If use_baseline is set, all agents use that baseline strategy and
        # the LLM is not used. If None, LLM evolution runs.
    ):
        self.population_size = population_size
        self.num_rounds_per_gen = num_rounds_per_gen
        self.benefit = benefit
        self.cost = cost
        self.observability = observability
        self.observability_p = observability_p
        self.elite_count = elite_count
        self.num_eliminate = num_eliminate
        self.tournament_size = tournament_size
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.mutation_temperature = mutation_temperature
        self.seed = seed
        self.results_dir = Path(results_dir)
        self.use_baseline = use_baseline
        self.agents: List[QuantitativeAgent] = []
        self.rng = random.Random(seed)
        # LLM client (lazy)
        self._llm_client = None

    def _get_llm_client(self):
        if self._llm_client is not None:
            return self._llm_client
        import openai
        self._llm_client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url,
        )
        return self._llm_client

    def _call_llm(self, system_msg: str, user_msg: str, max_retries: int = 3) -> Optional[str]:
        client = self._get_llm_client()
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self.mutation_temperature,
                    max_tokens=4000,
                )
                content = resp.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                print(f"  [LLM error attempt {attempt+1}]: {e}")
                time.sleep(2 ** attempt)
        return None

    def _make_agent(self, code: str, agent_id: int) -> QuantitativeAgent:
        executor = V2StrategyExecutor(code)
        a = QuantitativeAgent(agent_id, code, executor=executor)
        return a

    def _init_population_llm(self):
        """Generate N strategies via LLM."""
        print(f"  Initializing population via LLM ({self.population_size} agents)...")
        client = self._get_llm_client()
        user_msg = INIT_PROMPT_V2
        for i in range(self.population_size):
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": "You are a Python programmer. Output only valid Python code."},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=self.mutation_temperature,
                        max_tokens=4000,
                    )
                    content = resp.choices[0].message.content
                    code = _extract_code_from_response(content)
                    if code:
                        # Validate
                        try:
                            agent = self._make_agent(code, i)
                            self.agents.append(agent)
                            break
                        except Exception as e:
                            print(f"  [init agent {i} validation fail]: {e}")
                except Exception as e:
                    print(f"  [init agent {i} LLM error attempt {attempt+1}]: {e}")
                    time.sleep(2)
            else:
                # All attempts failed; use random fallback
                fb = self.rng.choice(FALLBACK_STRATEGIES)
                agent = self._make_agent(fb, i)
                self.agents.append(agent)
                print(f"  [init agent {i}] using FALLBACK strategy")

    def _init_population_baseline(self):
        """All agents use the same baseline strategy."""
        assert self.use_baseline is not None
        code = get_baseline(self.use_baseline)
        for i in range(self.population_size):
            try:
                agent = self._make_agent(code, i)
                self.agents.append(agent)
            except Exception as e:
                print(f"  [init baseline {i}] validation fail: {e}")
                fb = self.rng.choice(FALLBACK_STRATEGIES)
                self.agents.append(self._make_agent(fb, i))
        print(f"  Initialized {len(self.agents)} agents with baseline '{self.use_baseline}'")

    def _mutate(self, parent_code: str, parent_fitness: float) -> str:
        """LLM-driven mutation of parent code."""
        user_msg = MUTATION_PROMPT_V2.format(fitness=parent_fitness, parent_code=parent_code)
        for attempt in range(3):
            content = self._call_llm("You are a Python programmer. Output only valid Python code.", user_msg)
            code = _extract_code_from_response(content) if content else None
            if code:
                # Validate
                try:
                    V2StrategyExecutor(code)
                    return code
                except Exception as e:
                    print(f"  [mutate validation fail attempt {attempt+1}]: {e}")
        # Fallback: tiny variation of parent
        return parent_code  # No change on failure

    def _run_one_generation(self) -> Dict:
        """Run a single generation. Returns per-gen stats."""
        game = V2DonorGame(
            population_size=self.population_size,
            benefit=self.benefit,
            cost=self.cost,
            observability=self.observability,
            observability_p=self.observability_p,
            seed=self.seed + self.round_num_offset,  # different seed per gen
        )
        # Per-gen unique seed
        gen_seed = self.rng.randrange(10**9)
        game.rng = random.Random(gen_seed)
        game.setup_population(self.agents)
        # For baseline mode, do NOT reset agent reputation (it's built up)
        # For LLM mode, do NOT reset either.
        # Reset only the per-gen tracking on each agent
        for a in self.agents:
            a.reset_for_generation()
        # Run T rounds
        # NOTE: V2DonorGame.run_generation() uses population_size as T.
        # If T != population_size, we override the loop here:
        T = self.num_rounds_per_gen
        game.round_num = 0
        game.payoffs = [0.0] * self.population_size
        game._global_log = []
        for _ in range(T):
            game.play_round()
            game.distribute_observations_and_self_judgments()
        coop_count = sum(1 for inter in game._global_log if inter["donor_action"] == "cooperate")
        coop_rate = coop_count / max(1, len(game._global_log))
        # Sum payoffs for fitness
        fitness = [p for p in game.payoffs]
        return {
            "cooperation_rate_mean": coop_rate,
            "n_interactions": len(game._global_log),
            "round_num": T,
            "payoffs": fitness,
        }

    @property
    def round_num_offset(self) -> int:
        return getattr(self, "_round_offset", 0)

    def run_evolution(self, num_generations: int) -> Dict:
        """Run num_generations and return aggregate results."""
        # Initialize
        if self.use_baseline:
            self._init_population_baseline()
        else:
            self._init_population_llm()
        trajectory: List[Dict] = []
        final_population: List[Dict] = []
        # Generation 0: just initialize; we record initial stats by running
        # one generation with the initial population (no selection/mutation yet)
        for gen in range(num_generations):
            self._round_offset = gen
            stats = self._run_one_generation()
            # Update fitness on agents
            for i, a in enumerate(self.agents):
                a.fitness = stats["payoffs"][i] if i < len(stats["payoffs"]) else 0.0
            trajectory.append({
                "generation": gen,
                "cooperation_rate_mean": stats["cooperation_rate_mean"],
                "n_interactions": stats["n_interactions"],
                "fitness_mean": sum(stats["payoffs"]) / max(1, len(stats["payoffs"])),
                "fitness_max": max(stats["payoffs"]) if stats["payoffs"] else 0.0,
                "population": [
                    {
                        "agent_id": a.agent_id,
                        "code": a.code,
                        "fitness": a.fitness,
                        "cooperation_rate": a.cooperation_rate,
                        "self_reputation": a.get_self_reputation(),
                    } for a in self.agents
                ],
            })
            print(f"  Gen {gen}: coop={stats['cooperation_rate_mean']:.3f}, "
                  f"fitness_mean={sum(stats['payoffs'])/max(1,len(stats['payoffs'])):.1f}")
            # Selection + mutation (only for LLM mode)
            if not self.use_baseline and gen < num_generations - 1:
                self._select_and_reproduce()
        # Build final population
        for a in self.agents:
            final_population.append({
                "agent_id": a.agent_id,
                "code": a.code,
                "fitness": a.fitness,
                "cooperation_rate": a.cooperation_rate,
                "self_reputation": a.get_self_reputation(),
            })
        return {
            "trajectory": trajectory,
            "final_population": final_population,
            "config": {
                "schema_version": 2,
                "population_size": self.population_size,
                "num_rounds_per_gen": self.num_rounds_per_gen,
                "benefit": self.benefit,
                "cost": self.cost,
                "observability": self.observability,
                "observability_p": self.observability_p,
                "elite_count": self.elite_count,
                "num_eliminate": self.num_eliminate,
                "tournament_size": self.tournament_size,
                "llm_model": self.llm_model,
                "seed": self.seed,
                "use_baseline": self.use_baseline,
                "num_generations": num_generations,
            },
        }

    def _select_and_reproduce(self):
        """Tournament + elite selection; replace num_eliminate worst with mutated
        copies of the survivors."""
        N = len(self.agents)
        elite_count = self.elite_count
        num_eliminate = self.num_eliminate
        ts = self.tournament_size
        # Sort by fitness descending
        idx_sorted = sorted(range(N), key=lambda i: self.agents[i].fitness, reverse=True)
        # Elites: top `elite_count` survive
        survivors = [self.agents[i] for i in idx_sorted[:elite_count]]
        # Tournament selection for the rest of the survivors
        rest_pool = [self.agents[i] for i in idx_sorted]
        # We need N - num_eliminate survivors total
        n_needed = N - num_eliminate
        while len(survivors) < n_needed:
            # Pick tournament_size random distinct from rest_pool
            cand = self.rng.sample(rest_pool, min(ts, len(rest_pool)))
            winner = max(cand, key=lambda a: a.fitness)
            survivors.append(winner)
        # Population turnover: keep first n_needed of survivors,
        # replace the rest with mutated copies
        new_agents = list(survivors[:n_needed])
        for i in range(N - n_needed):
            parent = self.rng.choice(survivors)
            new_code = self._mutate(parent.code, parent.fitness)
            new_agent = self._make_agent(new_code, n_needed + i)
            new_agents.append(new_agent)
        # Reset reputation of removed agents (handled by each agent's
        # handle_agents_replaced). The new agents start with empty rep store.
        old_ids = [a.agent_id for a in self.agents]
        new_ids = [a.agent_id for a in new_agents]
        # Update each survivor's reputation store to drop removed IDs
        all_ids_to_remove = set(old_ids) - set(new_ids)
        for a in new_agents:
            for rid in all_ids_to_remove:
                a.reputations.pop(rid, None)
        # Reset the population
        self.agents = new_agents
        # Reassign agent IDs in case they changed
        for i, a in enumerate(self.agents):
            a.agent_id = i
