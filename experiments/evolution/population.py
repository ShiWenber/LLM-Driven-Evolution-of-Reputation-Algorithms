"""Multi-generation evolutionary population manager.

Orchestrates the evolutionary loop:
1. Initialize population (LLM generates diverse strategies)
2. For each generation:
   a. Run T rounds of Donor Game
   b. Compute fitness
   c. Select survivors / eliminate bottom
   d. Mutate surviving code to create children
   e. Replace population
3. Return evolutionary trajectory
"""

import random
import time
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..agents.code_agent import CodeAgent
from ..agents.prompts import build_init_prompt
from ..sandbox.validator import clean_code, validate_strategy_code, CodeValidationError
from ..sandbox.executor import StrategyExecutor
from ..game.donor_game import DonorGame
from .selection import select_survivors, compute_fitness_stats
from .mutation import MutationOperator, RandomMutationOperator, create_mutation_operator


class EvolutionaryPopulation:
    """Manages a population of code-based agents across generations."""

    def __init__(
        self,
        population_size: int = 20,
        num_rounds_per_gen: int = 30,
        benefit: int = 2,
        cost: int = 1,
        observability: str = "full",      # "private", "partial_X", "full"
        observability_p: float = 0.3,      # fraction for partial condition
        elite_count: int = 2,
        num_eliminate: int = 5,
        tournament_size: int = 3,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        api_key: str = "",
        api_base_url: str = "",
        mutation_temperature: float = 0.8,
        seed: int = 42,
        results_dir: str = "results"
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

        # State
        self.agents: List[CodeAgent] = []
        self.generation = 0
        self.history: List[Dict[str, Any]] = []
        self.mutation_op: Optional[MutationOperator] = None
        self._llm_client = None

        random.seed(seed)
        np.random.seed(seed)

    def _get_llm_client(self):
        """Lazy LLM client initialization."""
        if self._llm_client is not None:
            return self._llm_client

        if self.llm_provider == "openai":
            from openai import OpenAI
            kwargs = {}
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                kwargs["api_key"] = api_key
            if self.api_base_url:
                kwargs["base_url"] = self.api_base_url
            self._llm_client = OpenAI(**kwargs)
        elif self.llm_provider == "anthropic":
            import anthropic
            self._llm_client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:
            raise ValueError(f"Unknown provider: {self.llm_provider}")
        return self._llm_client

    def initialize_population(self) -> List[CodeAgent]:
        """Generate initial population using LLM."""
        print(f"\n{'='*50}")
        print(f"Initializing population (N={self.population_size})")
        print(f"LLM: {self.llm_provider}/{self.llm_model}")
        print(f"{'='*50}")

        prompt = build_init_prompt(
            num_strategies=self.population_size,
            population_size=self.population_size,
            num_rounds=self.num_rounds_per_gen
        )

        codes = self._generate_batch(prompt, self.population_size)
        print(f"  Generated {len(codes)} strategies")

        agents = []
        for i, code in enumerate(codes):
            cleaned = clean_code(code)
            try:
                validate_strategy_code(cleaned)
            except CodeValidationError as e:
                print(f"  Agent {i}: validation failed — {e}")
                # Use fallback: simple random strategy
                cleaned = _fallback_strategy()

            agent = CodeAgent(agent_id=i, code=cleaned)
            agent.generation = 0
            agents.append(agent)

        # Fill any missing slots with fallback strategies
        while len(agents) < self.population_size:
            i = len(agents)
            agents.append(CodeAgent(
                agent_id=i,
                code=_fallback_strategy()
            ))
            agents[-1].generation = 0

        self.agents = agents
        self.generation = 0
        return agents

    def _generate_batch(self, prompt: str, count: int) -> List[str]:
        """Generate multiple strategy pairs from LLM with retry on failure."""
        codes = []

        # Try to generate in fewer, larger batches
        batch_size = min(count, 5)
        remaining = count
        start_idx = 0
        max_retries = 3

        while remaining > 0:
            n = min(batch_size, remaining)
            batch_prompt = prompt
            batch_prompt = batch_prompt.replace("{num_strategies}", str(n))
            batch_prompt = batch_prompt.replace("{start_idx}", str(start_idx))

            success = False
            for attempt in range(max_retries):
                try:
                    response_text = self._call_llm_raw(
                        system_msg=(
                            "You are a Python programmer. "
                            "Generate strategy code pairs. "
                            "Separate multiple pairs with '# ---' on its own line. "
                            "Each pair MUST contain BOTH 'evaluate' AND 'decide' functions. "
                            "IMPORTANT: evaluate() takes (current_reputation, observation, "
                            "my_history, round_num) and returns float between -1.0 and 1.0. "
                            "decide() takes (recipient_reputation, round_num, my_history) "
                            "and returns bool. "
                            "Use observation['action'] (a string equal to 'cooperate' or 'defect') "
                            "to discriminate the two action options. "
                            "In my_history: entry['action'] is your own past action, "
                            "entry['partner_action'] is the other agent's action."
                        ),
                        user_msg=batch_prompt,
                        temperature=0.9,
                        max_tokens=8000
                    )

                    if response_text is None:
                        delay = 2 ** attempt
                        print(f"  [init] LLM returned None (attempt {attempt + 1}), "
                              f"retrying in {delay}s...")
                        time.sleep(delay)
                        continue

                    # Split on separator or extract individual function pairs
                    batch_codes = self._parse_batch_response(response_text)
                    if batch_codes:
                        codes.extend(batch_codes[:n])
                        start_idx += len(batch_codes[:n])
                        remaining -= len(batch_codes[:n])
                        success = True

                    if len(batch_codes) < n:
                        # Didn't get enough, fill with fallback
                        for _ in range(n - len(batch_codes)):
                            fallback = _fallback_strategy()
                            codes.append(fallback)
                            start_idx += 1
                            remaining -= 1
                            success = True

                    if success:
                        break

                except Exception as e:
                    delay = 2 ** attempt
                    print(f"  [init] LLM batch failed (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        print(f"  [init] Retrying in {delay}s...")
                        time.sleep(delay)

            if not success:
                print(f"  [init] All attempts failed, using fallback strategies")
                for _ in range(n):
                    codes.append(_fallback_strategy())
                    remaining -= 1

            time.sleep(0.5)  # Rate limiting

        return codes[:count]

    def _parse_batch_response(self, response: str) -> List[str]:
        """Parse batch LLM response into individual strategy pairs."""
        # Try splitting on separator
        parts = response.split("# ---")
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]

        # Try extracting pairs that have both evaluate and decide
        import re
        # Find blocks that contain both evaluate and decide functions
        # Split on double newline or use the fact that pairs have two def statements
        pairs = re.findall(
            r'(?:def evaluate\b.*?)(?:def decide\b.*)',
            response,
            re.DOTALL
        )
        if pairs:
            return [p.strip() for p in pairs]

        # Fallback: return whole response if it has both functions
        if "def evaluate" in response and "def decide" in response:
            return [response.strip()]

        return []

    def _call_llm_raw(
        self,
        system_msg: str,
        user_msg: str,
        temperature: float = 0.9,
        max_tokens: int = 8000
    ) -> Optional[str]:
        """Make a raw LLM API call. Returns None on malformed response."""
        client = self._get_llm_client()

        if self.llm_provider == "openai":
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            if response is None or not hasattr(response, 'choices') or not response.choices:
                return None
            content = response.choices[0].message.content
            return content if content else None

        elif self.llm_provider == "anthropic":
            response = client.messages.create(
                model=self.llm_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msg,
                messages=[{"role": "user", "content": user_msg}]
            )
            if response is None or not hasattr(response, 'content') or not response.content:
                return None
            return response.content[0].text

        raise ValueError(f"Unknown provider: {self.llm_provider}")

    def run_generation(self, gen_idx: int) -> Dict[str, Any]:
        """Run one generation: execute strategies, compute fitness."""
        # Reassign agent IDs to match list positions
        for i, agent in enumerate(self.agents):
            agent.agent_id = i
            agent.reset_for_generation()

        # Run donor game
        game = DonorGame(
            population_size=self.population_size,
            benefit=self.benefit,
            cost=self.cost,
            num_rounds=self.num_rounds_per_gen,
            observability=self.observability,
            observability_p=self.observability_p
        )

        game.setup_population(self.agents)
        results = game.run_simulation()

        # Update agent fitness from game results
        for agent_state in results["agent_states"]:
            agent_id = agent_state["agent_id"]
            self.agents[agent_id].fitness = agent_state["payoff"]
            self.agents[agent_id].total_donations = agent_state["donations_given"]
            self.agents[agent_id].total_decisions = agent_state["total_rounds"]

        # Compute generation stats
        stats = compute_fitness_stats(self.agents)
        stats["generation"] = gen_idx

        self.history.append(stats)
        return stats

    def evolve_generation(self) -> Dict[str, Any]:
        """Execute one full evolutionary step: run → select → mutate."""
        gen_idx = self.generation

        # 1. Run generation
        stats = self.run_generation(gen_idx)
        print(
            f"  Gen {gen_idx}: "
            f"coop={stats['cooperation_rate_mean']:.3f}, "
            f"fitness={stats['fitness_mean']:.1f}, "
            f"best={stats['fitness_max']:.1f}"
        )

        # 2. Selection
        survivors, eliminated, parents = select_survivors(
            self.agents,
            elite_count=self.elite_count,
            num_to_eliminate=self.num_eliminate,
            tournament_size=self.tournament_size
        )

        # 3. Create mutation operator if needed
        if self.mutation_op is None:
            # Allow override via env var LLM_MUTATION_WORKERS (default 5,
            # capped at 30 by caller constraint)
            workers = int(os.environ.get("LLM_MUTATION_WORKERS", "5"))
            workers = min(workers, 30)  # hard cap per user spec
            self.mutation_op = MutationOperator(
                llm_provider=self.llm_provider,
                model=self.llm_model,
                temperature=self.mutation_temperature,
                api_key=self.api_key,
                api_base_url=self.api_base_url,
                max_workers=workers
            )

        # 4. Mutate to create children (concurrent LLM calls)
        children = []
        parent_inputs = [(p.code, p.fitness) for p in parents]
        mutated_codes = self.mutation_op.mutate_batch(
            parent_inputs,
            self.population_size,
            max_workers=getattr(self.mutation_op, 'max_workers', 5)
        )

        for parent, mutated_code in zip(parents, mutated_codes):
            if mutated_code is None:
                # Mutation failed, use parent code with slight variation
                mutated_code = parent.code

            child = CodeAgent(
                agent_id=-1,  # Will be reassigned
                code=mutated_code
            )
            child.generation = gen_idx + 1
            child.parent_id = parent.strategy_id
            children.append(child)

        # 5. Build new population
        new_population = survivors + children

        # 6. Notify surviving agents about eliminated/replaced agents
        eliminated_ids = [a.agent_id for a in eliminated]
        # After reassigning IDs, old IDs of eliminated agents no longer exist,
        # and new agents will have fresh IDs. Survivors clear reputation for eliminated IDs.
        for agent in survivors:
            agent.handle_agents_replaced(
                old_ids=eliminated_ids,
                new_ids=[]  # New IDs assigned below
            )

        # Assign new IDs
        for i, agent in enumerate(new_population):
            agent.agent_id = i

        self.agents = new_population
        self.generation += 1

        stats["num_survivors"] = len(survivors)
        stats["num_eliminated"] = len(eliminated)
        stats["num_children"] = len(children)
        return stats

    def run_evolution(self, num_generations: int = 10) -> Dict[str, Any]:
        """Run the full evolutionary process."""
        print(f"\n{'='*60}")
        print(f"Evolutionary Run: {num_generations} generations")
        print(f"Population: {self.population_size}, "
              f"Observability: {self.observability}, "
              f"Seed: {self.seed}")
        print(f"{'='*60}")

        # Initialize
        self.initialize_population()

        # Run generations
        for gen in range(num_generations):
            print(f"\n--- Generation {gen + 1}/{num_generations} ---")
            stats = self.evolve_generation()

        # Collect final results
        results = self._collect_results()
        self._save_results(results)

        return results

    def _collect_results(self) -> Dict[str, Any]:
        """Collect evolutionary trajectory results."""
        final_agents = []
        for agent in self.agents:
            final_agents.append({
                "agent_id": agent.agent_id,
                "strategy_id": agent.strategy_id,
                "fitness": agent.fitness,
                "cooperation_rate": agent.cooperation_rate,
                "generation": agent.generation,
                "parent_id": agent.parent_id,
                "code": agent.code,
            })

        return {
            "config": {
                "population_size": self.population_size,
                "num_rounds_per_gen": self.num_rounds_per_gen,
                "benefit": self.benefit,
                "cost": self.cost,
                "observability": self.observability,
                "observability_p": self.observability_p,
                "elite_count": self.elite_count,
                "num_eliminate": self.num_eliminate,
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
                "seed": self.seed,
            },
            "trajectory": self.history,
            "final_population": final_agents,
            "timestamp": datetime.now().isoformat(),
        }

    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"evo_{self.observability}_{self.llm_model}_{timestamp}.json"
        )
        filepath = self.results_dir / filename

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {filepath}")


def _fallback_strategy() -> str:
    """Generate a fallback strategy pair when LLM fails."""
    variants = [
        # Always cooperate
        '''def evaluate(current_reputation, observation, my_history, round_num):
    return current_reputation

def decide(recipient_reputation, round_num, my_history):
    return True
''',
        # Never cooperate
        '''def evaluate(current_reputation, observation, my_history, round_num):
    return current_reputation

def decide(recipient_reputation, round_num, my_history):
    return False
''',
        # Random 50/50
        '''import random

def evaluate(current_reputation, observation, my_history, round_num):
    return current_reputation

def decide(recipient_reputation, round_num, my_history):
    return random.random() < 0.5
''',
        # Simple image-scoring-like: +1 for cooperate, -1 for defect
        '''def evaluate(current_reputation, observation, my_history, round_num):
    if observation["action"] == "cooperate":
        return current_reputation + 1.0
    else:
        return current_reputation - 1.0

def decide(recipient_reputation, round_num, my_history):
    return recipient_reputation >= 0.0
''',
    ]
    return random.choice(variants)
