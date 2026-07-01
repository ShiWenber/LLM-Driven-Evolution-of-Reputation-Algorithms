"""LLM-driven mutation operator for strategy code evolution.

Uses LLM APIs to rewrite successful strategy code, creating variants
that preserve core insights while exploring the strategy space.
"""

import os
import random
import time
from typing import Optional, List

from ..sandbox.validator import clean_code, validate_strategy_code, CodeValidationError
from ..agents.prompts import build_mutation_prompt
from ..config.load_env import get_api_key as _env_api_key, get_base_url as _env_base_url


class MutationOperator:
    """Mutates strategy code using LLM-based rewriting."""

    def __init__(
        self,
        llm_provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.8,
        max_retries: int = 3,
        rate_limit_delay: float = 0.5,
        max_tokens: int = 3000,  # reduced from 8000: Paratera/Intern-S2-Preview tends to emit 30K+ char garbage otherwise
        max_workers: int = 5,  # max concurrent LLM calls in mutate_batch
        api_key: str = "",
        api_base_url: str = "",
        use_exploration: bool = False,
        exploration_prob: float = 0.5,
        per_call_timeout: float = 60.0,  # max seconds per single LLM call (Intern thinking model can hang)
    ):
        """
        Initialize mutation operator.

        Args:
            llm_provider: "openai" or "anthropic"
            model: Model name
            temperature: LLM temperature for mutation creativity
            max_retries: Max attempts if mutation produces invalid code
            rate_limit_delay: Seconds between API calls
            use_exploration: If True, sample the exploration-mode mutation prompt
                (does NOT name specific algorithms) with probability
                exploration_prob on each mutation call. Used by
                --exploration-mutation flag (algorithmic-complexity probes).
            exploration_prob: Probability of using the exploration prompt on
                each call (default 0.5).
        """
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.use_exploration = use_exploration
        self.exploration_prob = exploration_prob
        self.per_call_timeout = per_call_timeout
        self._client = None

    def _get_client(self):
        """Lazy initialization of LLM client."""
        if self._client is not None:
            return self._client

        if self.llm_provider == "openai":
            from openai import OpenAI
            kwargs = {}
            # Try explicit arg -> env var -> load_env (covers .env + shell)
            api_key = self.api_key or _env_api_key("openai") or _env_api_key("deepseek")
            if api_key:
                kwargs["api_key"] = api_key
            base_url = self.api_base_url or _env_base_url("openai") or _env_base_url("deepseek")
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
        elif self.llm_provider == "anthropic":
            import anthropic
            api_key = self.api_key or _env_api_key("anthropic")
            self._client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {self.llm_provider}")

        return self._client

    def mutate(
        self,
        parent_code: str,
        parent_fitness: float,
        population_size: int = 20,
        recent_window: int = 0,
    ) -> Optional[str]:
        """
        Mutate a strategy code string.

        Args:
            parent_code: The successful parent strategy code
            parent_fitness: Fitness score of the parent
            population_size: Population size (for prompt context)
            recent_window: If >0, mention the recent_window observation field
                in the mutation prompt (LLM may then use it).

        Returns:
            Mutated code string, or None if all retries failed
        """
        if self.use_exploration and random.random() < self.exploration_prob:
            from experiments.agents.prompts import build_exploration_mutation_prompt
            prompt = build_exploration_mutation_prompt(
                parent_code, parent_fitness, population_size, recent_window=recent_window
            )
        else:
            prompt = build_mutation_prompt(
                parent_code, parent_fitness, population_size, recent_window=recent_window
            )

        for attempt in range(self.max_retries):
            try:
                response = self._call_llm(prompt)
                if response is None:
                    delay = 2 ** attempt
                    print(f"  [mutation] LLM returned None (attempt {attempt + 1}), "
                          f"retrying in {delay}s...")
                    time.sleep(delay)
                    continue

                cleaned = clean_code(response)

                # Validate the mutated code
                try:
                    validate_strategy_code(cleaned)
                except CodeValidationError as e:
                    print(f"  [mutation] Validation failed (attempt {attempt + 1}): {e}")
                    # Add error feedback to prompt for next attempt
                    prompt = (
                        f"Your previous output had an error: {e}\n\n"
                        f"Please fix the issue and return ONLY a valid Python function.\n\n"
                        f"{prompt}"
                    )
                    continue

                time.sleep(self.rate_limit_delay)
                return cleaned

            except Exception as e:
                delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"  [mutation] API call failed (attempt {attempt + 1}): {e}")
                print(f"  [mutation] Retrying in {delay}s...")
                time.sleep(delay)

        # All LLM retries failed — fall back to random mutation
        print(f"  [mutation] All LLM attempts failed, using random mutation fallback")
        random_op = RandomMutationOperator()
        return random_op.mutate(parent_code, parent_fitness)

    def _call_llm(self, prompt: str, max_tokens: int = None) -> Optional[str]:
        """Call the LLM API. Returns None on malformed response."""
        client = self._get_client()
        mt = max_tokens if max_tokens is not None else self.max_tokens

        if self.llm_provider == "openai":
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Python programmer. "
                            "Respond with valid Python code. "
                            "The code MUST contain BOTH 'evaluate' AND 'decide' functions. "
                            "evaluate(current_reputation, observation, my_history, round_num) -> float "
                            "decide(recipient_reputation, round_num, my_history) -> bool. "
                            "In evaluate: use observation['action'] (a string equal to 'cooperate' or "
                            "'defect') to discriminate the two action options. evaluate must return "
                            "a float between -1.0 and 1.0. Do NOT condition decide() on partner_action "
                            "from my_history — that would be direct reciprocity, not indirect."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=mt,
                timeout=self.per_call_timeout,
            )
            if response is None or not hasattr(response, 'choices') or not response.choices:
                return None
            content = response.choices[0].message.content
            return content if content else None

        elif self.llm_provider == "anthropic":
            response = client.messages.create(
                model=self.model,
                max_tokens=mt,
                temperature=self.temperature,
                system=(
                    "You are a Python programmer. "
                    "Respond with valid Python code only. "
                    "Must contain BOTH evaluate() and decide() functions. "
                    "evaluate(current_reputation, observation, my_history, round_num) -> float. "
                    "decide(recipient_reputation, round_num, my_history) -> bool. "
                    "In evaluate: use observation['action'] (a string equal to 'cooperate' or "
                    "'defect') to discriminate the two action options. evaluate must return "
                    "a float between -1.0 and 1.0. Do NOT condition decide() on partner_action "
                    "from my_history — that would be direct reciprocity, not indirect."
                ),
                messages=[{"role": "user", "content": prompt}]
            )
            if response is None or not hasattr(response, 'content') or not response.content:
                return None
            return response.content[0].text

        raise ValueError(f"Unknown provider: {self.llm_provider}")

    def mutate_batch(
        self,
        parents: List,  # list of (code, fitness) tuples
        population_size: int = 20,
        max_workers: int = 5,
        recent_window: int = 0,
    ) -> List[Optional[str]]:
        """
        Mutate multiple parents concurrently using a thread pool.

        Each parent gets its own LLM call (NOT a single combined-prompt batch),
        but the calls are issued in parallel using ThreadPoolExecutor. This
        preserves the per-parent prompt (no quality loss from cramming multiple
        parents into one prompt) while exploiting the API's ability to handle
        many concurrent requests.

        Args:
            parents: List of (code, fitness) tuples
            population_size: Population size (for prompt context)
            max_workers: Max number of concurrent LLM calls (default 5,
                capped at 30 by caller)
            recent_window: If >0, mention the recent_window observation field
                in each mutation prompt (LLM may then use it).

        Returns:
            List of mutated code strings, one per parent (in same order).
            Failures fall back to per-parent sequential mutate() (which itself
            falls back to random mutation).
        """
        if not parents:
            return []
        if len(parents) == 1:
            return [self.mutate(parents[0][0], parents[0][1], population_size, recent_window=recent_window)]

        # Cap workers to a sane maximum (caller may pass up to 30, but we
        # never want more workers than parents)
        n_workers = min(max_workers if max_workers is not None else self.max_workers, len(parents))

        from concurrent.futures import ThreadPoolExecutor, as_completed
        results: List[Optional[str]] = [None] * len(parents)

        def _one(i, code, fitness):
            return i, self.mutate(code, fitness, population_size, recent_window=recent_window)

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(_one, i, code, fitness)
                for i, (code, fitness) in enumerate(parents)
            ]
            for fut in as_completed(futures):
                i, mutated = fut.result()
                results[i] = mutated

        return results


class RandomMutationOperator:
    """Non-LLM mutation operator for control experiments.

    Performs simple syntactic perturbations: flip comparisons,
    change constants, swap branches. Used to test whether LLM-driven
    mutation is actually better than random variation.
    """

    def mutate(self, parent_code: str, parent_fitness: float = 0, population_size: int = 20) -> Optional[str]:
        """Apply random syntactic mutations to strategy code."""
        code = parent_code

        mutations = [
            self._flip_comparison,
            self._change_constant,
            self._swap_return,
        ]

        # Apply 1-3 random mutations
        for _ in range(random.randint(1, 3)):
            mutation_fn = random.choice(mutations)
            code = mutation_fn(code)

        try:
            validate_strategy_code(code)
        except CodeValidationError:
            return None

        return code

    def _flip_comparison(self, code: str) -> str:
        """Flip a comparison operator."""
        flips = [
            (">", "<="),
            ("<", ">="),
            (">=", "<"),
            ("<=", ">"),
            ("==", "!="),
            ("!=", "=="),
        ]
        old, new = random.choice(flips)
        if old in code:
            # Replace first occurrence
            code = code.replace(old, new, 1)
        return code

    def _change_constant(self, code: str) -> str:
        """Change a numeric constant by random factor."""
        import re
        # Find a number and scale it
        numbers = re.findall(r'\b(\d+\.?\d*)\b', code)
        if numbers:
            target = random.choice(numbers)
            factor = random.uniform(0.5, 2.0)
            new_val = float(target) * factor
            code = code.replace(target, f"{new_val:.1f}", 1)
        return code

    def _swap_return(self, code: str) -> str:
        """Swap True/False in a return statement."""
        if "return True" in code:
            code = code.replace("return True", "return False", 1)
        elif "return False" in code:
            code = code.replace("return False", "return True", 1)
        return code


def create_mutation_operator(
    llm_provider: str = "openai",
    model: str = "gpt-4o",
    use_random: bool = False
):
    """Factory function to create appropriate mutation operator."""
    if use_random:
        return RandomMutationOperator()
    return MutationOperator(
        llm_provider=llm_provider,
        model=model
    )
