"""Population + LLM-driven evolution for the v2 quantitative interface.

Mirrors the v1 EvolutionaryPopulation but uses the v2 QuantitativeAgent
and V2DonorGame.

Supports two agent types:
  - `agent_type="v2"`: type-1 agents. The LLM emits two top-level
    Python functions (`evaluate` + `decide`); the framework maintains a
    scalar reputation matrix for them.
  - `agent_type="v3"`: type-2 agents. The LLM emits a full Python class
    named `LLMAgent` with `__init__(agent_id)`, `decide()`, and
    `observe(...)` methods. The LLM owns its own internal state
    structure (dicts, lists, counters — anything). The framework still
    maintains a scalar `reputations` matrix for bookkeeping, but the
    LLM is not required to read it.

Type 2 baseline mode currently only supports ALLCClass and ALLDClass
(class wrappers around the trivial always-cooperate / always-defect
strategies). The 8 leading-eight rules live in type-1 land.
"""
from __future__ import annotations
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

from .agent import QuantitativeAgent
from .executor import V2StrategyExecutor
from .agent_full import (
    FullAgent, V3StrategyExecutor,
    ALLCClass, ALLDClass,
    ALLC_CLASS_SOURCE, ALLD_CLASS_SOURCE,
)
from .game import V2DonorGame
from .prompts import (
    INIT_PROMPT_V2, MUTATION_PROMPT_V2,
    INIT_PROMPT_V3, MUTATION_PROMPT_V3,
)
from .baselines import get_baseline


# Fallback strategies when LLM fails (type 1: two top-level functions)
FALLBACK_STRATEGIES = [
    # Always cooperate
    '''
def evaluate(target_reputation, target_action, my_reputation):
    if target_action == 'cooperate':
        new = target_reputation + 0.333
    else:
        new = target_reputation - 0.333
    return max(-1.0, min(1.0, new))
def decide(my_reputation, opponent_reputation):
    return True
''',
    # Always defect
    '''
def evaluate(target_reputation, target_action, my_reputation):
    if target_action == 'cooperate':
        new = target_reputation + 0.333
    else:
        new = target_reputation - 0.333
    return max(-1.0, min(1.0, new))
def decide(my_reputation, opponent_reputation):
    return False
''',
]


# Type-2 fallback: a complete LLMAgent class. We intentionally make
# it behaviorally NEUTRAL (50% cooperate / 50% defect, deterministic
# per agent_id) rather than the previous always-cooperate default.
# Why this matters: when the LLM init silently fails (content='',
# reasoning_content consumed all max_tokens, etc.), every init
# attempt returns the FALLBACK. With the old `return True`, the
# whole population starts as 15/15 perfect cooperators -> 1.000
# cooperation in gen 0 -> selection has nothing to amplify, and
# downstream metrics look deceptively good. With neutral FALLBACK,
# a heavy-FALLBACK run sits at ~0.5 cooperation, which is a clear
# signal that the LLM init is broken and the run is unreliable.
FALLBACK_CLASS_V3 = '''
import random as _rnd
class LLMAgent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self._ctx_opponent_id = None
        # Deterministic per-agent RNG, seeded by agent_id. Each
        # instance gets its own stream so a population of 15
        # FALLBACKs averages to ~0.5, not 1.0 or 0.0.
        self._rng = _rnd.Random(agent_id * 7919 + 42)

    def decide(self) -> bool:
        return self._rng.random() < 0.5

    def observe(self, donor_id, donor_action, recipient_id, recipient_action) -> None:
        return None
'''



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
        # target_interactions_per_gen: if set (and > 0), overrides
        # num_rounds_per_gen at construction time so the caller can
        # think in terms of total PD games per gen rather than
        # rounds. Computed as ceil(target / (population_size // 2));
        # with N=15 we get 7 pairs per round, so target=1000 ->
        # 143 rounds = 1001 games. The LLM call count is governed
        # separately by num_eliminate (5/gen) and does NOT scale
        # with rounds, so going from 30 -> 143 rounds is ~4.77x
        # more game time but the same ~5 LLM calls per gen.
        target_interactions_per_gen: Optional[int] = None,
        # fitness_window_interactions: if set, only the LAST
        # `fitness_window_interactions` joint actions of each gen
        # contribute to an agent's fitness for selection; the
        # earlier `total - window` are treated as burn-in (still
        # played so observe() / reputations evolve, but their
        # payoffs don't count). Default 200: with
        # target_interactions_per_gen=1000, the first 800 are
        # burn-in. Pass None (or 0) to use all interactions
        # (legacy behavior).
        fitness_window_interactions: Optional[int] = 200,
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
        agent_type: str = "v2",
        # If use_baseline is set, all agents use that baseline strategy and
        # the LLM is not used. If None, LLM evolution runs.
        # agent_type:
        #   "v2" (default) — type-1 agents. LLM emits two top-level
        #       functions (`evaluate` + `decide`); framework maintains
        #       a scalar `reputations` dict.
        #   "v3" — type-2 agents. LLM emits a full `LLMAgent` class
        #       with `__init__(agent_id)`, `decide()`, and
        #       `observe(...)` methods. LLM owns its own state
        #       structure; framework only handles bookkeeping.
        # llm_thinking: when True, sends `thinking={"type": "enabled"}` to
        #   the API and bumps max_tokens to llm_max_tokens_thinking to fit
        #   both reasoning_content and the final code. When False (default),
        #   sends `thinking={"type": "disabled"}` and uses
        #   llm_max_tokens_base. DeepSeek-v4-flash is a reasoning model,
        #   so the default off-state keeps wall time low (~11s/call) and
        #   avoids empty-content truncation. Set True for research
        #   questions that need to inspect the chain-of-thought.
        llm_thinking: bool = False,
        llm_max_tokens_base: int = 4000,
        llm_max_tokens_thinking: int = 12000,
    ):
        if agent_type not in ("v2", "v3"):
            raise ValueError(f"agent_type must be 'v2' or 'v3', got {agent_type!r}")
        self.population_size = population_size
        # If caller asked for a target interaction count, derive the
        # round count from it. With N=15 we get 7 pairs/round; with
        # N=20 we get 10 pairs/round. The result is rounded UP so we
        # hit the target (slightly over is fine; missing it by
        # hundreds would be a measurement bug).
        if target_interactions_per_gen is not None and target_interactions_per_gen > 0:
            pairs_per_round = max(1, population_size // 2)
            num_rounds_per_gen = (
                (target_interactions_per_gen + pairs_per_round - 1) // pairs_per_round
            )
        self.num_rounds_per_gen = num_rounds_per_gen
        self.target_interactions_per_gen = target_interactions_per_gen
        self.fitness_window_interactions = fitness_window_interactions
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
        self.agent_type = agent_type
        self.llm_thinking = llm_thinking
        self.llm_max_tokens_base = llm_max_tokens_base
        self.llm_max_tokens_thinking = llm_max_tokens_thinking
        # Derived: the actual max_tokens we'll send. Cached so
        # _call_llm doesn't recompute on every call.
        self._llm_max_tokens = (
            self.llm_max_tokens_thinking if self.llm_thinking
            else self.llm_max_tokens_base
        )
        # Derived: the extra_body payload. For DeepSeek-v4-flash we
        # always send an explicit `thinking` value because its
        # default is ON, which silently truncates our code at
        # max_tokens. With thinking=disabled (the default) the
        # full budget goes to the final code.
        if self.llm_thinking:
            self._llm_extra_body = {"thinking": {"type": "enabled"}}
        else:
            self._llm_extra_body = {"thinking": {"type": "disabled"}}
        self.agents: List[object] = []  # QuantitativeAgent or FullAgent
        self.rng = random.Random(seed)
        # Monotonic counter: each new agent gets a fresh, never-reused id.
        # This keeps agent_id stable across generations; reputations keyed
        # by agent_id remain valid for the lifetime of the agent.
        self._next_agent_id: int = 0
        # LLM client (lazy)
        self._llm_client = None
        # FALLBACK diagnostics. _fallback_init_count: how many of the
        # population_size init attempts ended up using the
        # deterministic-random FALLBACK (i.e., LLM init silently
        # failed 3x in a row). _fallback_mutation_count: how many
        # _select_and_reproduce cycles hit the Fix-B fallback path
        # (mutate produced a code that smoke-validated but failed at
        # real-id instantiation). Both are reported at run end so we
        # can flag runs where the LLM is misbehaving heavily.
        self._fallback_init_count: int = 0
        self._fallback_mutation_count: int = 0
        # Echo the effective interaction count so callers can verify
        # the override took effect (and so log analysis can grep for
        # it).
        if target_interactions_per_gen is not None and target_interactions_per_gen > 0:
            pairs_per_round = max(1, population_size // 2)
            actual = self.num_rounds_per_gen * pairs_per_round
            print(
                f"  [V2EvolutionaryPopulation] target_interactions_per_gen="
                f"{target_interactions_per_gen} -> num_rounds_per_gen="
                f"{self.num_rounds_per_gen} -> {actual} games/gen "
                f"(N={population_size}, pairs={pairs_per_round})"
            )

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
                    max_tokens=self._llm_max_tokens,
                    extra_body=self._llm_extra_body,
                )
                content = resp.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                print(f"  [LLM error attempt {attempt+1}]: {e}")
                time.sleep(2 ** attempt)
        return None

    def _make_agent(self, code: str, agent_id: int):
        """Validate `code` and instantiate one agent of the configured type."""
        if self.agent_type == "v3":
            executor = V3StrategyExecutor(code)
            return FullAgent(agent_id, executor=executor, code=code)
        # v2 (default)
        executor = V2StrategyExecutor(code)
        return QuantitativeAgent(agent_id, code, executor=executor)

    def _new_agent(self, code: str):
        """Allocate a fresh, never-reused agent_id and create the agent."""
        aid = self._next_agent_id
        self._next_agent_id += 1
        return self._make_agent(code, aid)

    def _validate_code(self, code: str) -> None:
        """Validate that `code` is acceptable for the current agent_type.

        For v2: instantiates the V2StrategyExecutor (which loads evaluate
        and decide). For v3: instantiates the V3StrategyExecutor (which
        loads the LLMAgent class). Raises on any error.
        """
        if self.agent_type == "v3":
            V3StrategyExecutor(code)
        else:
            V2StrategyExecutor(code)

    def _init_population_llm(self):
        """Generate N strategies via LLM.

        FALLBACK is the deterministic-random class (Fix C) — not
        the old always-cooperate one. Every FALLBACK hit increments
        self._fallback_init_count so we can audit run reliability at
        the end.
        """
        print(f"  Initializing population via LLM ({self.population_size} agents, agent_type={self.agent_type}, thinking={self.llm_thinking}, max_tokens={self._llm_max_tokens})...")
        client = self._get_llm_client()
        user_msg = INIT_PROMPT_V3 if self.agent_type == "v3" else INIT_PROMPT_V2
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
                        max_tokens=self._llm_max_tokens,
                        extra_body=self._llm_extra_body,
                    )
                    content = resp.choices[0].message.content
                    code = _extract_code_from_response(content)
                    if code:
                        # Validate
                        try:
                            agent = self._new_agent(code)
                            self.agents.append(agent)
                            break
                        except Exception as e:
                            print(f"  [init agent {i} validation fail]: {e}")
                except Exception as e:
                    print(f"  [init agent {i} LLM error attempt {attempt+1}]: {e}")
                    time.sleep(2)
            else:
                # All attempts failed; use the deterministic-random
                # FALLBACK (Fix C: ~50% cooperation, NOT 1.000). This
                # is also a signal that the LLM init is unreliable,
                # so we count it.
                if self.agent_type == "v3":
                    fb = FALLBACK_CLASS_V3
                else:
                    fb = self.rng.choice(FALLBACK_STRATEGIES)
                agent = self._new_agent(fb)
                self.agents.append(agent)
                self._fallback_init_count += 1
                print(f"  [init agent {i}] using FALLBACK strategy")

    def _init_population_baseline(self):
        """All agents use the same baseline strategy."""
        assert self.use_baseline is not None
        if self.agent_type == "v3":
            # Only ALLC / ALLD supported as type-2 baselines
            t2_baselines = {"ALLC": ALLC_CLASS_SOURCE, "ALLD": ALLD_CLASS_SOURCE}
            if self.use_baseline not in t2_baselines:
                raise ValueError(
                    f"agent_type='v3' only supports ALLC / ALLD as baselines; "
                    f"got {self.use_baseline!r}. The 8 leading-eight are type-1 only."
                )
            code = t2_baselines[self.use_baseline]
        else:
            code = get_baseline(self.use_baseline)
        for i in range(self.population_size):
            try:
                self.agents.append(self._new_agent(code))
            except Exception as e:
                print(f"  [init baseline {i}] validation fail: {e}")
                if self.agent_type == "v3":
                    fb = FALLBACK_CLASS_V3
                else:
                    fb = self.rng.choice(FALLBACK_STRATEGIES)
                self.agents.append(self._new_agent(fb))
        print(f"  Initialized {len(self.agents)} agents with baseline '{self.use_baseline}' (agent_type={self.agent_type})")

    def _mutate(self, parent_code: str, parent_fitness: float) -> str:
        """LLM-driven mutation of parent code."""
        if self.agent_type == "v3":
            user_msg = MUTATION_PROMPT_V3.format(fitness=parent_fitness, parent_code=parent_code)
        else:
            user_msg = MUTATION_PROMPT_V2.format(fitness=parent_fitness, parent_code=parent_code)
        for attempt in range(3):
            content = self._call_llm("You are a Python programmer. Output only valid Python code.", user_msg)
            code = _extract_code_from_response(content) if content else None
            if code:
                # Validate
                try:
                    self._validate_code(code)
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
            fitness_window_interactions=self.fitness_window_interactions,
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
        game._interaction_deltas = []
        for _ in range(T):
            game.play_round()
            game.distribute_observations_and_self_judgments()
        coop_count = sum(1 for inter in game._global_log if inter["donor_action"] == "cooperate")
        coop_rate = coop_count / max(1, len(game._global_log))
        # Windowed fitness: only the last
        # `fitness_window_interactions` interactions count toward
        # selection. Earlier interactions are played (so observe()
        # history and reputations evolve) but their payoffs are
        # treated as burn-in. See game.get_windowed_fitness().
        fitness = game.get_windowed_fitness()
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
        # FALLBACK diagnostics (Fix E). Print init and mutation
        # FALLBACK ratios so reviewers can judge run reliability.
        # Init ratio >30% usually means the LLM init is broken
        # (model name, API key, thinking-mode mismatch); mutation
        # ratio >30% usually means the model is producing invalid
        # code at a high rate.
        init_ratio = self._fallback_init_count / max(1, self.population_size)
        mut_total = (num_generations - 1) * self.num_eliminate if num_generations > 1 else 0
        mut_ratio = self._fallback_mutation_count / max(1, mut_total)
        print(
            f"  [FALLBACK stats] init={self._fallback_init_count}/"
            f"{self.population_size} ({init_ratio:.0%}), "
            f"mutation={self._fallback_mutation_count}/{mut_total} "
            f"({mut_ratio:.0%}), thinking={self.llm_thinking}, "
            f"max_tokens={self._llm_max_tokens}"
        )
        if init_ratio > 0.3:
            print(
                f"  [FALLBACK warning] init FALLBACK ratio > 30% "
                f"— LLM init is likely broken; run "
                f"results are NOT reliable."
            )
        return {
            "trajectory": trajectory,
            "final_population": final_population,
            "config": {
                "schema_version": 3,
                "agent_type": self.agent_type,
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
                "num_rounds_per_gen": self.num_rounds_per_gen,
                "target_interactions_per_gen": self.target_interactions_per_gen,
                "llm_thinking": self.llm_thinking,
                "llm_max_tokens": self._llm_max_tokens,
                "fallback_init_count": self._fallback_init_count,
                "fallback_mutation_count": self._fallback_mutation_count,
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
        # Elites: top `elite_count` survive (unique by definition)
        survivors = [self.agents[i] for i in idx_sorted[:elite_count]]
        rest_pool = [self.agents[i] for i in idx_sorted]
        # We need N - num_eliminate unique survivors total. Tournament can
        # only pick a winner that's not already in the survivor set.
        n_needed = N - num_eliminate
        survivor_set = set(survivors)
        while len(survivor_set) < n_needed:
            cand = self.rng.sample(rest_pool, min(ts, len(rest_pool)))
            winner = max(cand, key=lambda a: a.fitness)
            survivor_set.add(winner)
        # Convert to a stable order: by fitness desc, then by agent_id for ties
        survivors = sorted(
            survivor_set, key=lambda a: (a.fitness, -a.agent_id), reverse=True
        )[:n_needed]
        # Population turnover: keep n_needed survivors, replace the rest with
        # mutated copies that get a fresh, never-reused agent_id.
        new_agents = list(survivors[:n_needed])
        for _ in range(N - n_needed):
            parent = self.rng.choice(survivors)
            new_code = self._mutate(parent.code, parent.fitness)
            try:
                new_agents.append(self._new_agent(new_code))
            except Exception as e:
                # Defense in depth: if a mutated class somehow slips past
                # _validate_code but fails to instantiate for the actual
                # agent_id, fall back to a fresh clone of the parent.
                # Without this, the whole run crashes (e.g., the gen 7
                # crash in M4 smoke test). Count it for the run-end
                # FALLBACK diagnostics.
                self._fallback_mutation_count += 1
                print(
                    f"  [_select_and_reproduce fallback] using parent_code "
                    f"for agent after mutate: {type(e).__name__}: {e}"
                )
                new_agents.append(self._new_agent(parent.code))
        # Drop reputations pointing at removed agents. Each survivor retains
        # its own agent_id (stable), so we can safely pop by id.
        old_ids = {a.agent_id for a in self.agents}
        new_ids = {a.agent_id for a in new_agents}
        ids_to_drop = old_ids - new_ids
        for a in new_agents:
            for rid in ids_to_drop:
                a.reputations.pop(rid, None)
        # NOTE: do NOT reassign agent_id here. Each agent's id is its stable
        # global identity; list position in self.agents is just iteration
        # order and may differ across generations.
        self.agents = new_agents
