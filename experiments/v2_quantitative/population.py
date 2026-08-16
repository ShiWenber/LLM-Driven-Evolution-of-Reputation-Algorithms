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
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..evolution_log import (
    ORIGIN_INITIAL, ORIGIN_IMITATE, ORIGIN_INDEPENDENT_INIT, ORIGIN_MUTATE,
    build_evolution_results, lineage_event, make_config, population_entry,
    trajectory_entry,
    F_BIRTH_GEN, F_ORIGIN, F_PARENT_ID, F_PARENT_LINEAGE_ID,
    F_CONFIG_AGENT_TYPE, F_CONFIG_BENEFIT, F_CONFIG_COST,
    F_CONFIG_ELITE_COUNT, F_CONFIG_FALLBACK_INIT_COUNT,
    F_CONFIG_FALLBACK_MUTATION_COUNT, F_CONFIG_FERMI_BETA,
    F_CONFIG_FORBID_SELF_PAIRING, F_CONFIG_LLM_MAX_TOKENS,
    F_CONFIG_LLM_MODEL, F_CONFIG_LLM_THINKING,
    F_CONFIG_MUTATION_RATE_ON_ADOPTION, F_CONFIG_NUM_ELIMINATE,
    F_CONFIG_NUM_GENERATIONS, F_CONFIG_NUM_ROUNDS_PER_GEN,
    F_CONFIG_OBSERVABILITY, F_CONFIG_OBSERVABILITY_P,
    F_CONFIG_POPULATION_SIZE, F_CONFIG_SEED,
    F_CONFIG_TARGET_INTERACTIONS_PER_GEN, F_CONFIG_TOURNAMENT_SIZE,
    F_CONFIG_UPDATES_PER_GEN, F_CONFIG_USE_BASELINE, F_CONFIG_USE_FERMI,
)
from .agent import QuantitativeAgent
from .agent_full import (
    FullAgent, V3StrategyExecutor,
    ALLCClass, ALLDClass,
    ALLC_CLASS_SOURCE, ALLD_CLASS_SOURCE,
)
from .executor import V2StrategyExecutor
from .game import V2DonorGame
from .prompts import (
    INIT_PROMPT_V2, MUTATION_PROMPT_V2, SMUTATION_PROMPT_V2,
    INIT_PROMPT_V3, MUTATION_PROMPT_V3, SMALL_MUTATION_PROMPT_V3,
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
# whole population starts as 16/16 perfect cooperators -> 1.000
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
        # instance gets its own stream so a population of 16
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
        population_size: int = 16,
        num_rounds_per_gen: int = 30,
        # target_interactions_per_gen: if set (and > 0), overrides
        # num_rounds_per_gen at construction time so the caller can
        # think in terms of total PD games per gen rather than
        # rounds. Computed as ceil(target / (population_size // 2));
        # with N=16 we get 8 pairs per round, so target=1000 ->
        # 125 rounds = 1000 games. The LLM call count is governed
        # separately by num_eliminate (5/gen) and does NOT scale
        # with rounds, so going from 30 -> 125 rounds is ~4.17x
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
        # Selection rule. When use_fermi is True, the per-generation
        # update step is a synchronous Fermi imitation process
        # (Moran-process style) instead of tournament+elite. The
        # legacy tournament code path is kept behind use_fermi=False
        # for backward compatibility; see
        # _select_and_reproduce_fermi() for the implementation.
        #
        # Per update event we sample i (learner) and j (role model,
        # i != j) and apply
        #     P(i copies j) = 1 / (1 + exp(-fermi_beta * (phi_j - phi_i)))
        # with phi = per-agent windowed fitness from the just-finished
        # generation. On copy, with probability mutation_rate_on_adoption
        # the learner becomes an LLM-mutated version of j (counts as 1
        # LLM call); otherwise it becomes a verbatim copy of j's code
        # (no LLM call). Synchronous commit: all updates are decided
        # from the old generation's fitness, then written in one shot.
        #
        # Coverage math (N=16, 1000 inter/gen, updates_per_gen=15):
        #   Fermi events per gen  = 15
        #   Mutations per gen      = 15 * 0.1 = 1.5
        #   Fermi:mutation ratio   = 10:1 (drift-dominated)
        #   Pair coverage per gen  = 1.5% (relative to game inter)
        # Increase updates_per_gen to push selection strength up at
        # the cost of squashing heterogeneous LLM initial states faster.
        use_fermi: bool = False,
        fermi_beta: float = 5.0,
        mutation_rate_on_adoption: float = 0.1,
        updates_per_gen: int = 15,
        forbid_self_pairing: bool = True,
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
        # round count from it. With N=16 we get 8 pairs/round; with
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
        self.use_fermi = use_fermi
        self.fermi_beta = fermi_beta
        self.mutation_rate_on_adoption = mutation_rate_on_adoption
        self.updates_per_gen = updates_per_gen
        self.forbid_self_pairing = forbid_self_pairing
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
        # --- Lineage tracking (recorded directly, no post-hoc inference) ---
        # Each birth event (gen-0 init, Fermi imitation, independent μ-init,
        # or tournament mutation) is assigned a fresh, never-reused
        # `lineage_id`. A lineage persists as long as some slot keeps
        # carrying (or imitating) it. Because the Fermi path re-instantiates
        # the occupant object each update while PRESERVING its slot id, we
        # keep lineage state keyed by agent_id (= slot id in Fermi mode) on
        # the population manager, NOT on the agent object (which is rebuilt).
        self._next_lineage_id: int = 0
        # agent_id -> lineage_id of the slot's current occupant
        self._slot_lineage: Dict[int, int] = {}
        # agent_id -> birth record of the slot's current occupant
        self._slot_birth: Dict[int, Dict] = {}
        # global birth-event log (full phylogeny, incl. extinct lineages)
        self._lineage_events: List[Dict] = []
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

    def _new_lineage(
        self,
        slot_id: int,
        parent_slot_id: Optional[int],
        parent_lineage_id: Optional[int],
        origin: str,
        birth_gen: int,
    ) -> int:
        """Record one birth event and assign a fresh lineage_id.

        `origin` is one of:
          * "initial"          — gen-0 initialization (root, no parent)
          * "imitate"          — Fermi 1-μ path: small LLM mutation of a
                                 role model (parent = role model slot)
          * "independent_init" — Fermi μ path: fresh LLM init, no parent
          * "mutate"           — legacy tournament path: mutated copy of a
                                 survivor (parent = survivor slot)
        """
        lid = self._next_lineage_id
        self._next_lineage_id += 1
        rec = lineage_event(
            lineage_id=lid,
            parent_lineage_id=parent_lineage_id,
            parent_id=parent_slot_id,
            origin=origin,
            birth_gen=birth_gen,
        )
        self._slot_lineage[slot_id] = lid
        self._slot_birth[slot_id] = rec
        self._lineage_events.append(rec)
        return lid

    def _init_lineage(self) -> None:
        """Reset lineage state and register each initial agent as a root."""
        self._next_lineage_id = 0
        self._slot_lineage = {}
        self._slot_birth = {}
        self._lineage_events = []
        for a in self.agents:
            self._new_lineage(a.agent_id, None, None, ORIGIN_INITIAL, 0)

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

    def _agent_record(self, a) -> Dict:
        """Build one schema-compliant population record for agent `a`.

        Works identically for v2 (QuantitativeAgent) and v3 (FullAgent)
        because both expose `agent_id` / `code` / `fitness` /
        `cooperation_rate` / `get_self_reputation()`. Lineage fields come
        from the population manager's slot bookkeeping, so the record is
        uniform across both agent types.
        """
        birth = self._slot_birth.get(a.agent_id, {})
        return population_entry(
            agent_id=a.agent_id,
            code=a.code,
            fitness=a.fitness,
            cooperation_rate=a.cooperation_rate,
            self_reputation=a.get_self_reputation(),
            lineage_id=self._slot_lineage.get(a.agent_id),
            parent_id=birth.get(F_PARENT_ID),
            parent_lineage_id=birth.get(F_PARENT_LINEAGE_ID),
            origin=birth.get(F_ORIGIN),
            birth_gen=birth.get(F_BIRTH_GEN),
        )

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

    def _llm_init_one_agent(self, preserve_id: int) -> object:
        """Fermi μ-path: independent LLM agent generation.

        No reference to the donor j. The new strategy is sampled fresh
        from the LLM's prior over strategies. Returns a fully built
        agent (using preserve_id so the slot's id stays stable across
        the synchronous commit).

        FALLBACK on 3x LLM failure: a deterministic-random strategy
        (the FALLBACK_CLASS_V3 for v3; a random pick from
        FALLBACK_STRATEGIES for v2). Bumps _fallback_mutation_count
        so we can audit run reliability.
        """
        if self.agent_type == "v3":
            user_msg = INIT_PROMPT_V3
        else:
            user_msg = INIT_PROMPT_V2
        for attempt in range(3):
            content = self._call_llm(
                "You are a Python programmer. Output only valid Python code.",
                user_msg,
            )
            code = _extract_code_from_response(content) if content else None
            if code:
                try:
                    return self._make_agent(code, preserve_id)
                except Exception as e:
                    print(f"  [fermi μ-init validate fail attempt {attempt+1}]: {e}")
        # 3x failed: FALLBACK (same shape as _init_population_llm).
        self._fallback_mutation_count += 1
        print(f"  [fermi μ-init] FALLBACK for slot id={preserve_id}")
        if self.agent_type == "v3":
            fb = FALLBACK_CLASS_V3
        else:
            fb = self.rng.choice(FALLBACK_STRATEGIES)
        return self._make_agent(fb, preserve_id)

    def _llm_small_mutate(self, parent_code: str, preserve_id: int) -> object:
        """Fermi 1-μ path: small variant of parent.

        The parent code IS shown to the LLM (this is the whole point
        of "imitate with tiny mutation" — the offspring is
        recognizably the parent's strategy with a small perturbation).
        Contrast with the μ path which uses no parent reference.

        FALLBACK on 3x LLM failure: the parent code verbatim (the
        smallest possible mutation). Bumps _fallback_mutation_count.
        """
        if self.agent_type == "v3":
            user_msg = SMALL_MUTATION_PROMPT_V3.format(parent_code=parent_code)
        else:
            user_msg = SMUTATION_PROMPT_V2.format(fitness=0.0, parent_code=parent_code)
        for attempt in range(3):
            content = self._call_llm(
                "You are a Python programmer. Output only valid Python code.",
                user_msg,
            )
            code = _extract_code_from_response(content) if content else None
            if code:
                try:
                    return self._make_agent(code, preserve_id)
                except Exception as e:
                    print(f"  [fermi 1-μ small-mutate validate fail attempt {attempt+1}]: {e}")
        # 3x failed: parent code verbatim (the smallest mutation).
        self._fallback_mutation_count += 1
        print(f"  [fermi 1-μ small-mutate] FALLBACK (parent verbatim) for slot id={preserve_id}")
        return self._make_agent(parent_code, preserve_id)

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
        # Register the gen-0 population as root lineages.
        self._init_lineage()
        trajectory: List[Dict] = []
        # Generation 0: just initialize; we record initial stats by running
        # one generation with the initial population (no selection/mutation yet)
        for gen in range(num_generations):
            self._round_offset = gen
            stats = self._run_one_generation()
            # Update fitness on agents
            for i, a in enumerate(self.agents):
                a.fitness = stats["payoffs"][i] if i < len(stats["payoffs"]) else 0.0
            trajectory.append(trajectory_entry(
                generation=gen,
                cooperation_rate_mean=stats["cooperation_rate_mean"],
                n_interactions=stats["n_interactions"],
                fitness_mean=sum(stats["payoffs"]) / max(1, len(stats["payoffs"])),
                fitness_max=max(stats["payoffs"]) if stats["payoffs"] else 0.0,
                population=[self._agent_record(a) for a in self.agents],
            ))
            print(f"  Gen {gen}: coop={stats['cooperation_rate_mean']:.3f}, "
                  f"fitness_mean={sum(stats['payoffs'])/max(1,len(stats['payoffs'])):.1f}")
            # Selection + mutation (only for LLM mode)
            if not self.use_baseline and gen < num_generations - 1:
                if self.use_fermi:
                    self._select_and_reproduce_fermi(next_gen=gen + 1)
                else:
                    self._select_and_reproduce(next_gen=gen + 1)
        # Build final population
        final_population = [self._agent_record(a) for a in self.agents]
        # FALLBACK diagnostics (Fix E). Print init and mutation
        # FALLBACK ratios so reviewers can judge run reliability.
        # Init ratio >30% usually means the LLM init is broken
        # (model name, API key, thinking-mode mismatch); mutation
        # ratio >30% usually means the model is producing invalid
        # code at a high rate.
        init_ratio = self._fallback_init_count / max(1, self.population_size)
        if self.use_fermi:
            # Z-like: every Fermi copy event triggers exactly one LLM
            # call (μ path = init, 1-μ path = small_mutate). Upper
            # bound on LLM calls is updates_per_gen per gen.
            mut_total = int(
                (num_generations - 1) * self.updates_per_gen
            ) if num_generations > 1 else 0
        else:
            mut_total = (num_generations - 1) * self.num_eliminate if num_generations > 1 else 0
        mut_ratio = self._fallback_mutation_count / max(1, mut_total)
        print(
            f"  [FALLBACK stats] init={self._fallback_init_count}/"
            f"{self.population_size} ({init_ratio:.0%}), "
            f"mutation={self._fallback_mutation_count}/{mut_total} "
            f"({mut_ratio:.0%}), thinking={self.llm_thinking}, "
            f"max_tokens={self._llm_max_tokens}, "
            f"use_fermi={self.use_fermi} (beta={self.fermi_beta}, "
            f"mu={self.mutation_rate_on_adoption}, updates/gen={self.updates_per_gen})"
        )
        if init_ratio > 0.3:
            print(
                f"  [FALLBACK warning] init FALLBACK ratio > 30% "
                f"— LLM init is likely broken; run "
                f"results are NOT reliable."
            )
        return build_evolution_results(
            trajectory=trajectory,
            final_population=final_population,
            # Full birth-event log: the complete phylogeny (roots + every
            # imitation / independent-init / mutation birth, including
            # lineages that later went extinct). Together with the
            # per-agent lineage_id/parent_id fields this lets the
            # evolutionary tree be built directly, with no code-similarity
            # inference.
            lineage_events=self._lineage_events,
            config=make_config(
                **{
                    F_CONFIG_AGENT_TYPE: self.agent_type,
                    F_CONFIG_POPULATION_SIZE: self.population_size,
                    F_CONFIG_NUM_ROUNDS_PER_GEN: self.num_rounds_per_gen,
                    F_CONFIG_BENEFIT: self.benefit,
                    F_CONFIG_COST: self.cost,
                    F_CONFIG_OBSERVABILITY: self.observability,
                    F_CONFIG_OBSERVABILITY_P: self.observability_p,
                    F_CONFIG_ELITE_COUNT: self.elite_count,
                    F_CONFIG_NUM_ELIMINATE: self.num_eliminate,
                    F_CONFIG_TOURNAMENT_SIZE: self.tournament_size,
                    F_CONFIG_LLM_MODEL: self.llm_model,
                    F_CONFIG_SEED: self.seed,
                    F_CONFIG_USE_BASELINE: self.use_baseline,
                    F_CONFIG_NUM_GENERATIONS: num_generations,
                    F_CONFIG_TARGET_INTERACTIONS_PER_GEN:
                        self.target_interactions_per_gen,
                    F_CONFIG_LLM_THINKING: self.llm_thinking,
                    F_CONFIG_LLM_MAX_TOKENS: self._llm_max_tokens,
                    F_CONFIG_USE_FERMI: self.use_fermi,
                    F_CONFIG_FERMI_BETA: self.fermi_beta,
                    F_CONFIG_MUTATION_RATE_ON_ADOPTION:
                        self.mutation_rate_on_adoption,
                    F_CONFIG_UPDATES_PER_GEN: self.updates_per_gen,
                    F_CONFIG_FORBID_SELF_PAIRING: self.forbid_self_pairing,
                    F_CONFIG_FALLBACK_INIT_COUNT: self._fallback_init_count,
                    F_CONFIG_FALLBACK_MUTATION_COUNT:
                        self._fallback_mutation_count,
                }
            ),
        )

    def _select_and_reproduce(self, next_gen: Optional[int] = None):
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
        # Full re-instantiation per generation (lifecycle = one
        # generation): survivors are rebuilt from their own code with
        # the SAME agent_id. Both internal state AND the reputation
        # matrix reset each generation (no cross-gen memory); only
        # lineage (bloodline) persists. Applies to both v2 and v3
        # agents.
        new_agents = []
        for a in survivors[:n_needed]:
            try:
                new_agents.append(self._make_agent(a.code, a.agent_id))
            except Exception as e:
                # Defensive: survivor code already instantiated
                # successfully before, so this should never fire;
                # keep the old object rather than crash the run.
                print(
                    f"  [_select_and_reproduce re-instantiate fallback] "
                    f"{type(e).__name__}: {e}"
                )
                new_agents.append(a)
        if next_gen is None:
            next_gen = 1
        for _ in range(N - n_needed):
            parent = self.rng.choice(survivors)
            new_code = self._mutate(parent.code, parent.fitness)
            try:
                child = self._new_agent(new_code)
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
                child = self._new_agent(parent.code)
            new_agents.append(child)
            self._new_lineage(
                child.agent_id,
                parent.agent_id,
                self._slot_lineage.get(parent.agent_id),
                ORIGIN_MUTATE,
                next_gen,
            )
        # Reputations reset every generation: each agent is rebuilt
        # (or newly imitated) with its own initial matrix
        # {agent_id: INITIAL_REPUTATION}, so old ids never appear in
        # the new agents' matrices. The pop loop below is therefore a
        # no-op; kept as defense-in-depth in case reputation carryover
        # is ever re-enabled.
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

    def _select_and_reproduce_fermi(self, next_gen: Optional[int] = None):
        """Synchronous Fermi imitation + LLM mutation (Moran-process style, Z-like).

        Per generation we run `updates_per_gen` independent update
        events. For each event we sample (i, j) with i != j (when
        forbid_self_pairing=True) and apply

            P(i copies j) = 1 / (1 + exp(-fermi_beta * (phi_j - phi_i)))

        where phi is the per-agent windowed fitness from the just-
        finished generation. On copy, with probability
        mutation_rate_on_adoption the offspring is an INDEPENDENT
        LLM init (no reference to j); with probability 1-mu the
        offspring is a SMALL LLM mutation of j's code (j is shown to
        the LLM, the prompt asks for a tiny change). Both paths
        always perform exactly one LLM call per copy event.

        All decisions are made from the old generation's fitness+code
        and committed synchronously at the end (Moran style, no in-
        place mutation of j that other events could read).

        μ=0 degenerate: with mutation_rate_on_adoption=0 the 1-μ
        path is still LLM-mutate, NOT verbatim copy. This is the
        Z-like scheme (vs the Y scheme where 1-μ was free verbatim).
        To get pure Fermi + no mutation, set
        mutation_rate_on_adoption=1 so every copy is a free LLM
        init — but note: that still costs LLM calls. For pure
        replicator dynamics, run with use_fermi=False (legacy
        tournament+elite path, no LLM in selection step).

        Sanity checks (should pass):
          * Fermi + ALLC, mu=0       -> stays at 1.0
          * Fermi + ALLD, mu=0       -> stays at 0.0
          * Fermi + 1 IS+ + 14 ALLD, mu=0 -> 14/1 (IS+ invades)
          * Fermi + 1 ALLD + 14 ALLC, mu=0 -> 15/0 (ALLD contained)
        """
        import math
        N = len(self.agents)
        if N < 2:
            return  # nothing to update
        beta = self.fermi_beta
        mu = self.mutation_rate_on_adoption
        # Build a fresh list; we'll mutate entries in-place, but
        # always read phi and code from the OLD generation.
        next_agents = list(self.agents)
        old_agents = list(self.agents)
        # slot agent_id -> (parent agent_id or None, parent_lineage or None, origin)
        updates = {}
        for _ in range(self.updates_per_gen):
            # Sample learner i
            i = self.rng.randrange(N)
            # Sample role model j != i (unless population has 1)
            if self.forbid_self_pairing and N > 1:
                j = self.rng.randrange(N - 1)
                if j >= i:
                    j += 1
            else:
                j = self.rng.randrange(N)
            # Fermi imitation probability
            phi_i = old_agents[i].fitness
            phi_j = old_agents[j].fitness
            try:
                p_imitate = 1.0 / (1.0 + math.exp(-beta * (phi_j - phi_i)))
            except OverflowError:
                # exp(-beta * large_negative) underflows to 0; p -> 1
                p_imitate = 0.0 if (phi_j - phi_i) < 0 else 1.0
            if self.rng.random() >= p_imitate:
                continue  # no update this event
            # i imitates j. Construct the offspring:
            #   with prob mu  -> INDEPENDENT LLM init (no j reference)
            #   with prob 1-mu -> SMALL LLM mutation of j.code
            # In both cases exactly one LLM call per copy event.
            if self.rng.random() < mu:
                new_agent = self._llm_init_one_agent(old_agents[i].agent_id)
                updates[new_agent.agent_id] = (None, None, ORIGIN_INDEPENDENT_INIT)
            else:
                new_agent = self._llm_small_mutate(
                    old_agents[j].code, old_agents[i].agent_id
                )
                # Parent lineage is j's lineage from the OLD generation
                # (synchronous commit: j's code hasn't been overwritten yet).
                updates[new_agent.agent_id] = (
                    old_agents[j].agent_id,
                    self._slot_lineage.get(old_agents[j].agent_id),
                    ORIGIN_IMITATE,
                )
            next_agents[i] = new_agent
        # Synchronous commit. The set of agent_ids is preserved
        # (every slot retains its old id). Reputations are NOT
        # inherited: each generation is a fresh lifecycle, so every
        # agent (rebuilt or newly imitated) starts from its own
        # initial matrix {agent_id: INITIAL_REPUTATION}.
        for slot, new_a in enumerate(next_agents):
            old_a = old_agents[slot]
            if new_a is old_a:
                # Full re-instantiation per generation (lifecycle =
                # one generation): an untouched slot is rebuilt from
                # its own code with the SAME agent_id, so agent
                # internal state starts fresh each generation.
                # Applies to both v2 and v3 agents. No new lineage
                # event is recorded for a rebuild (the strategy
                # bloodline is unchanged).
                new_a = self._make_agent(old_a.code, old_a.agent_id)
                next_agents[slot] = new_a
        self.agents = next_agents
        # Record lineage for every slot that was updated this generation.
        # Unchanged slots keep their existing lineage (same occupant).
        if next_gen is None:
            next_gen = 1
        for slot, (parent_slot, parent_lineage, origin) in updates.items():
            self._new_lineage(slot, parent_slot, parent_lineage, origin, next_gen)
