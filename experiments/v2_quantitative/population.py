"""Population + LLM-driven evolution for the v2 quantitative interface.

Mirrors the v1 EvolutionaryPopulation but uses the v2 QuantitativeAgent
and DonorGame.

Supports two agent types:
  - `agent_type="agent-type1"`: type-1 agents. The LLM
    emits two top-level Python functions (`observe` + `decide`); the
    framework maintains a scalar reputation matrix for them. observe()
    is ONE-DIRECTIONAL: it judges a single target player and returns
    that player's new reputation; the framework calls it twice per
    joint action (roles swapped) to update both players.
  - `agent_type="agent-type2"`: type-2 agents. The LLM
    emits a full Python class named `LLMAgent` with `__init__(agent_id)`,
    `decide()`, and `observe(...)` methods. The LLM owns its own
    internal state structure (dicts, lists, counters — anything). The
    framework still maintains a scalar `reputations` matrix for
    bookkeeping, but the LLM is not required to read it.

Type 2 baseline mode currently only supports ALLC and ALLD
(via the ALLC_CLASS_SOURCE / ALLD_CLASS_SOURCE source strings for the
trivial always-cooperate / always-defect strategies). The 8
leading-eight rules live in type-1 land.
"""
from __future__ import annotations
import copy
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..evolution_log import (
    ORIGIN_INITIAL,
    build_evolution_results, lineage_event, make_config, population_entry,
    trajectory_entry,
    F_BIRTH_GEN, F_ORIGIN, F_PARENT_ID, F_PARENT_LINEAGE_ID,
    F_CONFIG_AGENT_TYPE, F_CONFIG_BENEFIT, F_CONFIG_COST,
    F_CONFIG_ELITE_COUNT, F_CONFIG_FALLBACK_INIT_COUNT,
    F_CONFIG_FALLBACK_MUTATION_COUNT, F_CONFIG_FERMI_BETA,
    F_CONFIG_LEARNING_METHOD,
    F_CONFIG_LLM_MAX_TOKENS,
    F_CONFIG_IMITATION_LEARNING_MODE,
    F_CONFIG_LLM_MODEL, F_CONFIG_LLM_THINKING,
    F_CONFIG_MUTATION_RATE_ON_ADOPTION, F_CONFIG_NUM_ELIMINATE,
    F_CONFIG_NUM_GENERATIONS, F_CONFIG_NUM_ROUNDS_PER_GEN,
    F_CONFIG_OBSERVABILITY, F_CONFIG_OBSERVABILITY_P,
    F_CONFIG_POPULATION_SIZE, F_CONFIG_SEED,
    F_CONFIG_TARGET_INTERACTIONS_PER_GEN, F_CONFIG_TOURNAMENT_SIZE,
    F_CONFIG_UPDATES_PER_GEN, F_CONFIG_USE_BASELINE,
    F_CONFIG_INITIAL_REPUTATION, F_CONFIG_LLM_CONCURRENCY,
    F_CONFIG_FITNESS_WINDOW_FRACTION,
    F_CONFIG_OBSERVATION_SCHEDULE,
    F_CONFIG_ACTION_ERROR, F_CONFIG_OBSERVATION_ERROR,
)
from .agent import INITIAL_REPUTATION, QuantitativeAgent
from .agent_full import (
    INITIAL_REPUTATION as FULL_INITIAL_REPUTATION,
    FullAgent, V3StrategyExecutor,
    ALLC_CLASS_SOURCE, ALLD_CLASS_SOURCE,
)
from .executor import V2StrategyExecutor
from .evolution_architecture import (
    AgentSnapshot,
    EvolutionRule,
    FermiEvolutionRule,
    GameScenario,
    GenerationPlan,
    OffspringJob,
    OffspringGenerator,
    OffspringResult,
    ReputationPrisonersDilemmaScenario,
    TournamentEvolutionRule,
)
from .game import OBSERVATION_SCHEDULE
from .prompts import (
    INIT_PROMPT_V2, MUTATION_PROMPT_V2, SMUTATION_PROMPT_V2,
    DELIBERATE_MUTATION_PROMPT_V2,
    INIT_PROMPT_V3, MUTATION_PROMPT_V3, SMALL_MUTATION_PROMPT_V3,
    DELIBERATE_MUTATION_PROMPT_V3,
)
from .baselines import get_baseline, BASELINE_VERSION


FITNESS_METRIC = "mean_payoff_per_counted_action_v1"


# Fallback strategies when LLM fails (type 1: two top-level functions)
FALLBACK_STRATEGIES = [
    # Always cooperate
    '''
def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    if A_action == 'cooperate':
        new = A_rep + 0.333
    else:
        new = A_rep - 0.333
    return max(-1.0, min(1.0, new))
def decide(my_reputation, opponent_reputation):
    return True
''',
    # Always defect
    '''
def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    if A_action == 'cooperate':
        new = A_rep + 0.333
    else:
        new = A_rep - 0.333
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
        # rounds. One pair plays per interaction, so target=1000 ->
        # 1000 interactions. The LLM call count is governed separately
        # by num_eliminate (5/gen) and does NOT scale with the
        # interaction count, so more game time means the same ~5 LLM
        # calls per gen.
        target_interactions_per_gen: Optional[int] = None,
        # fitness_window_fraction: the trailing share of joint actions
        # used for selection. Fitness is each agent's payoff in that
        # window divided by its own action count there; earlier actions
        # are burn-in but still update reputations. Default 0.2. Pass
        # None (or 0) to use the whole generation.
        fitness_window_fraction: Optional[float] = 0.2,
        benefit: float = 3.0,
        cost: float = 1.0,
        # Noise. Both default to 0.0; with no noise these knobs consume
        # no RNG draws and do not change game action trajectories.
        #   action_error_probability      -- execution error ("trembling
        #     hand"): each player's intended action is mis-executed with
        #     this probability. The executed action drives payoffs and is
        #     what every observer sees.
        #   observation_error_probability -- perception error: each observer
        #     independently misperceives each action it rates with this
        #     probability, corrupting the reputation written (never the
        #     payoff paid). Participants rating themselves are observers too.
        # Names deliberately match the invasion / fixation / consensus
        # config keys so evolution and measurement share one noise model.
        action_error_probability: float = 0.0,
        observation_error_probability: float = 0.0,
        # Total number of generations to evolve. Injected into the
        # prompt templates so the LLM sees the real simulation
        # horizon (default 30 matches the legacy hardcoded text).
        num_generations: int = 30,
        observability: str = "full",
        observability_p: float = 1.0,
        elite_count: int = 2,
        num_eliminate: int = 5,
        tournament_size: int = 3,
        # Selection rule. The per-generation update step is a
        # synchronous Fermi imitation process (Moran-process style) by
        # default; ``learning_method="tournament"`` selects the
        # tournament+elite path instead. See
        # _select_and_reproduce_fermi() and _select_and_reproduce()
        # for the implementations.
        #
        # Per update event we sample i (learner) and j (role model,
        # i != j) and apply
        #     P(i copies j) = 1 / (1 + exp(-fermi_beta * (phi_j - phi_i)))
        # with phi = per-agent windowed fitness from the just-finished
        # generation. On an accepted copy, probability
        # mutation_rate_on_adoption selects an independent LLM rewrite;
        # otherwise the LLM creates a parent-conditioned child using
        # imitation_learning_mode ("random" or "deliberate"). The actual
        # role-model fitness is included in either parent-conditioned prompt.
        learning_method: str = "fermi",
        fermi_beta: float = 5.0,
        mutation_rate_on_adoption: float = 0.1,
        imitation_learning_mode: str = "random",
        updates_per_gen: Optional[int] = None,
        llm_concurrency: Optional[int] = None,
        llm_provider: str = "openai",
        llm_model: str = "deepseek-v4-flash",
        api_key: str = "",
        api_base_url: str = "",
        mutation_temperature: float = 0.8,
        seed: int = 42,
        results_dir: str = "results",
        use_baseline: Optional[str] = None,
        agent_type: str = "agent-type1",
        # If use_baseline is set, all agents use that baseline strategy and
        # the LLM is not used. If None, LLM evolution runs.
        # agent_type:
        #   "agent-type1" (default) — type-1 agents. LLM
        #       emits two top-level functions (`observe` + `decide`);
        #       framework maintains a scalar `reputations` dict.
        #   "agent-type2" — type-2 agents. LLM emits a
        #       full `LLMAgent` class with `__init__(agent_id)`,
        #       `decide()`, and `observe(...)` methods. LLM owns its own
        #       state structure; framework only handles bookkeeping.
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
        game_scenario: Optional[GameScenario] = None,
        evolution_rule: Optional[EvolutionRule] = None,
        offspring_generator: Optional[OffspringGenerator] = None,
    ):
        if agent_type not in ("agent-type1", "agent-type2"):
            raise ValueError(
                f"agent_type must be 'agent-type1' or 'agent-type2', "
                f"got {agent_type!r}"
            )
        if imitation_learning_mode not in ("random", "deliberate"):
            raise ValueError(
                "imitation_learning_mode must be 'random' or 'deliberate', "
                f"got {imitation_learning_mode!r}"
            )
        learning_method = learning_method.lower()
        if learning_method not in ("fermi", "tournament"):
            raise ValueError(
                "learning_method must be 'fermi' or 'tournament', "
                f"got {learning_method!r}"
            )
        self.population_size = population_size
        # The framework implements one protocol: a single randomly drawn pair
        # per interaction, with its observations delivered immediately. The
        # value is recorded in each result config so archives stay
        # self-describing.
        self.observation_schedule = OBSERVATION_SCHEDULE
        # If the caller asked for a target interaction count, that IS the step
        # count, because exactly one pair plays per interaction.
        if target_interactions_per_gen is not None and target_interactions_per_gen > 0:
            num_rounds_per_gen = target_interactions_per_gen
        self.num_rounds_per_gen = num_rounds_per_gen
        self.target_interactions_per_gen = target_interactions_per_gen
        if fitness_window_fraction is not None:
            try:
                fitness_window_fraction = float(fitness_window_fraction)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "fitness_window_fraction must be a number in [0, 1] or "
                    f"None, got {fitness_window_fraction!r}"
                ) from exc
            if not 0.0 <= fitness_window_fraction <= 1.0:
                raise ValueError(
                    "fitness_window_fraction must be in [0, 1] or None, "
                    f"got {fitness_window_fraction!r}"
                )
        self.fitness_window_fraction = fitness_window_fraction
        self.benefit = benefit
        self.cost = cost
        for noise_name, noise_value in (
            ("action_error_probability", action_error_probability),
            ("observation_error_probability", observation_error_probability),
        ):
            if not 0.0 <= float(noise_value) <= 1.0:
                raise ValueError(
                    f"{noise_name} must be in [0, 1], got {noise_value!r}"
                )
        self.action_error_probability = float(action_error_probability)
        self.observation_error_probability = float(observation_error_probability)
        self.num_generations = num_generations
        self.observability = observability
        self.observability_p = observability_p
        self.elite_count = elite_count
        self.num_eliminate = num_eliminate
        self.tournament_size = tournament_size
        self.learning_method = learning_method
        self.fermi_beta = fermi_beta
        self.mutation_rate_on_adoption = mutation_rate_on_adoption
        self.imitation_learning_mode = imitation_learning_mode
        effective_updates_per_gen = (
            population_size if updates_per_gen is None else updates_per_gen
        )
        if effective_updates_per_gen < 0 or effective_updates_per_gen > population_size:
            raise ValueError(
                "updates_per_gen must be between 0 and population_size "
                "for without-replacement learner sampling"
            )
        self.updates_per_gen = effective_updates_per_gen
        if llm_concurrency is not None and llm_concurrency < 1:
            raise ValueError("llm_concurrency must be >= 1")
        self.llm_concurrency = llm_concurrency or population_size
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.mutation_temperature = mutation_temperature
        self.seed = seed
        self.results_dir = Path(results_dir)
        self.use_baseline = use_baseline
        self.agent_type = agent_type
        self.game_scenario = game_scenario or ReputationPrisonersDilemmaScenario(
            population_size=population_size,
            benefit=benefit,
            cost=cost,
            observability=observability,
            observability_p=observability_p,
            fitness_window_fraction=fitness_window_fraction,
            num_rounds_per_gen=num_rounds_per_gen,
            action_error_probability=self.action_error_probability,
            observation_error_probability=self.observation_error_probability,
        )
        # A supplied rule bypasses the legacy string dispatcher.  The default
        # paths retain their public methods for backward compatibility.
        self.evolution_rule = evolution_rule
        self.offspring_generator = offspring_generator
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
        self._llm_client_lock = threading.Lock()
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
            actual = self.num_rounds_per_gen
            print(
                f"  [V2EvolutionaryPopulation] target_interactions_per_gen="
                f"{target_interactions_per_gen} -> num_rounds_per_gen="
                f"{self.num_rounds_per_gen} -> {actual} games/gen "
                f"(N={population_size}, one pair per interaction)"
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
        with self._llm_client_lock:
            if self._llm_client is None:
                import openai
                self._llm_client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base_url,
                )
        return self._llm_client

    def _parallel_llm_map(self, fn, jobs):
        """Run one independent batch of LLM jobs, preserving input order."""
        jobs = list(jobs)
        if not jobs:
            return []
        if self.llm_concurrency == 1 or len(jobs) == 1:
            return [fn(job) for job in jobs]
        # The worker owns any provider initialization it needs.  `_get_llm_client`
        # is lock-protected, while keeping this generic map provider-agnostic
        # allows injected/local offspring operators to run without credentials.
        with ThreadPoolExecutor(
            max_workers=min(self.llm_concurrency, len(jobs)),
            thread_name_prefix="llm",
        ) as executor:
            return list(executor.map(fn, jobs))

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
                if attempt + 1 < max_retries:
                    time.sleep(2 ** attempt)
        return None

    def _make_agent(self, code: str, agent_id: int):
        """Validate `code` and instantiate one agent of the configured type."""
        if self.agent_type == "agent-type2":
            executor = V3StrategyExecutor(code)
            return FullAgent(agent_id, executor=executor, code=code)
        # agent-type1 (default)
        executor = V2StrategyExecutor(code)
        return QuantitativeAgent(agent_id, code, executor=executor)

    def _new_agent(self, code: str):
        """Allocate a fresh, never-reused agent_id and create the agent."""
        aid = self._next_agent_id
        self._next_agent_id += 1
        return self._make_agent(code, aid)

    def _agent_record(self, a) -> Dict:
        """Build one schema-compliant population record for agent `a`.

        Works identically for agent-type1 (QuantitativeAgent) and
        agent-type2 (FullAgent) because both expose `agent_id` / `code` / `fitness` /
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

        For agent-type1: instantiates the V2StrategyExecutor (which loads
        observe and decide). For agent-type2: instantiates the
        V3StrategyExecutor (which loads the LLMAgent class). Raises on
        any error.
        """
        if self.agent_type == "agent-type2":
            V3StrategyExecutor(code)
        else:
            V2StrategyExecutor(code)

    def _request_valid_code(self, user_msg: str, label: str) -> Optional[str]:
        """Make at most three API calls and return the first valid strategy."""
        system_msg = "You are a Python programmer. Output only valid Python code."
        # Inject the authoritative game mechanics at the final common gateway,
        # so init, tournament mutation, and both Fermi mutation modes (for both
        # agent interfaces) cannot accidentally omit them.
        user_msg = (
            self._overall_game_rules_prompt()
            + "\n\nSTRATEGY-GENERATION TASK:\n"
            + user_msg
        )
        for attempt in range(3):
            content = self._call_llm(system_msg, user_msg, max_retries=1)
            code = _extract_code_from_response(content) if content else None
            if code:
                try:
                    self._validate_code(code)
                    return code
                except Exception as exc:
                    print(f"  [{label} validation fail attempt {attempt+1}]: {exc}")
            if attempt < 2:
                time.sleep(2 ** attempt)
        return None

    def _sim_params(self) -> Dict[str, object]:
        """Simulation parameters injected into prompt templates.

        The LLM never sees template placeholders; these values describe
        the actual simulation the generated strategies will run in, so
        the prompts stay in sync with population_size / rounds / benefit
        / cost / generations even when they differ from the defaults.
        """
        return dict(self._get_game_scenario().simulation_parameters(
            num_generations=self.num_generations,
            initial_reputation=INITIAL_REPUTATION,
        ))

    def _get_game_scenario(self) -> GameScenario:
        """Return the configured scenario, including support for legacy fixtures.

        Some focused tests and downstream callers construct the population with
        ``__new__`` and fill only the historical attributes.  Lazily creating
        the default adapter preserves that supported testing pattern.
        """
        scenario = getattr(self, "game_scenario", None)
        if scenario is None:
            scenario = ReputationPrisonersDilemmaScenario(
                population_size=self.population_size,
                benefit=self.benefit,
                cost=self.cost,
                observability=getattr(self, "observability", "full"),
                observability_p=getattr(self, "observability_p", 1.0),
                fitness_window_fraction=getattr(
                    self, "fitness_window_fraction", 0.2
                ),
                num_rounds_per_gen=self.num_rounds_per_gen,
                action_error_probability=getattr(
                    self, "action_error_probability", 0.0
                ),
                observation_error_probability=getattr(
                    self, "observation_error_probability", 0.0
                ),
            )
            self.game_scenario = scenario
        return scenario

    def _overall_game_rules_prompt(self) -> str:
        """Return the authoritative mechanics prepended to every LLM task."""
        return self._get_game_scenario().overall_rules_prompt(
            num_generations=self.num_generations,
            initial_reputation=INITIAL_REPUTATION,
        )

    def _init_prompt(self) -> str:
        """The agent-type-appropriate init prompt with real sim params."""
        template = (
            INIT_PROMPT_V3 if self.agent_type == "agent-type2" else INIT_PROMPT_V2
        )
        return template.format(**self._sim_params())

    def _init_population_llm(self):
        """Generate N strategies via LLM.

        FALLBACK is the deterministic-random class (Fix C) — not
        the old always-cooperate one. Every FALLBACK hit increments
        self._fallback_init_count so we can audit run reliability at
        the end.
        """
        print(f"  Initializing population via LLM ({self.population_size} agents, agent_type={self.agent_type}, thinking={self.llm_thinking}, max_tokens={self._llm_max_tokens})...")
        user_msg = self._init_prompt()

        def generate_one(i):
            return self._request_valid_code(user_msg, f"init agent {i}")

        codes = self._parallel_llm_map(generate_one, range(self.population_size))
        for i, code in enumerate(codes):
            if code is None:
                fb = FALLBACK_CLASS_V3 if self.agent_type == "agent-type2" else self.rng.choice(FALLBACK_STRATEGIES)
                code = fb
                self._fallback_init_count += 1
                print(f"  [init agent {i}] using FALLBACK strategy")
            self.agents.append(self._new_agent(code))

    def _init_population_baseline(self):
        """All agents use the same baseline strategy."""
        assert self.use_baseline is not None
        if self.agent_type == "agent-type2":
            # Only ALLC / ALLD supported as type-2 baselines
            t2_baselines = {"ALLC": ALLC_CLASS_SOURCE, "ALLD": ALLD_CLASS_SOURCE}
            if self.use_baseline not in t2_baselines:
                raise ValueError(
                    f"agent_type='agent-type2' only supports ALLC / ALLD as baselines; "
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
                if self.agent_type == "agent-type2":
                    fb = FALLBACK_CLASS_V3
                else:
                    fb = self.rng.choice(FALLBACK_STRATEGIES)
                self.agents.append(self._new_agent(fb))
        print(f"  Initialized {len(self.agents)} agents with baseline '{self.use_baseline}' (agent_type={self.agent_type})")

    def _mutate_code(
        self, parent_code: str, parent_fitness: float
    ) -> Optional[str]:
        """Generate validated mutation code without applying fallback.

        Returning ``None`` keeps failure handling and metrics in the ordered,
        single-threaded commit phase.
        """
        sim = self._sim_params()
        if self.agent_type == "agent-type2":
            user_msg = MUTATION_PROMPT_V3.format(
                fitness=parent_fitness, parent_code=parent_code, **sim
            )
        else:
            user_msg = MUTATION_PROMPT_V2.format(
                fitness=parent_fitness, parent_code=parent_code, **sim
            )
        return self._request_valid_code(user_msg, "mutate")

    def _mutate(self, parent_code: str, parent_fitness: float) -> str:
        """Backward-compatible single-mutation helper with parent fallback."""
        return self._mutate_code(parent_code, parent_fitness) or parent_code

    def _llm_init_code(self, preserve_id: int) -> Optional[str]:
        """Generate and validate code for one independent Fermi update.

        The μ path: no reference to the donor j is shown to the LLM, so
        the new strategy is sampled fresh from the LLM's prior over
        strategies. Contrast with ``_llm_small_mutate_code``, which
        does show the parent code.

        Returns ``None`` on repeated LLM/validation failure; FALLBACK
        selection and ``_fallback_mutation_count`` bookkeeping happen in
        the ordered commit phase (``_fallback_code_for_job``), not here.
        """
        return self._request_valid_code(
            self._init_prompt(), f"fermi μ-init slot {preserve_id}"
        )

    def _llm_small_mutate_code(
        self, parent_code: str, parent_fitness: float, preserve_id: int
    ) -> Optional[str]:
        """Generate and validate code for one parent-conditioned update."""
        if self.agent_type == "agent-type2":
            template = (
                DELIBERATE_MUTATION_PROMPT_V3
                if self.imitation_learning_mode == "deliberate"
                else SMALL_MUTATION_PROMPT_V3
            )
        else:
            template = (
                DELIBERATE_MUTATION_PROMPT_V2
                if self.imitation_learning_mode == "deliberate"
                else SMUTATION_PROMPT_V2
            )
        user_msg = template.format(
            fitness=parent_fitness,
            parent_code=parent_code,
            **self._sim_params(),
        )
        return self._request_valid_code(
            user_msg, f"fermi 1-μ small-mutate slot {preserve_id}"
        )

    def _llm_small_mutate(
        self, parent_code: str, parent_fitness: float, preserve_id: int
    ) -> object:
        """Fermi 1-μ path: configurable parent-conditioned child generation.

        The parent code IS shown to the LLM (this is the whole point
        of "imitate with tiny mutation" — the offspring is
        related to the parent's strategy).
        Contrast with the μ path which uses no parent reference.

        FALLBACK on 3x LLM failure: the parent code verbatim (the
        smallest possible mutation). Bumps _fallback_mutation_count.
        """
        code = self._llm_small_mutate_code(
            parent_code, parent_fitness, preserve_id
        )
        if code is None:
            self._fallback_mutation_count += 1
            print(
                "  [fermi 1-μ small-mutate] FALLBACK (parent verbatim) "
                f"for slot id={preserve_id}"
            )
            code = parent_code
        return self._make_agent(code, preserve_id)

    def _run_one_generation(self) -> Dict:
        """Run a single generation. Returns per-gen stats."""
        gen_seed = self.rng.randrange(10**9)
        result = self._get_game_scenario().evaluate(
            self.agents,
            generation_seed=gen_seed,
            num_rounds=self.num_rounds_per_gen,
        )
        return result.as_dict()

    @property
    def round_num_offset(self) -> int:
        return getattr(self, "_round_offset", 0)

    @staticmethod
    def _tuple_tree(value):
        """Convert JSON-loaded RNG-state lists back to tuples."""
        if isinstance(value, list):
            return tuple(V2EvolutionaryPopulation._tuple_tree(v) for v in value)
        return value

    def _result_config(self, num_generations: int, **extra) -> Dict:
        """Build the common config block, including a resumable RNG checkpoint."""
        scenario = self._get_game_scenario()
        custom_rule = getattr(self, "evolution_rule", None)
        fields = {
            "cooperation_metric": "both_players_per_joint_action_v1",
            "fitness_metric": (
                FITNESS_METRIC
                if isinstance(scenario, ReputationPrisonersDilemmaScenario)
                else "scenario_defined"
            ),
            "baseline_version": BASELINE_VERSION if self.use_baseline else None,
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
            F_CONFIG_FITNESS_WINDOW_FRACTION: self.fitness_window_fraction,
            F_CONFIG_OBSERVATION_SCHEDULE: self.observation_schedule,
            # Noise actually in force for this run. Read back by --resume-json,
            # so a continued lineage keeps the noise level it started under
            # instead of silently turning noise off mid-run.
            F_CONFIG_ACTION_ERROR: self.action_error_probability,
            F_CONFIG_OBSERVATION_ERROR: self.observation_error_probability,
            F_CONFIG_LLM_THINKING: self.llm_thinking,
            F_CONFIG_LLM_MAX_TOKENS: self._llm_max_tokens,
            F_CONFIG_LEARNING_METHOD: self.learning_method,
            F_CONFIG_FERMI_BETA: self.fermi_beta,
            F_CONFIG_MUTATION_RATE_ON_ADOPTION:
                self.mutation_rate_on_adoption,
            F_CONFIG_IMITATION_LEARNING_MODE:
                self.imitation_learning_mode,
            F_CONFIG_UPDATES_PER_GEN: self.updates_per_gen,
            F_CONFIG_LLM_CONCURRENCY: self.llm_concurrency,
            "mutation_temperature": self.mutation_temperature,
            F_CONFIG_INITIAL_REPUTATION: (
                FULL_INITIAL_REPUTATION
                if self.agent_type == "agent-type2"
                else INITIAL_REPUTATION
            ),
            F_CONFIG_FALLBACK_INIT_COUNT: self._fallback_init_count,
            F_CONFIG_FALLBACK_MUTATION_COUNT:
                self._fallback_mutation_count,
            "game_scenario": getattr(scenario, "name", type(scenario).__name__),
            "evolution_rule": (
                getattr(custom_rule, "name", type(custom_rule).__name__)
                if custom_rule is not None
                else self.learning_method
            ),
            # random.Random state contains only JSON-safe numbers/tuples.
            # json.dump writes tuples as arrays; _tuple_tree restores them.
            "rng_state": self.rng.getstate(),
            "rng_state_format": "python_random_v1",
        }
        fields.update(extra)
        return make_config(**fields)

    def _restore_from_evolution_log(self, previous: Dict) -> int:
        """Restore the evaluated final population and lineage bookkeeping.

        Returns the last recorded generation number. Reputations and other
        within-generation state are intentionally not restored: an evolution
        checkpoint lies at a generation boundary, where agents are rebuilt.
        """
        trajectory = previous.get("trajectory", [])
        final_population = previous.get("final_population", [])
        if not trajectory:
            raise ValueError("resume log has an empty trajectory")
        if len(final_population) != self.population_size:
            raise ValueError(
                "resume population size mismatch: "
                f"log has {len(final_population)}, configured {self.population_size}"
            )

        restored = []
        self._slot_lineage = {}
        self._slot_birth = {}
        for rec in final_population:
            aid = int(rec["agent_id"])
            agent = self._make_agent(rec["code"], aid)
            agent.fitness = float(rec.get("fitness", 0.0))
            restored.append(agent)
            lineage_id = rec.get("lineage_id")
            if lineage_id is not None:
                self._slot_lineage[aid] = int(lineage_id)
            self._slot_birth[aid] = lineage_event(
                lineage_id=lineage_id,
                parent_lineage_id=rec.get(F_PARENT_LINEAGE_ID),
                parent_id=rec.get(F_PARENT_ID),
                origin=rec.get(F_ORIGIN),
                birth_gen=rec.get(F_BIRTH_GEN),
            )

        self.agents = restored
        self._lineage_events = copy.deepcopy(previous.get("lineage_events", []))
        lineage_ids = [
            ev.get("lineage_id") for ev in self._lineage_events
            if ev.get("lineage_id") is not None
        ]
        self._next_lineage_id = max(lineage_ids, default=-1) + 1
        self._next_agent_id = max((a.agent_id for a in restored), default=-1) + 1
        return int(trajectory[-1]["generation"])

    def resume_evolution(
        self,
        previous: Dict,
        additional_generations: int,
        *,
        derived_rng_seed: Optional[int] = None,
        source_path: Optional[str] = None,
    ) -> Dict:
        """Append generations to an existing Fermi evolution log.

        The prior final generation is already evaluated, but its transition
        to the next generation was never performed. Consequently every
        appended generation begins with selection/reproduction using the
        previous generation's saved fitness, then evaluates the new cohort.
        """
        if additional_generations < 1:
            raise ValueError("additional_generations must be >= 1")
        if self.use_baseline or self.learning_method != "fermi":
            raise ValueError("resume currently supports non-baseline Fermi runs only")

        old_config = previous.get("config", {})
        if isinstance(self._get_game_scenario(), ReputationPrisonersDilemmaScenario):
            old_metric = old_config.get("fitness_metric")
            if old_metric != FITNESS_METRIC:
                raise ValueError(
                    "cannot resume a run with a different fitness metric: "
                    f"checkpoint has {old_metric or 'legacy cumulative payoff'}, "
                    f"current metric is {FITNESS_METRIC}; start a new run"
                )
        last_gen = self._restore_from_evolution_log(previous)
        old_fallback_init = int(old_config.get(F_CONFIG_FALLBACK_INIT_COUNT, 0))
        old_fallback_mutation = int(old_config.get(F_CONFIG_FALLBACK_MUTATION_COUNT, 0))
        self._fallback_init_count = old_fallback_init
        self._fallback_mutation_count = old_fallback_mutation

        saved_rng_state = old_config.get("rng_state")
        if saved_rng_state is not None:
            self.rng.setstate(self._tuple_tree(saved_rng_state))
            rng_mode = "checkpoint"
            effective_derived_seed = None
        else:
            if derived_rng_seed is None:
                # Stable across Python processes and versions; deliberately
                # does not use hash(), whose salt changes between processes.
                derived_rng_seed = (
                    int(self.seed) * 1_000_003
                    + int(last_gen + 1) * 97_409
                    + 0x5EED_C0DE
                ) & ((1 << 63) - 1)
            self.rng.seed(derived_rng_seed)
            rng_mode = "derived_branch"
            effective_derived_seed = int(derived_rng_seed)

        trajectory = copy.deepcopy(previous["trajectory"])
        total_generations = len(trajectory) + additional_generations
        self.num_generations = total_generations

        for gen in range(last_gen + 1, last_gen + 1 + additional_generations):
            # Complete the transition omitted after the old run's final gen.
            self._select_and_reproduce_fermi(next_gen=gen)
            self._round_offset = gen
            stats = self._run_one_generation()
            for i, agent in enumerate(self.agents):
                agent.fitness = stats["payoffs"][i] if i < len(stats["payoffs"]) else 0.0
            trajectory.append(trajectory_entry(
                generation=gen,
                cooperation_rate_mean=stats["cooperation_rate_mean"],
                n_interactions=stats["n_interactions"],
                fitness_mean=sum(stats["payoffs"]) / max(1, len(stats["payoffs"])),
                fitness_max=max(stats["payoffs"]) if stats["payoffs"] else 0.0,
                population=[self._agent_record(a) for a in self.agents],
            ))
            print(
                f"  Gen {gen}: coop={stats['cooperation_rate_mean']:.3f}, "
                f"fitness_mean={sum(stats['payoffs'])/max(1,len(stats['payoffs'])):.3f}"
            )

        resume_meta = {
            "source_path": source_path,
            "source_generations": len(previous["trajectory"]),
            "additional_generations": additional_generations,
            "rng_mode": rng_mode,
            "derived_rng_seed": effective_derived_seed,
            "uses_current_prompt": True,
        }
        return build_evolution_results(
            trajectory=trajectory,
            final_population=[self._agent_record(a) for a in self.agents],
            lineage_events=self._lineage_events,
            config=self._result_config(
                total_generations,
                resumed=True,
                resume=resume_meta,
            ),
        )

    def run_evolution(self, num_generations: int) -> Dict:
        """Run num_generations and return aggregate results."""
        # Keep the prompt templates in sync with the actual run length
        # (the constructor default may differ from the run-time value).
        self.num_generations = num_generations
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
                  f"fitness_mean={sum(stats['payoffs'])/max(1,len(stats['payoffs'])):.3f}")
            # Selection + mutation (only for LLM mode)
            if not self.use_baseline and gen < num_generations - 1:
                self._select_and_reproduce_by_method(next_gen=gen + 1)
        # Build final population
        final_population = [self._agent_record(a) for a in self.agents]
        # FALLBACK diagnostics (Fix E). Print init and mutation
        # FALLBACK ratios so reviewers can judge run reliability.
        # Init ratio >30% usually means the LLM init is broken
        # (model name, API key, thinking-mode mismatch); mutation
        # ratio >30% usually means the model is producing invalid
        # code at a high rate.
        init_ratio = self._fallback_init_count / max(1, self.population_size)
        if self.learning_method == "fermi":
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
            f"learning_method={self.learning_method} (beta={self.fermi_beta}, "
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
            config=self._result_config(num_generations),
        )

    def _population_snapshot(self) -> tuple[AgentSnapshot, ...]:
        """Freeze the planner-visible state before any concurrent work starts."""
        return tuple(
            AgentSnapshot(
                agent_id=agent.agent_id,
                code=agent.code,
                fitness=agent.fitness,
                lineage_id=self._slot_lineage.get(agent.agent_id),
            )
            for agent in self.agents
        )

    def _run_offspring_job(self, job: OffspringJob) -> OffspringResult:
        """Worker-side operation: generate and validate code, mutate no state."""
        if job.operator == "llm_init":
            assert job.preserve_agent_id is not None
            code = self._llm_init_code(job.preserve_agent_id)
        elif job.operator == "llm_mutate" and job.mutation_kind == "small":
            assert job.parent_code is not None
            assert job.parent_fitness is not None
            assert job.preserve_agent_id is not None
            code = self._llm_small_mutate_code(
                job.parent_code, job.parent_fitness, job.preserve_agent_id
            )
        elif job.operator == "llm_mutate" and job.mutation_kind == "full":
            assert job.parent_code is not None
            assert job.parent_fitness is not None
            # Preserve compatibility with callers that historically replaced
            # the instance-level ``_mutate`` hook in tests or experiments.
            if "_mutate" in self.__dict__:
                code = self._mutate(job.parent_code, job.parent_fitness)
            else:
                code = self._mutate_code(job.parent_code, job.parent_fitness)
        else:
            raise ValueError(f"unsupported offspring job: {job!r}")
        return OffspringResult(
            job=job,
            code=code,
            error_kind="generation_failed" if code is None else None,
        )

    def _fallback_code_for_job(self, job: OffspringJob) -> str:
        if job.operator == "llm_init":
            return (
                FALLBACK_CLASS_V3
                if self.agent_type == "agent-type2"
                else self.rng.choice(FALLBACK_STRATEGIES)
            )
        assert job.parent_code is not None
        return job.parent_code

    def _commit_generation_plan(
        self,
        plan: GenerationPlan,
        results: List[OffspringResult],
    ) -> None:
        """Commit a complete generation plan deterministically on one thread."""
        if plan.population_size != len(self.agents):
            raise ValueError("generation plan population size does not match")
        old_by_id = {agent.agent_id: agent for agent in self.agents}
        next_agents: List[Optional[object]] = [None] * plan.population_size

        for retained in plan.retained:
            snapshot = retained.snapshot
            try:
                agent = self._make_agent(snapshot.code, snapshot.agent_id)
            except Exception as exc:
                print(
                    "  [retained re-instantiate fallback] "
                    f"{type(exc).__name__}: {exc}"
                )
                agent = old_by_id[snapshot.agent_id]
            next_agents[retained.output_index] = agent

        result_by_ordinal = {result.job.ordinal: result for result in results}
        if len(result_by_ordinal) != len(plan.jobs):
            raise ValueError("offspring results are missing or have duplicate ordinals")

        lineage_updates = []
        for job in sorted(plan.jobs, key=lambda item: item.ordinal):
            result = result_by_ordinal.get(job.ordinal)
            if result is None:
                raise ValueError(f"missing offspring result for ordinal {job.ordinal}")
            code = result.code
            used_fallback = code is None
            if used_fallback:
                code = self._fallback_code_for_job(job)
                self._fallback_mutation_count += 1
                print(
                    f"  [{job.operator}] FALLBACK for output index="
                    f"{job.output_index}"
                )
            try:
                if job.preserve_agent_id is None:
                    agent = self._new_agent(code)
                else:
                    agent = self._make_agent(code, job.preserve_agent_id)
            except Exception as exc:
                if not used_fallback:
                    self._fallback_mutation_count += 1
                fallback_code = self._fallback_code_for_job(job)
                print(
                    "  [offspring instantiate fallback] "
                    f"{type(exc).__name__}: {exc}"
                )
                if job.preserve_agent_id is None:
                    agent = self._new_agent(fallback_code)
                else:
                    agent = self._make_agent(fallback_code, job.preserve_agent_id)
            next_agents[job.output_index] = agent
            lineage_updates.append((agent.agent_id, job))

        if any(agent is None for agent in next_agents):
            raise ValueError("generation plan did not fill every population slot")
        committed_agents = [agent for agent in next_agents if agent is not None]

        old_ids = set(old_by_id)
        new_ids = {agent.agent_id for agent in committed_agents}
        for agent in committed_agents:
            for removed_id in old_ids - new_ids:
                agent.reputations.pop(removed_id, None)

        # This is the generation transaction boundary.  No worker can observe
        # or partially modify the live population before this assignment.
        self.agents = committed_agents
        for agent_id, job in lineage_updates:
            self._new_lineage(
                agent_id,
                job.parent_id,
                job.parent_lineage_id,
                job.origin,
                job.birth_gen,
            )

    def _execute_generation_plan(self, plan: GenerationPlan) -> None:
        offspring_generator = getattr(self, "offspring_generator", None)
        if offspring_generator is None:
            results = self._parallel_llm_map(self._run_offspring_job, plan.jobs)
        else:
            results = list(offspring_generator.execute_batch(plan.jobs))
        self._commit_generation_plan(plan, results)

    def _select_and_reproduce(self, next_gen: Optional[int] = None):
        """Tournament + elite selection; replace num_eliminate worst with mutated
        copies of the survivors."""
        if next_gen is None:
            next_gen = 1
        rule = TournamentEvolutionRule(
            elite_count=self.elite_count,
            num_eliminate=self.num_eliminate,
            tournament_size=self.tournament_size,
        )
        plan = rule.plan(
            self._population_snapshot(),
            rng=self.rng,
            next_gen=next_gen,
            lineage_by_agent_id=self._slot_lineage,
        )
        self._execute_generation_plan(plan)

    def _select_and_reproduce_by_method(
        self, next_gen: Optional[int] = None
    ) -> None:
        """Dispatch a generation transition to the configured learning rule."""
        if self.evolution_rule is not None:
            if next_gen is None:
                next_gen = 1
            plan = self.evolution_rule.plan(
                self._population_snapshot(),
                rng=self.rng,
                next_gen=next_gen,
                lineage_by_agent_id=self._slot_lineage,
            )
            self._execute_generation_plan(plan)
        elif self.learning_method == "fermi":
            self._select_and_reproduce_fermi(next_gen=next_gen)
        elif self.learning_method == "tournament":
            self._select_and_reproduce(next_gen=next_gen)
        else:  # Constructor validation makes this a defensive guard.
            raise RuntimeError(f"unsupported learning method: {self.learning_method!r}")

    def _select_and_reproduce_fermi(self, next_gen: Optional[int] = None):
        """Synchronous Fermi imitation + LLM mutation (Moran-process style, Z-like).

        Per generation we sample `updates_per_gen` distinct learners without
        replacement. For each learner i we sample a role model j with i != j
        (role models may repeat; self-pairing is always forbidden) and apply

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
        replicator dynamics, run with learning_method="tournament"
        (tournament+elite path, no LLM in selection step).

        Sanity checks (should pass):
          * Fermi + ALLC, mu=0       -> stays at 1.0
          * Fermi + ALLD, mu=0       -> stays at 0.0
          * Fermi + 1 IS+ + 14 ALLD, mu=0 -> 14/1 (IS+ invades)
          * Fermi + 1 ALLD + 14 ALLC, mu=0 -> 15/0 (ALLD contained)
        """
        if next_gen is None:
            next_gen = 1
        rule = FermiEvolutionRule(
            beta=self.fermi_beta,
            mutation_rate=self.mutation_rate_on_adoption,
            updates_per_gen=self.updates_per_gen,
        )
        plan = rule.plan(
            self._population_snapshot(),
            rng=self.rng,
            next_gen=next_gen,
            lineage_by_agent_id=self._slot_lineage,
        )
        self._execute_generation_plan(plan)
