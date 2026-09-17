"""Shared fixed-strategy invasion primitives.

Everything in the invasion experiment is a two-type mixture: one CANDIDATE
strategy against one RESIDENT. A resident is either a canonical norm or another
arbitrary strategy, and both are represented the same way -- as an
``EvolvedSource``. A norm is simply the source whose code is ``BASELINES[norm]``
(see ``norm_source``), so competitor construction and imitation have exactly one
code path regardless of what the two sides are.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.v2_quantitative.agent import QuantitativeAgent
from experiments.v2_quantitative.agent_full import FullAgent, V3StrategyExecutor
from experiments.v2_quantitative.agent_signal import SignalAgent
from experiments.v2_quantitative.signal_executor import SignalStrategyExecutor
from experiments.v2_quantitative.baselines import BASELINES, LEADING_EIGHT
from experiments.v2_quantitative.executor import V2StrategyExecutor
from experiments.v2_quantitative.game import (
    prisoners_dilemma_payoff,
    resolve_fitness_window,
)


AGENT_TYPES = ("agent-type1", "agent-type2", "agent-type2-signal")
NORMS = (*LEADING_EIGHT, "ALLC", "ALLD")

# The invasion experiment measures a single frequency axis: the candidate
# strategy's share ``x``. A count of ``k`` places the candidate at ``x = k/100``,
# so sweeping ``k = 1..99`` already covers every composition. Whether the
# candidate is itself invadable is a separate question, measured directly by the
# fixation benchmark rather than inferred from this sweep.
#
# Archived sweeps were written before that simplification, when results were
# nested under a direction level (``<label>/<direction>/<norm>/...``). Only this
# one label was ever produced, so it is retained purely so those archived
# results stay readable; nothing new is written under it.
ARCHIVED_DIRECTION_LABEL = "evolved_invades_norm"

# Default number of joint actions (pairs) played per generation by the sweep.
# CHANGED 2026-09-13: 1_000 -> 10_000. This is only the DEFAULT for
# ``run_invasion --interactions``. Every result records the value it was
# produced with under ``config.interactions_per_generation``, and
# ``cache_matches`` compares against that recorded value, so raising the
# default does not reinterpret archived runs -- it only (a) changes what a
# new run does when the flag is omitted, and (b) makes the cache correctly
# treat the archived 1000-interaction runs as stale.
#
# With N=100 each generation pairs all 100 agents into 50 pairs per pass, so
# 10_000 interactions = 200 passes = 200 pairings per agent -- 10x the 1000
# interactions the 2026-09 archived sweeps used.
INTERACTIONS_PER_GENERATION = 10_000
NUM_GENERATIONS = 50

# The payoff matrix and the fitness window are both taken from the main
# evolutionary engine (``experiments.v2_quantitative.game``) rather than
# redefined here.
#
# BENEFIT / COST are the FALLBACK payoff matrix, used only when nothing in the
# run says otherwise. A strategy loaded from an ``evolutionary.json`` carries
# the matrix it was actually selected under (``config.benefit`` / ``config.cost``)
# on its ``EvolvedSource``, and ``resolve_payoff_matrix`` prefers that: testing a
# b=3 candidate with a b=2 matrix silently changes the game the mutant was
# adapted to. These two values remain the defaults because every archived result
# was produced with them, so a run whose sources carry no log still reproduces
# the archive.
BENEFIT = 2.0
COST = 1.0
FITNESS_WINDOW_FRACTION = 0.2


def resolve_window(
    interactions: int,
    population_size: int,
    fraction: float = FITNESS_WINDOW_FRACTION,
) -> int:
    """Trailing interactions of a generation that count toward fitness.

    Delegates to the main engine's ``resolve_fitness_window`` so the burn-in
    semantics are defined in exactly one place; returns ``interactions`` when
    the window is disabled or covers the whole generation.
    """
    window = resolve_fitness_window(
        fraction, interactions, max(1, population_size // 2)
    )
    return interactions if window is None else window


# The two roles in a mixture. The candidate is the strategy under test; the
# resident is what it is pitted against. KIND_NORM / KIND_RESIDENT distinguish a
# canonical-norm resident from an arbitrary-strategy resident for reporting only;
# neither affects behaviour.
KIND_CANDIDATE = "candidate"
KIND_NORM = "norm"
KIND_RESIDENT = "resident"


@dataclass(frozen=True)
class EvolvedSource:
    agent_type: str
    path: Path
    agent_id: int
    lineage_id: int
    root_lineage_id: int
    root_family_size: int
    fitness: float
    code: str
    code_sha256: str
    # Payoff matrix this strategy was selected under, read from the run log.
    # ``None`` means "no log says", which is the case for canonical norms and
    # hand-written ``.py`` strategies; ``resolve_payoff_matrix`` then falls back
    # to the archived protocol constant.
    benefit: float | None = None
    cost: float | None = None


def resolve_payoff_matrix(
    *sources: EvolvedSource,
    benefit: float | None = None,
    cost: float | None = None,
) -> tuple[float, float]:
    """Pick the ``(benefit, cost)`` a two-type mixture is played under.

    The payoff matrix is a property of the population, not of either strategy:
    both members of every pair are paid from the same table, so one pair of
    values has to cover all of them. Each field is resolved independently, in
    this order:

    1. The explicit ``benefit`` / ``cost`` argument (the ``--benefit`` /
       ``--cost`` CLI override).
    2. The value recorded in the log the source was loaded from. Sources with
       no log (canonical norms, hand-written ``.py``) are skipped.
    3. ``BENEFIT`` / ``COST``, the constant every archived result was produced
       with.

    Two log-derived sources that disagree on a field make the mixture
    ill-defined -- they were selected under different games, so no single table
    realises both -- and that raises rather than silently picking one.
    Overriding the field explicitly suppresses that error, which is how a
    deliberate cross-benefit comparison is run.
    """
    logged_benefits = {s.benefit for s in sources if s.benefit is not None}
    logged_costs = {s.cost for s in sources if s.cost is not None}

    def pick(
        explicit: float | None,
        logged: set[float],
        default: float,
        field: str,
    ) -> float:
        if explicit is not None:
            return float(explicit)
        if len(logged) > 1:
            origins = "; ".join(
                f"{s.path}: {field}={getattr(s, field)}" for s in sources
                if getattr(s, field) is not None
            )
            raise ValueError(
                f"sources were selected under different payoff matrices, so no "
                f"single matrix covers the mixture ({field} disagrees): {origins}. "
                f"Pass --{field} to force one explicitly."
            )
        if logged:
            return float(next(iter(logged)))
        return float(default)

    return (
        pick(benefit, logged_benefits, BENEFIT, "benefit"),
        pick(cost, logged_costs, COST, "cost"),
    )


def norm_source(norm: str) -> EvolvedSource:
    """A canonical norm expressed as an ordinary strategy source.

    Norms used to be a separate competitor kind, which forced competitor
    construction and the imitation copier to branch. Representing a norm as a
    source removes that branch and lets one implementation serve norm residents
    and strategy residents alike.
    """
    if norm not in NORMS:
        raise ValueError(f"Unknown norm: {norm}")
    code = BASELINES[norm]
    return EvolvedSource(
        agent_type="agent-type1",
        path=Path(f"<baseline:{norm}>"),
        agent_id=-1,
        lineage_id=-1,
        root_lineage_id=-1,
        root_family_size=0,
        fitness=0.0,
        code=code,
        code_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
    )


def root_lineage(lineage_id: int, parent_by_lineage: dict[int, int | None]) -> int:
    root = lineage_id
    seen: set[int] = set()
    while parent_by_lineage.get(root) is not None:
        if root in seen:
            raise ValueError(f"Cycle in lineage graph at {root}")
        seen.add(root)
        root = int(parent_by_lineage[root])
    return root


# ---------------------------------------------------------------------------
# Judgment memoisation
# ---------------------------------------------------------------------------
# ``play_generation_noisy`` makes every one of the ``N`` agents judge every
# interaction, so a single generation performs ``2 * N * interactions``
# strategy judgments -- 100 million per run at N=100, 10_000 interactions and
# 50 generations. That cost dominates the sweep, yet almost all of it is
# redundant. A judgment is a pure function of the judging strategy's source and
# the five floats/strings passed to it, and the observers' reputation states
# are strongly correlated because they all observed the same history. In the
# N=100 protocol a single (reputation, action) combination accounts for more
# than half of every judgment made.
#
# The memo is keyed on the strategy's code digest, so it is shared by every
# slot running that strategy and never leaks between two different strategies.
# It is installed only for strategies whose source is free of nondeterministic
# calls: memoising a strategy that consults ``random`` (or the clock, or
# ``id()``) would change its behaviour, which is exactly what the guard below
# prevents.
NONDETERMINISTIC_TOKENS = (
    "random", "time.", "datetime", "uuid", "os.", "sys.", "id(", "hash(",
    "getrandbits", "urandom", "perf_counter", "monotonic",
)

# Bounded so a long-lived worker cannot accumulate a cache for every strategy it
# has ever seen. A sweep holds only two or three codes at a time. The cap is
# deliberately small: measured over a full 50-generation run it costs 0.2% of
# the hit rate (66.8% vs 67.0% served from memo) while using 2.7x less memory
# per worker (39 MB vs 105 MB), which matters because the sweep runs 48 of them.
JUDGE_MEMO_MAX_CODES = 8
JUDGE_MEMO_MAX_ENTRIES = 50_000
_JUDGE_MEMO: dict[str, dict[tuple[Any, ...], float]] = {}
_JUDGE_MEMO_MISS = object()


def strategy_is_deterministic(code: str) -> bool:
    """True when the source never consults a nondeterministic source."""
    return not any(token in code for token in NONDETERMINISTIC_TOKENS)


def _install_judge_memo(agent: Any, source: EvolvedSource) -> None:
    """Memoise ``agent``'s judgments on its strategy's code digest.

    A no-op for agent types whose executor does not expose the one-directional
    five-argument ``observe`` this key is built from.
    """
    executor = getattr(agent, "_executor", None)
    if not isinstance(executor, V2StrategyExecutor):
        return
    if not strategy_is_deterministic(source.code):
        return
    cache = _JUDGE_MEMO.get(source.code_sha256)
    if cache is None:
        if len(_JUDGE_MEMO) >= JUDGE_MEMO_MAX_CODES:
            _JUDGE_MEMO.pop(next(iter(_JUDGE_MEMO)))
        cache = _JUDGE_MEMO[source.code_sha256] = {}
    original = executor.observe

    def memoised_observe(
        A_rep: float,
        A_action: str,
        B_rep: float,
        B_action: str,
        my_rep: float,
    ) -> float:
        key = (A_rep, A_action, B_rep, B_action, my_rep)
        cached = cache.get(key, _JUDGE_MEMO_MISS)
        if cached is not _JUDGE_MEMO_MISS:
            return cached  # type: ignore[return-value]
        value = original(A_rep, A_action, B_rep, B_action, my_rep)
        if len(cache) < JUDGE_MEMO_MAX_ENTRIES:
            cache[key] = value
        return value

    executor.observe = memoised_observe


def clear_judge_memo() -> None:
    """Drop every memoised judgment table."""
    _JUDGE_MEMO.clear()


@dataclass
class Competitor:
    agent_id: int
    kind: str
    label: str
    source: EvolvedSource
    agent: Any

    @classmethod
    def create(
        cls, agent_id: int, kind: str, label: str, source: EvolvedSource,
        *, signal_seed: int | None = None,
    ) -> Competitor:
        if source.agent_type == "agent-type1":
            agent = QuantitativeAgent(
                agent_id, source.code, executor=V2StrategyExecutor(source.code)
            )
        elif source.agent_type == "agent-type2":
            agent = FullAgent(
                agent_id, V3StrategyExecutor(source.code), code=source.code
            )
        elif source.agent_type == "agent-type2-signal":
            agent = SignalAgent(
                agent_id, SignalStrategyExecutor(
                    source.code,
                    seed=random.getrandbits(128) if signal_seed is None else signal_seed,
                ), code=source.code
            )
        else:
            raise ValueError(f"Unknown source agent type: {source.agent_type}")
        _install_judge_memo(agent, source)
        return cls(agent_id, kind, label, source, agent)

    @property
    def fitness(self) -> float:
        return float(self.agent.fitness)

    @fitness.setter
    def fitness(self, value: float) -> None:
        self.agent.fitness = float(value)

    def reset_generation_tracking(self) -> None:
        self.agent.reset_for_generation()
        self.fitness = 0.0

    def choose(self, opponent_id: int) -> bool:
        return bool(self.agent.choose(opponent_id))

    def observe(
        self, actor_id: int, actor_action: str,
        recipient_id: int, recipient_action: str,
    ) -> None:
        self.agent.observe_and_judge(
            actor_id, actor_action, recipient_id, recipient_action
        )


def payoff_imitation_update(
    population: list[Competitor], rng: Any, updates: int,
) -> list[Competitor]:
    """Synchronously copy a sampled role model only when strictly fitter.

    The learner adopts the MODEL's whole identity -- kind, label and source.
    Copying only the kind is correct when every member shares one source, but it
    silently mismatches a slot's declared type with the code it runs as soon as
    the population holds two different strategies.
    """
    old = population
    next_population = list(old)
    size = len(old)
    for _ in range(updates):
        learner_pos = rng.randrange(size)
        model_pos = rng.randrange(size - 1)
        if model_pos >= learner_pos:
            model_pos += 1
        learner, model = old[learner_pos], old[model_pos]
        if model.fitness <= learner.fitness or learner.kind == model.kind:
            continue
        next_population[learner_pos] = Competitor.create(
            learner.agent_id, model.kind, model.label, model.source
        )
    return [
        member if member is not old[pos] else Competitor.create(
            member.agent_id, member.kind, member.label, member.source
        )
        for pos, member in enumerate(next_population)
    ]


def _flip_action(action: str, rng: Any, probability: float) -> str:
    if probability <= 0.0:
        return action
    if rng.random() >= probability:
        return action
    return "defect" if action == "cooperate" else "cooperate"


def play_generation_noisy(
    population: list[Competitor],
    rng: Any,
    interactions: int,
    fitness_window_fraction: float = FITNESS_WINDOW_FRACTION,
    action_error: float = 0.0,
    observation_error: float = 0.0,
    benefit: float = BENEFIT,
    cost: float = COST,
) -> dict[str, Any]:
    """Play one generation with independent action and perception flips.

    ``fitness_window_fraction`` selects the trailing share of interactions that
    counts toward fitness (see ``resolve_window``); earlier ones are burn-in
    that still updates reputations.

    ``benefit`` / ``cost`` are the payoff matrix the generation is paid from.
    Callers pass the values ``resolve_payoff_matrix`` picked, so a candidate is
    evaluated in the game it was selected under rather than a fixed one; the
    defaults keep the archived protocol reproduces when nothing overrides them.
    """
    for agent in population:
        agent.reset_generation_tracking()
    window_start = interactions - resolve_window(
        interactions, len(population), fitness_window_fraction
    )
    payoffs = [0.0] * len(population)
    counted = [0.0] * len(population)
    completed = 0
    cooperation_count = 0
    # Perception noise is applied once per (observer, actor) INSIDE the observer
    # loop, so the loop draws 2 * N random numbers per interaction -- the single
    # largest RNG cost in the sweep. Hoisting the zero check out of the loop and
    # inlining the flip keeps the draw order identical while removing four
    # million function calls per run. The zero case must still skip the draws
    # entirely, exactly as _flip_action does.
    observes_with_noise = observation_error > 0.0
    while completed < interactions:
        order = list(range(len(population)))
        rng.shuffle(order)
        for offset in range(0, len(order) - 1, 2):
            if completed >= interactions:
                break
            first_pos, second_pos = order[offset], order[offset + 1]
            first, second = population[first_pos], population[second_pos]
            first_intended = "cooperate" if first.choose(second.agent_id) else "defect"
            second_intended = "cooperate" if second.choose(first.agent_id) else "defect"
            first_action = _flip_action(first_intended, rng, action_error)
            second_action = _flip_action(second_intended, rng, action_error)
            first_cooperates = first_action == "cooperate"
            second_cooperates = second_action == "cooperate"
            # Payoffs come from the shared matrix definition in the main
            # engine, evaluated at the matrix this mixture resolved to.
            first_payoff = prisoners_dilemma_payoff(
                my_cooperates=first_cooperates,
                other_cooperates=second_cooperates,
                benefit=benefit,
                cost=cost,
            )
            second_payoff = prisoners_dilemma_payoff(
                my_cooperates=second_cooperates,
                other_cooperates=first_cooperates,
                benefit=benefit,
                cost=cost,
            )
            payoffs[first_pos] += first_payoff
            payoffs[second_pos] += second_payoff
            if completed >= window_start:
                counted[first_pos] += first_payoff
                counted[second_pos] += second_payoff

            second_id = second.agent_id
            first_id = first.agent_id
            if observes_with_noise:
                for observer in population:
                    seen_first = first_action
                    if rng.random() < observation_error:
                        seen_first = (
                            "defect" if seen_first == "cooperate" else "cooperate"
                        )
                    seen_second = second_action
                    if rng.random() < observation_error:
                        seen_second = (
                            "defect" if seen_second == "cooperate" else "cooperate"
                        )
                    if observer.agent_id == second_id:
                        observer.observe(second_id, seen_second, first_id, seen_first)
                    else:
                        observer.observe(first_id, seen_first, second_id, seen_second)
            else:
                for observer in population:
                    if observer.agent_id == second_id:
                        observer.observe(
                            second_id, second_action, first_id, first_action
                        )
                    else:
                        observer.observe(
                            first_id, first_action, second_id, second_action
                        )
            cooperation_count += int(first_cooperates) + int(second_cooperates)
            completed += 1
    for pos, agent in enumerate(population):
        agent.fitness = counted[pos]
    return {
        "cooperation_rate": cooperation_count / (2 * completed),
        "fitness": counted,
        "all_interaction_payoffs": payoffs,
    }


# Historical private name, still imported by the legacy runners.
_play_generation_noisy = play_generation_noisy


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
