"""v2 quantitative interface — agent type 2: full Python class.

Type 1 (QuantitativeAgent in agent.py) constrains the LLM to two functions
(observe + decide) and forces it to use a private scalar `reputations`
matrix as the only state. This is fine for studying quantitative-assessment
strategies, but it caps what the LLM can express: every strategy is a
function `(reputation, action) -> reputation` plus a threshold rule.

Type 2 (this file) lets the LLM emit a *full Python class* with
arbitrary state — arbitrary attributes, arbitrary data structures, any
internal logic. The class only needs three methods:

    class LLMAgent:
        def __init__(self, agent_id: int) -> None:
            self.agent_id = agent_id
            # ... any state the LLM wants ...

        def decide(self) -> bool:
            # Returns True to cooperate, False to defect.
            # The framework sets self._ctx_opponent_id on the instance
            # BEFORE calling this method, so the LLM can read it.
            ...

        def observe(
            self,
            donor_id: int,
            donor_action: str,    # 'cooperate' or 'defect'
            recipient_id: int,
            recipient_action: str, # 'cooperate' or 'defect'
        ) -> None:
            # Called for every joint action the agent witnesses (whether
            # self-judgment or third-party observation). The LLM can
            # detect self-judgment by checking `donor_id == self.agent_id`
            # or `recipient_id == self.agent_id`.
            ...

The framework additionally guarantees:
  * `self._ctx_opponent_id: int` is set on the instance right before
    each `decide()` call.
  * `observe()` is called for BOTH self-judgments and third-party
    observations; the LLM is responsible for filtering if needed.
  * When the population turns over, dead agents' instances are dropped
    along with their state — there is no "zombie" cleanup. The LLM
    never needs to handle this.

The class name MUST be `LLMAgent` (so we can locate it generically in
LLM-generated code). The executor enforces this.

The fallback classes (ALLC / ALLD) live in this file too; the
population manager picks one of them when the LLM fails to produce
valid code.
"""
from __future__ import annotations
import math
from typing import Optional


INITIAL_REPUTATION = 0.1  # default for unseen (incl. self at start)


# -----------------------------------------------------------------------
# Fallback classes — used when the LLM fails to produce valid code.
# These are intentionally trivial: just cooperate or just defect.
# -----------------------------------------------------------------------
class LLMAgent:
    """Fallback base. Real fallback subclasses override `decide`."""

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        # Set by the framework before decide() is called.
        self._ctx_opponent_id: Optional[int] = None

    def decide(self) -> bool:  # pragma: no cover
        raise NotImplementedError

    def observe(
        self,
        donor_id: int,
        donor_action: str,
        recipient_id: int,
        recipient_action: str,
    ) -> None:
        return None


class ALLCClass(LLMAgent):
    """Always cooperate. Fallback for when LLM code is broken."""

    def decide(self) -> bool:
        return True


class ALLDClass(LLMAgent):
    """Always defect. Fallback for when LLM code is broken."""

    def decide(self) -> bool:
        return False


# Source strings for the fallback classes (V3StrategyExecutor looks for
# a class literally named `LLMAgent`, so we wrap the trivial behavior
# inside that class name rather than reusing ALLCClass/ALLDClass).
ALLC_CLASS_SOURCE = '''
class LLMAgent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self._ctx_opponent_id = None

    def decide(self) -> bool:
        return True

    def observe(self, donor_id, donor_action, recipient_id, recipient_action) -> None:
        return None
'''

ALLD_CLASS_SOURCE = '''
class LLMAgent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self._ctx_opponent_id = None

    def decide(self) -> bool:
        return False

    def observe(self, donor_id, donor_action, recipient_id, recipient_action) -> None:
        return None
'''


# -----------------------------------------------------------------------
# V3 executor — loads a `LLMAgent` class from arbitrary Python code.
# -----------------------------------------------------------------------
class V3StrategyExecutor:
    """Compiles LLM-generated Python source that defines a class named
    `LLMAgent`. Validates the required surface (`__init__(agent_id)`,
    `decide()`, `observe(...)`) and exposes an `instantiate(agent_id)`
    method that returns a fresh instance.
    """

    def __init__(self, code: str):
        self.code = code
        self._compile(code)

    def _compile(self, code: str):
        # Safe-ish execution: pre-import common libs as None so the LLM
        # can `import math` / `import random` if it wants. Builtins
        # remain available.
        ns: dict = {
            "__builtins__": __builtins__,
            "math": math,
        }
        try:
            exec(self.code, ns)
        except Exception as e:
            raise ValueError(f"Failed to compile agent class code: {e}")
        # The class must be named LLMAgent
        if "LLMAgent" not in ns:
            raise ValueError(
                "Agent class must define a class named `LLMAgent`"
            )
        cls = ns["LLMAgent"]
        if not isinstance(cls, type):
            raise ValueError("`LLMAgent` must be a class")
        # Required methods
        for method in ("__init__", "decide", "observe"):
            if not hasattr(cls, method):
                raise ValueError(
                    f"`LLMAgent` must define a `{method}` method"
                )
        # Smoke-test instantiate with id=0. Catches classes that compile
        # but fail at __init__ (e.g., a class without its own
        # `__init__(self, agent_id)` inherits object.__init__ which
        # rejects the extra arg). Without this, the bad class slips
        # past _validate_code and crashes the run when FullAgent tries
        # to instantiate it for the assigned agent_id.
        try:
            _smoke = cls(0)
            del _smoke
        except Exception as e:
            raise ValueError(
                f"`LLMAgent` smoke instantiate failed: {e}"
            )
        self._cls = cls

    def instantiate(self, agent_id: int) -> "LLMAgent":
        """Create a fresh instance of the LLM-defined class for a
        specific agent. Each call returns a brand-new object with its
        own state."""
        try:
            return self._cls(agent_id)
        except Exception as e:
            raise ValueError(
                f"LLMAgent({agent_id}) failed at __init__: {e}"
            )


# -----------------------------------------------------------------------
# FullAgent — type1-compatible wrapper around an LLM class instance.
#
# Implements the SAME interface as QuantitativeAgent
# (choose / observe_and_judge / self_judge / record_donation / ...),
# but delegates decide/observe to a `LLMAgent` class instance instead of
# two top-level functions. The framework can therefore treat type1 and
# type2 agents uniformly.
# -----------------------------------------------------------------------
class FullAgent:
    """A type1-compatible agent whose brain is a `LLMAgent` instance.

    The framework still maintains the `reputations` dict on this
    wrapper for bookkeeping / fitness / observability — but the
    `LLMAgent` class is not required to read or write it. The
    `LLMAgent` decides purely from its own state plus
    `self._ctx_opponent_id` (which we set before each `decide()` call).
    """

    def __init__(self, agent_id: int, executor: V3StrategyExecutor, code: str = ""):
        self.agent_id = agent_id
        self._executor = executor
        # Keep the source string for logging / debugging / replay
        self.code = code
        # Fresh per-instance brain
        self.brain = executor.instantiate(agent_id)
        # Same bookkeeping as type1
        self.reputations: dict = {agent_id: INITIAL_REPUTATION}
        self.fitness: float = 0.0
        self.total_donations: int = 0
        self.total_decisions: int = 0
        self.cooperations: int = 0

    # --- Reputation accessors (same as type1) ---------------------------
    @property
    def cooperation_rate(self) -> float:
        return (self.cooperations / self.total_decisions) if self.total_decisions else 0.0

    def get_reputation(self, other_id: int) -> float:
        return self.reputations.get(other_id, INITIAL_REPUTATION)

    def get_self_reputation(self) -> float:
        return self.reputations.get(self.agent_id, INITIAL_REPUTATION)

    def update_reputation(self, other_id: int, new_rep: float):
        new_rep = max(-1.0, min(1.0, new_rep))
        self.reputations[other_id] = new_rep

    # --- Framework-driven actions ---------------------------------------
    def choose(self, opponent_id: int, round_num: int = 0) -> bool:
        """Set the opponent context, then ask the brain to decide.

        Some LLM-generated classes define `_ctx_opponent_id` as a
        @property WITHOUT a setter; in that case direct attribute
        assignment raises AttributeError. We catch it and bypass the
        property by injecting directly into the instance __dict__
        (or, if the property has no underlying storage at all, we
        fall back to a None opponent_id and let decide() cope).
        """
        try:
            self.brain._ctx_opponent_id = opponent_id
        except AttributeError as e:
            # Property without setter. Try to bypass by going through
            # __dict__ — works iff there's a backing field. If the
            # property fully owns the attribute, this also fails and
            # we just give up on setting it; decide() will see the
            # property's default value.
            try:
                # Walk the MRO: if any class has _ctx_opponent_id in
                # __dict__ (the storage), we can write to it there.
                found = False
                for cls in type(self.brain).__mro__:
                    if "_ctx_opponent_id" in cls.__dict__:
                        # The storage is the descriptor; if it's a
                        # property, this won't help. But if it's a
                        # plain attribute (LLM put it in __init__),
                        # we can bypass the property by writing
                        # directly to instance __dict__.
                        descr = cls.__dict__["_ctx_opponent_id"]
                        if not isinstance(descr, property):
                            self.brain.__dict__["_ctx_opponent_id"] = opponent_id
                            found = True
                            break
                if not found:
                    # Pure property with no setter; decide() will use
                    # whatever the property returns (likely None or
                    # a stored value). Log once per LLM-emitted class
                    # so the run isn't spammed.
                    cls = type(self.brain)
                    if not getattr(cls, "_ctx_opponent_id_logged", False):
                        cls._ctx_opponent_id_logged = True
                        print(
                            f"  [choose] brain class {cls.__name__} "
                            f"defines _ctx_opponent_id as a property "
                            f"without a setter; opponent_id={opponent_id} "
                            f"will be IGNORED for this class. "
                            f"({type(e).__name__}: {e})"
                        )
            except Exception:
                # Last resort: silently ignore.
                pass
        try:
            return bool(self.brain.decide())
        except Exception:
            return False

    def observe_and_judge(
        self,
        donor_id: int,
        donor_action: str,
        recipient_id: int,
        recipient_action: str,
    ):
        """Forward the joint action to the brain's `observe`. The LLM
        class decides internally how to update its state (if at all)."""
        try:
            self.brain.observe(
                donor_id=donor_id,
                donor_action=donor_action,
                recipient_id=recipient_id,
                recipient_action=recipient_action,
            )
        except Exception:
            pass

    def self_judge(
        self,
        donor_action: str,
        recipient_id: int,
        recipient_action: str,
    ):
        self.observe_and_judge(
            donor_id=self.agent_id,
            donor_action=donor_action,
            recipient_id=recipient_id,
            recipient_action=recipient_action,
        )

    # --- Generation tracking (same as type1) ----------------------------
    def reset_for_generation(self):
        self.total_donations = 0
        self.total_decisions = 0
        self.cooperations = 0

    def record_donation(self, partner_id: int, donated: bool, round_num: int):
        self.total_decisions += 1
        if donated:
            self.cooperations += 1

    def handle_agents_replaced(self, old_ids, new_ids):
        for old in old_ids:
            self.reputations.pop(old, None)
