"""Minimal executor for the v2 quantitative interface (2-player PD version).

The v2 interface has two functions:
    observe(A_rep, A_action, B_rep, B_action, my_rep) -> float
  decide(my_reputation, opponent_reputation) -> bool

This module compiles the agent's Python source and exposes those two
callables, with safety: only the two functions can be invoked, and
returned reputations are clamped.

The 5-arg `observe` is the PD shape but ONE-DIRECTIONAL: it judges a
single target player (A) and returns only that player's new reputation.
The framework calls observe() TWICE per joint action — once with
(A=donor, B=recipient) and once with (A=recipient, B=donor) — so the
strategy code never has to repeat a symmetric update for both players.
"""
from __future__ import annotations
import math
from typing import Callable


class V2StrategyExecutor:
    def __init__(self, code: str):
        self.code = code
        self._compile(code)

    def _compile(self, code: str):
        ns: dict = {"__builtins__": __builtins__}
        # Pre-import common safe libs as None so the agent code can
        # `import math` or `import random` if it wants.
        ns["math"] = math
        try:
            exec(self.code, ns)
        except Exception as e:
            raise ValueError(f"Failed to compile strategy code: {e}")
        if "observe" not in ns or "decide" not in ns:
            raise ValueError(
                "Strategy must define both observe(A_rep, A_action, B_rep, "
                "B_action, my_reputation) and decide(my_reputation, "
                "opponent_reputation)"
            )
        self._observe: Callable = ns["observe"]
        self._decide: Callable = ns["decide"]

    def observe(
        self,
        A_rep: float,
        A_action: str,
        B_rep: float,
        B_action: str,
        my_rep: float,
    ) -> float:
        """Return observer's new rating of player A from one joint action.

        One-directional: only A (the first player) is judged. The
        framework calls this twice per joint action with the roles
        swapped to update both players' reputations.
        """
        try:
            out = self._observe(
                float(A_rep),
                str(A_action),
                float(B_rep),
                str(B_action),
                float(my_rep),
            )
        except Exception:
            return float(A_rep)

        try:
            v = float(out)
        except (TypeError, ValueError):
            return float(A_rep)
        if v != v:  # NaN
            v = 0.0
        return max(-1.0, min(1.0, v))

    def decide(self, my_reputation: float, opponent_reputation: float) -> bool:
        try:
            v = self._decide(float(my_reputation), float(opponent_reputation))
        except Exception:
            v = False
        return bool(v)
