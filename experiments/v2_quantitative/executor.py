"""Minimal executor for the v2 quantitative interface (2-player PD version).

The v2 interface has two functions:
  evaluate(target_reputation, target_action, my_reputation) -> float
  decide(my_reputation, opponent_reputation) -> bool

This module compiles the agent's Python source and exposes those two
callables, with safety: only the two functions can be invoked, and the
return values are clamped.

The 3-arg `evaluate` is the new (PD) shape. Each call updates the
observer's view of a SINGLE target (the donor or the recipient in a
joint action). The framework calls evaluate() twice per observed
joint action (once for the donor, once for the recipient), using
each player's own action as `target_action`.
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
        if "evaluate" not in ns or "decide" not in ns:
            raise ValueError(
                "Strategy must define both evaluate(target_reputation, "
                "target_action, my_reputation) and decide(my_reputation, "
                "opponent_reputation)"
            )
        self._evaluate: Callable = ns["evaluate"]
        self._decide: Callable = ns["decide"]

    def evaluate(
        self,
        target_reputation: float,
        target_action: str,
        my_reputation: float,
    ) -> float:
        """Return observer's new rating of `target` after seeing target's action."""
        try:
            v = float(self._evaluate(
                float(target_reputation),
                str(target_action),
                float(my_reputation),
            ))
        except Exception:
            v = float(target_reputation)
        if v != v:  # NaN
            v = 0.0
        return max(-1.0, min(1.0, v))

    def decide(self, my_reputation: float, opponent_reputation: float) -> bool:
        try:
            v = self._decide(float(my_reputation), float(opponent_reputation))
        except Exception:
            v = False
        return bool(v)
