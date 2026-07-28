"""Minimal executor for the v2 quantitative interface.

The v2 interface has two functions:
  evaluate(donor_reputation, recipient_reputation, donor_action, recipient_action, my_reputation) -> float
  decide(my_reputation, opponent_reputation) -> bool

This module compiles the agent's Python source and exposes those two
callables, with safety: only the two functions can be invoked, and the
return values are clamped.
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
            raise ValueError("Strategy must define both evaluate() and decide()")
        self._evaluate: Callable = ns["evaluate"]
        self._decide: Callable = ns["decide"]

    def evaluate(
        self,
        donor_reputation: float,
        recipient_reputation: float,
        donor_action: str,
        recipient_action: str,
        my_reputation: float,
    ) -> float:
        try:
            v = float(self._evaluate(
                float(donor_reputation),
                float(recipient_reputation),
                str(donor_action),
                str(recipient_action),
                float(my_reputation),
            ))
        except Exception:
            v = float(donor_reputation)
        if v != v:  # NaN
            v = 0.0
        return max(-1.0, min(1.0, v))

    def decide(self, my_reputation: float, opponent_reputation: float) -> bool:
        try:
            v = self._decide(float(my_reputation), float(opponent_reputation))
        except Exception:
            v = False
        return bool(v)
