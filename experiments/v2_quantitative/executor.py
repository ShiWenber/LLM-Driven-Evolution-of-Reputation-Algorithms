"""Minimal executor for the v2 quantitative interface (2-player PD version).

The v2 interface has two functions:
    observe(donor_reputation, donor_action,
                    recipient_reputation, recipient_action,
                    my_reputation) -> tuple[float, float] | dict
  decide(my_reputation, opponent_reputation) -> bool

This module compiles the agent's Python source and exposes those two
callables, with safety: only the two functions can be invoked, and
returned reputations are clamped.

The 5-arg `observe` is the PD shape. Each call updates the observer's
view of BOTH players in one joint action.
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
                "Strategy must define both observe(donor_reputation, "
                "donor_action, recipient_reputation, recipient_action, "
                "my_reputation) and decide(my_reputation, "
                "opponent_reputation)"
            )
        self._observe: Callable = ns["observe"]
        self._decide: Callable = ns["decide"]

    def observe(
        self,
        donor_reputation: float,
        donor_action: str,
        recipient_reputation: float,
        recipient_action: str,
        my_reputation: float,
    ) -> tuple[float, float]:
        """Return observer's new ratings of (donor, recipient)."""
        try:
            out = self._observe(
                float(donor_reputation),
                str(donor_action),
                float(recipient_reputation),
                str(recipient_action),
                float(my_reputation),
            )
        except Exception:
            out = (float(donor_reputation), float(recipient_reputation))

        donor_v = float(donor_reputation)
        recipient_v = float(recipient_reputation)
        if isinstance(out, dict):
            donor_v = float(out.get("donor_reputation", donor_v))
            recipient_v = float(out.get("recipient_reputation", recipient_v))
        elif isinstance(out, (tuple, list)) and len(out) >= 2:
            donor_v = float(out[0])
            recipient_v = float(out[1])

        if donor_v != donor_v:  # NaN
            donor_v = 0.0
        if recipient_v != recipient_v:  # NaN
            recipient_v = 0.0
        donor_v = max(-1.0, min(1.0, donor_v))
        recipient_v = max(-1.0, min(1.0, recipient_v))
        return donor_v, recipient_v

    def decide(self, my_reputation: float, opponent_reputation: float) -> bool:
        try:
            v = self._decide(float(my_reputation), float(opponent_reputation))
        except Exception:
            v = False
        return bool(v)
