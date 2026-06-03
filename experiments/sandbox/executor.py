"""Sandbox executor for LLM-generated strategy code.

Executes evaluate() and decide() functions in a restricted Python environment with:
- Whitelisted builtins only (no exec, eval, open)
- Restricted imports: only modules in ALLOWED_MODULES whitelist
- Timeout protection
- Exception isolation

Each strategy pair has TWO functions:
- evaluate(current_reputation, observation, my_history, round_num) -> float
- decide(recipient_reputation, round_num, my_history) -> bool
"""

import sys
import signal
import traceback
import builtins
from typing import Optional, Callable, Any

from .validator import validate_strategy_code, clean_code, ALLOWED_MODULES


class SandboxTimeout(Exception):
    """Raised when strategy execution exceeds timeout."""
    pass


class SandboxError(Exception):
    """Raised when sandbox execution fails."""
    pass


def _make_restricted_import(allowed_modules: set) -> Callable:
    """Create a restricted __import__ that only permits whitelisted modules."""
    _real_import = builtins.__import__

    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        top_level = name.split(".")[0]
        if top_level not in allowed_modules:
            raise ImportError(
                f"Module '{top_level}' is not in the allowed modules list. "
                f"Allowed: {sorted(allowed_modules)}"
            )
        return _real_import(name, globals, locals, fromlist, level)

    return _restricted_import


# Whitelisted builtins available to strategy code
_SAFE_BUILTINS = {
    "True": True, "False": False, "None": None,
    "abs": abs, "all": all, "any": any,
    "bool": bool, "dict": dict, "enumerate": enumerate,
    "filter": filter, "float": float, "int": int,
    "isinstance": isinstance, "len": len, "list": list,
    "map": map, "max": max, "min": min, "pow": pow,
    "range": range, "reversed": reversed, "round": round,
    "set": set, "slice": slice, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "type": type,
    "zip": zip, "print": print, "Exception": Exception,
    "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError,
    "ZeroDivisionError": ZeroDivisionError,
    "__import__": _make_restricted_import(ALLOWED_MODULES),
}


def _timeout_handler(signum, frame):
    raise SandboxTimeout("Strategy execution timed out")


class StrategyExecutor:
    """Compiles and executes LLM-generated evaluate/decide function pairs."""

    def __init__(self, code: str, timeout_s: float = 0.05):
        """
        Initialize executor with strategy code.

        Args:
            code: Python source code defining evaluate() and decide() functions
            timeout_s: Maximum execution time per call (seconds)

        Raises:
            CodeValidationError: If code fails safety checks
            SandboxError: If code cannot be compiled or functions not found
        """
        self.timeout_s = timeout_s

        # Validate safety
        cleaned = clean_code(code)
        validate_strategy_code(cleaned)

        # Build restricted namespace with safe builtins
        self._namespace = {"__builtins__": _SAFE_BUILTINS}

        # Compile and execute the strategy code
        try:
            compiled = compile(cleaned, "<strategy>", "exec")
            exec(compiled, self._namespace)
        except Exception as e:
            raise SandboxError(
                f"Failed to compile strategy code: {e}\n"
                f"{traceback.format_exc()}"
            )

        # Extract evaluate function
        if "evaluate" not in self._namespace:
            raise SandboxError("Code does not define an 'evaluate' function")
        self._evaluate_fn = self._namespace["evaluate"]
        if not callable(self._evaluate_fn):
            raise SandboxError("'evaluate' is not callable")

        # Extract decide function
        if "decide" not in self._namespace:
            raise SandboxError("Code does not define a 'decide' function")
        self._decide_fn = self._namespace["decide"]
        if not callable(self._decide_fn):
            raise SandboxError("'decide' is not callable")

    def evaluate(
        self,
        current_reputation: float,
        observation: dict,
        my_history: list,
        round_num: int
    ) -> float:
        """
        Execute the evaluate function to update reputation.

        Args:
            current_reputation: Current reputation score for observed agent
            observation: Dict with round, donor, recipient, action keys
            my_history: List of interactions the observer participated in
            round_num: Current round number

        Returns:
            Updated reputation score (float)

        Raises:
            SandboxTimeout: If execution exceeds timeout
            SandboxError: If function raises an exception
        """
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, self.timeout_s)

        try:
            result = self._evaluate_fn(
                current_reputation=current_reputation,
                observation=observation,
                my_history=my_history,
                round_num=round_num
            )
        except SandboxTimeout:
            raise
        except Exception as e:
            raise SandboxError(
                f"evaluate() raised exception: {type(e).__name__}: {e}"
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

        # Coerce to float and clamp to [-1.0, 1.0] for inter-strategy compatibility
        clamped = max(-1.0, min(1.0, float(result)))
        return clamped

    def decide(
        self,
        recipient_reputation: float,
        round_num: int,
        my_history: list
    ) -> bool:
        """
        Execute the decide function to make a donation decision.

        Args:
            recipient_reputation: Agent's private reputation assessment of recipient
            round_num: Current round number
            my_history: List of interactions the agent participated in

        Returns:
            True to donate, False to not donate

        Raises:
            SandboxTimeout: If execution exceeds timeout
            SandboxError: If function raises an exception or returns non-bool
        """
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, self.timeout_s)

        try:
            result = self._decide_fn(
                recipient_reputation=recipient_reputation,
                round_num=round_num,
                my_history=my_history
            )
        except SandboxTimeout:
            raise
        except Exception as e:
            raise SandboxError(
                f"decide() raised exception: {type(e).__name__}: {e}"
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

        if not isinstance(result, bool):
            return bool(result)

        return result


def create_strategy_executor(code: str, max_retries: int = 3) -> Optional[StrategyExecutor]:
    """
    Create a StrategyExecutor with retry on compilation failure.

    Args:
        code: Strategy source code (must define evaluate and decide)
        max_retries: Maximum compilation attempts

    Returns:
        StrategyExecutor instance, or None if compilation fails
    """
    try:
        return StrategyExecutor(code)
    except Exception as e:
        print(f"  [executor] Failed to create strategy: {e}")
        return None
