"""Static contract validation for LLM-generated strategy code.

This module validates the *shape* of LLM-produced code before it is
compiled by an executor, so that bad code is rejected early (during
``_validate_code``) instead of surfacing later at ``exec`` time. It is a
pure static check — no runtime execution, no restricted environment, so
it never changes how valid strategies run.

Two contracts are supported, matching the two executor interfaces:

* V2 (``V2StrategyExecutor``): two top-level functions
  ``observe(A_rep, A_action, B_rep, B_action, my_reputation)`` and
  ``decide(my_reputation, opponent_reputation)``.
* V3 (``V3StrategyExecutor``): a class named ``LLMAgent`` with
  ``__init__(self, agent_id)``, ``decide(self)`` and
  ``observe(self, donor_id, donor_action, recipient_id,
  recipient_action)``.

Only syntax validity and required functions/signatures are enforced.
Dangerous imports/builtins are intentionally *not* rejected here: the
current architecture trusts LLM output and falls back via try/except at
runtime. The (now‑removed) sandbox's forbidden-import list is not
reintroduced by design.
"""

from __future__ import annotations

import ast
from typing import Dict, Sequence

# Maximum allowed strategy code length, guarded to avoid pathological
# prompt output. Mirrors the old sandbox limit.
MAX_CODE_LEN = 3000


class CodeContractError(Exception):
    """Raised when strategy code fails static contract validation."""


# --- internal helpers -----------------------------------------------------


def _syntax_ok(code: str) -> None:
    if len(code) > MAX_CODE_LEN:
        raise CodeContractError(
            f"code too long ({len(code)} chars, max {MAX_CODE_LEN})"
        )
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise CodeContractError(f"syntax error: {e}")


def _validate_signature(
    node: ast.FunctionDef,
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    """Ensure positional calls made by the framework are safe and unambiguous."""
    positional = [*node.args.posonlyargs, *node.args.args]
    names = [argument.arg for argument in positional]
    missing = [name for name in expected if name not in names]
    if missing:
        raise CodeContractError(
            f"{label} missing parameter(s): {sorted(missing)}"
        )
    if names[: len(expected)] != list(expected):
        raise CodeContractError(
            f"{label} parameter order must start with {list(expected)}"
        )

    required_positional = len(positional) - len(node.args.defaults)
    required_keyword_only = [
        argument.arg
        for argument, default in zip(
            node.args.kwonlyargs, node.args.kw_defaults
        )
        if default is None
    ]
    if required_positional > len(expected) or required_keyword_only:
        raise CodeContractError(
            f"{label} declares additional required parameters"
        )


def _require_functions(
    tree: ast.Module,
    required: Dict[str, Sequence[str]],
) -> None:
    """Require callable-compatible functions directly in the module body."""
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for func_name, expected in required.items():
        node = functions.get(func_name)
        if node is None:
            raise CodeContractError(
                f"must define {func_name}(); top-level {func_name}() is required"
            )
        _validate_signature(node, expected, label=f"{func_name}()")


def _require_class_methods(
    tree: ast.Module,
    class_name: str,
    required_methods: Dict[str, Sequence[str]],
) -> None:
    """Ensure a class named ``class_name`` exists with the required methods.

    ``required_methods`` maps method name -> set of parameter names that MUST
    be present on that method (beyond ``self``).
    """
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if not classes:
        raise CodeContractError(f"must define top-level class {class_name}")
    cls = classes[0]
    methods = {
        n.name: n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
    }
    for method, expected in required_methods.items():
        if method not in methods:
            raise CodeContractError(f"{class_name} must define {method}()")
        _validate_signature(
            methods[method],
            expected,
            label=f"{class_name}.{method}()",
        )


# --- public contract validators ------------------------------------------


def validate_v2_contract(code: str) -> None:
    """Validate the V2 two-function contract (observe + decide)."""
    _syntax_ok(code)
    tree = ast.parse(code)
    _require_functions(
        tree,
        {
            "observe": (
                "A_rep",
                "A_action",
                "B_rep",
                "B_action",
                "my_reputation",
            ),
            "decide": ("my_reputation", "opponent_reputation"),
        },
    )


def validate_v3_contract(code: str) -> None:
    """Validate the V3 ``LLMAgent`` class contract."""
    _syntax_ok(code)
    tree = ast.parse(code)
    _require_class_methods(
        tree,
        "LLMAgent",
        {
            "__init__": ("self", "agent_id"),
            "decide": ("self",),
            "observe": (
                "self",
                "donor_id",
                "donor_action",
                "recipient_id",
                "recipient_action",
            ),
        },
    )
