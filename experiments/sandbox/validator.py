"""AST validation for LLM-generated strategy code.

Ensures generated code is safe to execute by checking for:
- Syntax errors
- Dangerous builtins (exec, eval, open, etc.)
- Blacklisted imports (os, sys, subprocess, etc.)
- Structural requirements (must define both `evaluate` and `decide` functions)
"""

import ast
from typing import Optional

# Builtins that are explicitly forbidden in strategy code
FORBIDDEN_BUILTINS = {
    "exec", "eval", "compile", "__import__",
    "open", "input",
    "globals", "locals", "vars",
    "getattr", "setattr", "delattr",
    "breakpoint",
}

# Modules that cannot be imported
FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket",
    "importlib", "inspect", "ctypes",
    "pathlib", "io", "pickle", "marshal",
    "builtins", "__builtins__",
    "requests", "urllib", "http",
    "multiprocessing", "threading",
}

# Modules allowed for strategy logic
ALLOWED_MODULES = {
    "math", "statistics", "collections",
    "itertools", "functools", "random",
    "copy", "operator", "hashlib",
    "json", "re", "typing",
}


class CodeValidationError(Exception):
    """Raised when strategy code fails validation."""
    pass


def validate_strategy_code(code: str) -> str:
    """
    Validate LLM-generated strategy code.

    Checks:
    1. Syntactically valid Python
    2. Defines `strategy` function with correct signature
    3. No dangerous builtins or imports
    4. Code stays within length limits

    Args:
        code: Python source code string

    Returns:
        The validated code string (possibly cleaned)

    Raises:
        CodeValidationError: If code fails any check
    """
    if not code or not code.strip():
        raise CodeValidationError("Empty code")

    if len(code) > 3000:
        raise CodeValidationError(
            f"Code too long ({len(code)} chars, max 3000)"
        )

    # Parse AST
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeValidationError(f"Syntax error: {e}")

    # Check for forbidden constructs
    _check_imports(tree)
    _check_builtins(tree)
    _check_strategy_function(tree)

    return code


def _check_imports(tree: ast.AST):
    """Check all imports are safe."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module in FORBIDDEN_MODULES:
                    raise CodeValidationError(
                        f"Forbidden import: {module}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            module = node.module.split(".")[0]
            if module in FORBIDDEN_MODULES:
                raise CodeValidationError(
                    f"Forbidden import: {module}"
                )


def _check_builtins(tree: ast.AST):
    """Check for forbidden builtin function calls."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_BUILTINS:
                    raise CodeValidationError(
                        f"Forbidden builtin: {node.func.id}()"
                    )


def _check_strategy_function(tree: ast.AST):
    """Check that both `evaluate` and `decide` functions are defined with correct signatures."""
    has_evaluate = False
    has_decide = False

    evaluate_required = ["current_reputation", "observation", "my_history", "round_num"]
    decide_required = ["recipient_reputation", "round_num", "my_history"]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            if node.name == "evaluate":
                has_evaluate = True
                for req in evaluate_required:
                    if req not in args:
                        raise CodeValidationError(
                            f"evaluate() missing required parameter: '{req}'"
                        )
            elif node.name == "decide":
                has_decide = True
                for req in decide_required:
                    if req not in args:
                        raise CodeValidationError(
                            f"decide() missing required parameter: '{req}'"
                        )

    if not has_evaluate:
        raise CodeValidationError(
            "Code must define an `evaluate` function"
        )
    if not has_decide:
        raise CodeValidationError(
            "Code must define a `decide` function"
        )


def clean_code(code: str) -> str:
    """Extract strategy function from LLM output (strip markdown fences)."""
    code = code.strip()

    # Remove markdown code fences if present
    if code.startswith("```"):
        lines = code.split("\n")
        # Remove first line (```python or ```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it's closing ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    return code.strip()
