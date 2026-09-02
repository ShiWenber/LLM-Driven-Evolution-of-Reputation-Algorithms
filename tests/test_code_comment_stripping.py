"""Tests for comment/docstring removal before clustering analysis input.

Every clustering representation (code-embedding transformer input in
``clustering.pipeline.embed_codes``, TF-IDF in the root analysis scripts)
receives strategy code with comments and docstrings stripped by
:func:`experiments.analysis.clustering.comments.strip_code_comments`.

Comments in evolved strategies are LLM-written documentation ("Adaptive image
scoring: step size depends on current reputation"); if they reached the
embedding, strategies would be pulled together purely on commentary style
while the actual decision logic was drowned out. These tests pin down the
exact contract of the stripper:

- ``#`` comments are removed, but ``#`` inside string literals is preserved;
- docstrings (module/class/function) are removed;
- line and indentation structure is preserved;
- the operation is idempotent;
- broken / truncated LLM output falls back to a quote-aware stripper instead
  of raising.
"""
from __future__ import annotations

import pytest

from experiments.analysis.clustering.comments import (
    _strip_hash_comments,
    strip_code_comments,
)

# A representative evolved strategy as produced by the LLM evolution loop:
# full of explanatory comments that must NOT influence the clustering input.
EVOLVED_STRATEGY = '''\
def observe(
    A_rep: float,
    A_action: str,
    B_rep: float,
    B_action: str,
    my_reputation: float
) -> float:
    # Adaptive image scoring: step size depends on current reputation
    # High reputation updates are smaller (stable), low reputation updates are larger (recovery)
    def update_step(rep: float) -> float:
        if rep >= 0.5:
            return 0.05
        elif rep >= 0.0:
            return 0.1
        elif rep >= -0.5:
            return 0.15
        else:
            return 0.2

    if A_action == 'cooperate':
        new_A = A_rep + update_step(A_rep)
    else:
        new_A = A_rep - update_step(A_rep)

    # Clamp to [-1, 1] to keep within reasonable bounds
    return max(-1.0, min(1.0, new_A))
'''


# ---------------------------------------------------------------- comments


def test_trailing_hash_comments_removed():
    assert strip_code_comments("x = 1  # trailing comment") == "x = 1  "


def test_full_line_hash_comments_removed():
    code = "def f():\n    # only a comment line\n    return 1\n"
    out = strip_code_comments(code)
    assert "#" not in out
    assert "def f():" in out
    assert "return 1" in out


def test_hash_inside_string_literal_preserved():
    code = 's = "a # b"  # real comment'
    out = strip_code_comments(code)
    # The '#' inside the string literal survives; the trailing comment goes.
    assert out == 's = "a # b"  '
    assert "# real comment" not in out


def test_hash_inside_triple_quoted_string_preserved():
    code = 's3 = """multi\n# line\nstring"""  # comment\n'
    out = strip_code_comments(code)
    assert "# line" in out
    assert "# comment" not in out
    assert '"""multi\n# line\nstring"""' in out


# ------------------------------------------------------------- docstrings


def test_module_and_function_docstrings_removed():
    code = (
        '"""Module docstring."""\n'
        "# header comment\n"
        "def foo():\n"
        '    """Function docstring."""\n'
        "    x = 1  # trailing\n"
        "    return x\n"
    )
    out = strip_code_comments(code)
    assert "Module docstring" not in out
    assert "Function docstring" not in out
    assert "def foo():" in out
    assert "x = 1" in out
    assert "return x" in out


def test_assigned_string_literal_is_not_a_docstring():
    # A string literal that is an expression (not the first suite statement)
    # must be kept -- it is data, not documentation.
    code = 'x = """not a docstring"""\n# c\ny = x + 1\n'
    out = strip_code_comments(code)
    assert '"""not a docstring"""' in out
    assert "y = x + 1" in out


def test_class_docstring_removed():
    code = (
        "class Strategy:\n"
        '    """Class docstring."""\n'
        "    def decide(self):\n"
        '        """Method docstring."""\n'
        "        return True  # always\n"
    )
    out = strip_code_comments(code)
    assert "Class docstring" not in out
    assert "Method docstring" not in out
    assert "def decide(self):" in out
    assert "return True" in out


# ------------------------------------------------------------- structure


def test_line_structure_preserved():
    code = "def f():\n    x = 1  # t\n    return x\n"
    out = strip_code_comments(code)
    assert out.count("\n") == code.count("\n")
    # Indentation of surviving lines is untouched.
    assert "    x = 1" in out


def test_idempotent():
    code = (
        '"""Doc."""\n'
        "# c\n"
        "def f():\n"
        "    x = 1  # t\n"
        "    return x\n"
    )
    once = strip_code_comments(code)
    assert strip_code_comments(once) == once


def test_no_comments_input_unchanged():
    assert strip_code_comments("x = 1") == "x = 1"
    assert strip_code_comments("x = 1\n") == "x = 1\n"


def test_comment_only_input_becomes_blank():
    out = strip_code_comments("# just a comment\n")
    assert "#" not in out
    assert out.strip() == ""


# --------------------------------------------------------------- fallback


def test_fallback_strips_hash_in_truncated_code():
    # Tokenize raises on truncated output; the fallback must still remove '#'
    # without raising (and without touching '#' inside strings).
    broken = 'def f(:\n    x = 1  # trunc\n    s = "keep # me"\n'
    out = strip_code_comments(broken)
    assert "# trunc" not in out
    assert '"keep # me"' in out
    assert "x = 1" in out


def test_fallback_respects_strings_inside_lines():
    assert _strip_hash_comments('a = "#"  # gone\n') == 'a = "#"  \n'
    assert _strip_hash_comments("b = '#'  # gone\n") == "b = '#'  \n"


# ------------------------------------------------------- evolved strategy


def test_evolved_strategy_comments_removed():
    out = strip_code_comments(EVOLVED_STRATEGY)
    assert "#" not in out
    assert "def observe(" in out
    assert "def update_step(rep: float) -> float:" in out
    assert "return max(-1.0, min(1.0, new_A))" in out
    # Code that only differs by comments collapses to the same stripped form.
    comment_variant = EVOLVED_STRATEGY.replace(
        "# Clamp to [-1, 1] to keep within reasonable bounds",
        "# Some totally different commentary",
    )
    assert strip_code_comments(comment_variant) == out


def test_stripped_evolved_strategy_still_valid_python():
    # The stripped code must remain syntactically valid Python.
    compile(strip_code_comments(EVOLVED_STRATEGY), "<stripped>", "exec")
