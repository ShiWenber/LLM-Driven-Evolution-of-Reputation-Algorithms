"""Comment/docstring removal from evolved strategy code.

The clustering analysis embeds strategy code as its input representation
(TF-IDF features or code-embedding transformer). Comments and docstrings are
LLM-generated documentation text (e.g. "Adaptive image scoring: step size
depends on current reputation") -- they describe intent rather than behavior,
and can dominate the token features: strategies get pulled together purely on
commentary style while the actual decision logic is drowned out. Every
clustering input path therefore runs the code through
:func:`strip_code_comments` first, so only executable code is analyzed.

Implementation notes
--------------------
- ``#`` comments are dropped with the stdlib tokenizer, so a ``#`` inside a
  string literal (``"a # b"``) is preserved.
- Docstrings (module/class/function) are dropped as well: a STRING token
  whose previous significant token is INDENT is the first statement of a
  suite, i.e. a docstring by Python semantics.
- Broken / truncated LLM output that cannot be tokenized falls back to a
  quote-aware ``#`` stripper, and ultimately returns the code unchanged.
"""
from __future__ import annotations

import io
import tokenize


def strip_code_comments(code: str) -> str:
    """Return ``code`` with all comments and docstrings removed.

    Line and indentation structure is preserved so downstream tokenizers
    (code-embedding transformers, TF-IDF) see the same layout minus the
    documentation text. Idempotent: stripping already-stripped code is a
    no-op.
    """
    try:
        return _strip_with_tokenizer(code)
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return _strip_hash_comments(code)


def _strip_with_tokenizer(code: str) -> str:
    """Remove comments and docstrings using :mod:`tokenize`."""
    result = []
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for toktype, ttext, (srow, scol), (erow, ecol), _line in tokenize.generate_tokens(
        io.StringIO(code).readline
    ):
        if srow > last_lineno:
            last_col = 0
        if scol > last_col:
            result.append(" " * (scol - last_col))
        if toktype == tokenize.COMMENT:
            pass  # drop comment
        elif toktype == tokenize.STRING and prev_toktype == tokenize.INDENT:
            pass  # drop docstring
        else:
            result.append(ttext)
        prev_toktype = toktype
        last_col = ecol
        last_lineno = erow
    return "".join(result)


def _strip_hash_comments(code: str) -> str:
    """Fallback: drop ``#`` comments while respecting string literals.

    Used when the code cannot be tokenized (e.g. truncated LLM output).
    Tracks single/double/triple-quoted strings including escapes, so a ``#``
    inside a string is preserved. Docstrings are not removed in this mode.
    """
    lines = code.split("\n")
    out = []
    in_triple: str | None = None  # quote char of an open triple-quoted string
    for line in lines:
        line_out, in_triple = _strip_line(line, in_triple)
        out.append(line_out)
    return "\n".join(out)


def _strip_line(line: str, in_triple: str | None) -> tuple[str, str | None]:
    result = []
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if in_triple is not None:
            if line.startswith(in_triple * 3, i):
                result.append(in_triple * 3)
                i += 3
                in_triple = None
            else:
                result.append(ch)
                i += 1
            continue
        if ch in "\"'":
            if line.startswith(ch * 3, i):
                end = line.find(ch * 3, i + 3)
                if end == -1:
                    in_triple = ch
                    result.append(line[i:])
                    break
                result.append(line[i : end + 3])
                i = end + 3
            else:
                j = i + 1
                while j < n:
                    if line[j] == "\\":
                        j = min(j + 2, n)
                        continue
                    if line[j] == ch:
                        j += 1
                        break
                    j += 1
                result.append(line[i:j])
                i = j
            continue
        if ch == "#":
            break  # comment: drop the rest of the line
        result.append(ch)
        i += 1
    return "".join(result), in_triple
