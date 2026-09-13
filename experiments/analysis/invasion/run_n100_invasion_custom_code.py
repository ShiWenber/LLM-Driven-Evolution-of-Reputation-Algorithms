"""DEPRECATED shim -- use :mod:`experiments.analysis.invasion.run_invasion`.

This was the variant of the invasion sweep for hand-written strategy ``.py``
files. ``run_invasion`` now accepts both ``.py`` and ``evolutionary.json``
candidates through the same ``--source`` flag, so the variant is redundant.

The CLI still works and is equivalent to::

    run_invasion --norms <all norms> --source LABEL=PATH_TO_PY ...
"""

from __future__ import annotations

import sys

from .core import NORMS, EvolvedSource  # noqa: F401 - re-exported
from .run_invasion import (  # noqa: F401 - re-exported for existing importers
    DEFAULT_COUNTS,
    DEFAULT_SEEDS,
    POPULATION_SIZE,
    cache_matches,
    execute,
    existing_result_path,
    experiment_name,
    load_custom_source,
    result_path,
    write_json_atomic,
    write_summary,
)

AGENT_TYPE = "agent-type1"

_DEPRECATION = (
    "experiments.analysis.invasion.run_n100_invasion_custom_code is deprecated; "
    "use experiments.analysis.invasion.run_invasion (same engine, unified CLI). "
)


def parse_sources(values: list[str]) -> dict[str, EvolvedSource]:
    """Parse repeatable ``LABEL=PATH_TO_PY`` specifications."""
    sources: dict[str, EvolvedSource] = {}
    for value in values:
        try:
            label, raw_path = value.split("=", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --source {value!r}; expected LABEL=PATH_TO_PY"
            ) from exc
        if not label or label in sources:
            raise ValueError(f"Source labels must be non-empty and unique: {label!r}")
        sources[label] = load_custom_source(label, raw_path)
    return sources


def main(argv: list[str] | None = None) -> int:
    print(f"[deprecated] {_DEPRECATION}", file=sys.stderr)
    from .run_invasion import main as unified_main

    return unified_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
