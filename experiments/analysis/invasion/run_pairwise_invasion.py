"""DEPRECATED shim -- use :mod:`experiments.analysis.invasion.run_invasion`.

This module generalised the invasion sweep to an arbitrary ordered pair of
strategies. That is now one mode of the unified runner rather than a separate
engine: pass ``--source`` once per strategy plus ``--residents A>B``.

The CLI still works and is equivalent to::

    run_invasion --source A=... --source B=... --residents A>B ...

Note on ordering: "A invades B at count k" and "B invades A at count 100-k" are
the same composition with the slots drawn differently, so a single count sweep
already realises both. The historical ``--directions`` flag filtered ordered
pairs; ``--residents`` now does that job under a name that matches what it does.
"""

from __future__ import annotations

import sys

from .core import (  # noqa: F401 - re-exported for existing importers
    FITNESS_WINDOW_FRACTION,
    INTERACTIONS_PER_GENERATION,
    NUM_GENERATIONS,
    Competitor,
    EvolvedSource,
    payoff_imitation_update,
    play_generation_noisy,
    write_json_atomic,
)
from .run_invasion import (  # noqa: F401 - re-exported for existing importers
    POPULATION_SIZE,
    execute,
    load_candidate,
    parse_pairs,
    parse_sources,
    result_path,
    run_one,
)

KIND_A, KIND_B = "A", "B"

_DEPRECATION = (
    "experiments.analysis.invasion.run_pairwise_invasion is deprecated; use "
    "experiments.analysis.invasion.run_invasion --residents A>B (same engine). "
)


def load_source(label: str, agent_type: str, raw_path: str) -> EvolvedSource:
    """Load a strategy from a .py file or an evolutionary.json run."""
    return load_candidate(label, agent_type, raw_path)


def parse_strategies(values: list[str]) -> dict[str, EvolvedSource]:
    """Parse repeatable ``LABEL=[AGENT_TYPE=]PATH`` strategy specifications."""
    return parse_sources(values)


# The unified copier now adopts the model's whole identity, which is what this
# module's private ``pairwise_update`` existed to do.
pairwise_update = payoff_imitation_update


def main(argv: list[str] | None = None) -> int:
    print(f"[deprecated] {_DEPRECATION}", file=sys.stderr)
    from .run_invasion import main as unified_main

    return unified_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
