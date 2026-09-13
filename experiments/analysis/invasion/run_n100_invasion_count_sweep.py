"""DEPRECATED shim -- use :mod:`experiments.analysis.invasion.run_invasion`.

This module used to be the main N=100 invasion sweep. Its engine has been
folded into the unified :mod:`~experiments.analysis.invasion.run_invasion`,
which serves norm residents, hand-written candidates and arbitrary
strategy-vs-strategy pairs with one code path.

The CLI still works and is equivalent to::

    run_invasion --norms <all norms> --source LABEL=AGENT_TYPE=PATH ...

``load_representative_from_path`` and the output-path helpers are re-exported so
existing callers keep importing from here.
"""

from __future__ import annotations

import sys

from .core import (  # noqa: F401 - re-exported for existing importers
    AGENT_TYPES,
    ARCHIVED_DIRECTION_LABEL,
    FITNESS_WINDOW_FRACTION,
    INTERACTIONS_PER_GENERATION,
    KIND_CANDIDATE,
    KIND_NORM,
    KIND_RESIDENT,
    NORMS,
    NUM_GENERATIONS,
    Competitor,
    EvolvedSource,
    norm_source,
    payoff_imitation_update,
    play_generation_noisy,
    root_lineage,
    write_json_atomic,
)
from .run_invasion import (  # noqa: F401 - re-exported for existing importers
    DEFAULT_COUNTS,
    DEFAULT_SEEDS,
    POPULATION_SIZE,
    cache_matches,
    default_output,
    execute,
    existing_result_path,
    experiment_name,
    load_candidate,
    load_custom_source,
    load_representative_from_path,
    noisy_output,
    parse_pairs,
    parse_sources,
    result_path,
    run_one,
    write_summary,
)

_DEPRECATION = (
    "experiments.analysis.invasion.run_n100_invasion_count_sweep is deprecated; "
    "use experiments.analysis.invasion.run_invasion (same engine, unified CLI). "
)


def main(argv: list[str] | None = None) -> int:
    print(f"[deprecated] {_DEPRECATION}", file=sys.stderr)
    from .run_invasion import main as unified_main

    return unified_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
