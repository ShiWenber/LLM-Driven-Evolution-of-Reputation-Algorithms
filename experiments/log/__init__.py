"""Unified evolution-log storage contract (schema v4).

Single source of truth for the on-disk format shared by the v2/v3
quantitative evolution experiment and every analysis consumer.

Writer side (must produce exactly this shape)
==============================================
  * ``experiments.v2_quantitative.population.V2EvolutionaryPopulation.run_evolution()``
    -- both ``agent_type="agent-type1"`` (QuantitativeAgent)
    and ``agent_type="agent-type2"`` (FullAgent) converge on
    the SAME record layout. ``run_evolution()``
    assembles its return value through ``build_evolution_results()``.
  * CLI runners persist it via ``write_evolution_json()`` at the canonical
    path ``<output_root>/<label>_seed<N>/evolutionary.json`` (see
    ``evolution_json_path()``).

Reader side (must only rely on this contract)
=============================================
  * ``experiments.analysis.*`` -- all loaders use ``load_evolution_json()``
    (optionally validating) and the field-name constants below instead of
    hard-coding string literals.

Layout
======
``experiments/evolution_log.py`` re-exports this package, so both the old
module path and ``experiments.log`` work. Inside the package:

  ``schema``    version, section keys, field names, enum values, and the
                ``schema_version`` / ``require_schema_v4`` version queries
  ``builders``  record constructors (the only sanctioned way to build them)
  ``validate``  core + v4 schema validation
  ``io``        canonical run paths, atomic write, validated load

On-disk shape
=============
::

    {
      "trajectory": [                      # one record per generation
        {
          "generation": 0,
          "cooperation_rate_mean": 0.53,
          "n_interactions": 1000,
          "fitness_mean": 1.2,
          "fitness_max": 2.0,
          "population": [                  # per-agent snapshot
            {
              "agent_id": 0, "code": "...", "fitness": 1.0,
              "cooperation_rate": 0.5, "self_reputation": 0.1,
              "lineage_id": 0, "parent_id": null,
              "parent_lineage_id": null, "origin": "initial",
              "birth_gen": 0
            }, ...
          ]
        }, ...
      ],
      "final_population": [ ... ],         # same per-agent record shape
      "lineage_events": [                  # full phylogeny, incl. extinct
        {"lineage_id": 0, "parent_lineage_id": null, "parent_id": null,
         "origin": "initial", "birth_gen": 0}, ...
      ],
      "config": {
        "schema_version": 4, "agent_type": "agent-type1" | "agent-type2", ...
      }
    }

``origin`` is one of the ``ORIGIN_*`` constants:
  * ``initial``          -- gen-0 initialization (root, no parent)
  * ``imitate``          -- Fermi 1-mu path: small LLM mutation of a role model
  * ``independent_init`` -- Fermi mu path: fresh LLM init, no parent
  * ``mutate``           -- legacy tournament path: mutated copy of a survivor

Adding a field never breaks readers (they use ``.get``); removing or
*renaming* a field MUST bump ``SCHEMA_VERSION`` and update the
migration-aware readers. ``validate_evolution_results`` is version-aware:
records declaring an older ``schema_version`` are checked against the stable
core only (so pre-v4 runs stay readable), while current-version records must
carry the full v4 shape (``lineage_events`` and per-agent lineage fields).
"""
from __future__ import annotations

from .builders import (
    build_evolution_results,
    lineage_event,
    make_config,
    population_entry,
    trajectory_entry,
)
from .io import (
    evolution_json_path,
    load_evolution_json,
    run_dir,
    write_evolution_json,
)
from .schema import *  # noqa: F401,F403 - the vocabulary, re-exported wholesale
from .schema import __all__ as _schema_all
from .validate import validate_evolution_results

__all__ = [
    *_schema_all,
    "population_entry",
    "trajectory_entry",
    "lineage_event",
    "make_config",
    "build_evolution_results",
    "validate_evolution_results",
    "run_dir",
    "evolution_json_path",
    "write_evolution_json",
    "load_evolution_json",
]
