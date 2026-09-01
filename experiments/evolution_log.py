"""Unified evolution-log storage contract (schema v4).

Single source of truth for the on-disk format shared by the v2/v3
quantitative evolution experiment and every analysis consumer.

Writer side (must produce exactly this shape)
==============================================
  * ``experiments.v2_quantitative.population.V2EvolutionaryPopulation.run_evolution()``
    — both ``agent_type="agent-type1"`` (QuantitativeAgent, legacy "v2")
    and ``agent_type="agent-type2"`` (FullAgent, legacy "v3") converge on
    the SAME record layout. ``run_evolution()``
    assembles its return value through ``build_evolution_results()``.
  * CLI runners persist it via ``write_evolution_json()`` at the canonical
    path ``<output_root>/<label>_seed<N>/evolutionary.json`` (see
    ``evolution_json_path()``).

Reader side (must only rely on this contract)
=============================================
  * ``experiments.analysis.*`` — all loaders use ``load_evolution_json()``
    (optionally validating) and the field-name constants below instead of
    hard-coding string literals.

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
  * ``initial``          — gen-0 initialization (root, no parent)
  * ``imitate``          — Fermi 1-μ path: small LLM mutation of a role model
  * ``independent_init`` — Fermi μ path: fresh LLM init, no parent
  * ``mutate``           — legacy tournament path: mutated copy of a survivor

Adding a field never breaks readers (they use ``.get``); removing or
*renaming* a field MUST bump ``SCHEMA_VERSION`` and update the
migration-aware readers. ``validate_evolution_results`` is version-aware:
records declaring an older ``schema_version`` are checked against the stable
core only (so pre-v4 runs stay readable), while current-version records must
carry the full v4 shape (``lineage_events`` and per-agent lineage fields).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Version + file layout
# ---------------------------------------------------------------------------
SCHEMA_VERSION = 4
RESULTS_FILENAME = "evolutionary.json"


class EvolutionLogError(ValueError):
    """Raised when a record violates the evolution-log schema contract."""


# ---------------------------------------------------------------------------
# Top-level section keys
# ---------------------------------------------------------------------------
K_TRAJECTORY = "trajectory"
K_FINAL_POPULATION = "final_population"
K_LINEAGE_EVENTS = "lineage_events"
K_CONFIG = "config"

TOP_LEVEL_KEYS = (K_TRAJECTORY, K_FINAL_POPULATION, K_LINEAGE_EVENTS, K_CONFIG)

# ---------------------------------------------------------------------------
# Trajectory record fields
# ---------------------------------------------------------------------------
F_GENERATION = "generation"
F_COOPERATION_RATE_MEAN = "cooperation_rate_mean"
F_N_INTERACTIONS = "n_interactions"
F_FITNESS_MEAN = "fitness_mean"
F_FITNESS_MAX = "fitness_max"
F_POPULATION = "population"

# ---------------------------------------------------------------------------
# Per-agent population record fields (trajectory + final population)
# ---------------------------------------------------------------------------
F_AGENT_ID = "agent_id"
F_CODE = "code"
F_FITNESS = "fitness"
F_COOPERATION_RATE = "cooperation_rate"
F_SELF_REPUTATION = "self_reputation"
F_LINEAGE_ID = "lineage_id"
F_PARENT_ID = "parent_id"
F_PARENT_LINEAGE_ID = "parent_lineage_id"
F_ORIGIN = "origin"
F_BIRTH_GEN = "birth_gen"

# ---------------------------------------------------------------------------
# Lineage event record fields (subset of the population-record lineage fields)
# ---------------------------------------------------------------------------
LINEAGE_EVENT_FIELDS = (
    F_LINEAGE_ID, F_PARENT_LINEAGE_ID, F_PARENT_ID, F_ORIGIN, F_BIRTH_GEN,
)

# ---------------------------------------------------------------------------
# Origin values
# ---------------------------------------------------------------------------
ORIGIN_INITIAL = "initial"
ORIGIN_IMITATE = "imitate"
ORIGIN_INDEPENDENT_INIT = "independent_init"
ORIGIN_MUTATE = "mutate"
ORIGINS = (
    ORIGIN_INITIAL, ORIGIN_IMITATE, ORIGIN_INDEPENDENT_INIT, ORIGIN_MUTATE,
)

# ---------------------------------------------------------------------------
# Config record fields
# ---------------------------------------------------------------------------
F_CONFIG_SCHEMA_VERSION = "schema_version"
F_CONFIG_AGENT_TYPE = "agent_type"
F_CONFIG_POPULATION_SIZE = "population_size"
F_CONFIG_NUM_ROUNDS_PER_GEN = "num_rounds_per_gen"
F_CONFIG_BENEFIT = "benefit"
F_CONFIG_COST = "cost"
F_CONFIG_OBSERVABILITY = "observability"
F_CONFIG_OBSERVABILITY_P = "observability_p"
F_CONFIG_ELITE_COUNT = "elite_count"
F_CONFIG_NUM_ELIMINATE = "num_eliminate"
F_CONFIG_TOURNAMENT_SIZE = "tournament_size"
F_CONFIG_LLM_MODEL = "llm_model"
F_CONFIG_SEED = "seed"
F_CONFIG_USE_BASELINE = "use_baseline"
F_CONFIG_NUM_GENERATIONS = "num_generations"
F_CONFIG_TARGET_INTERACTIONS_PER_GEN = "target_interactions_per_gen"
F_CONFIG_LLM_THINKING = "llm_thinking"
F_CONFIG_LLM_MAX_TOKENS = "llm_max_tokens"
F_CONFIG_USE_FERMI = "use_fermi"
F_CONFIG_FERMI_BETA = "fermi_beta"
F_CONFIG_MUTATION_RATE_ON_ADOPTION = "mutation_rate_on_adoption"
F_CONFIG_IMITATION_LEARNING_MODE = "imitation_learning_mode"
F_CONFIG_UPDATES_PER_GEN = "updates_per_gen"
F_CONFIG_FORBID_SELF_PAIRING = "forbid_self_pairing"
F_CONFIG_FALLBACK_INIT_COUNT = "fallback_init_count"
F_CONFIG_FALLBACK_MUTATION_COUNT = "fallback_mutation_count"

# Minimal keys every config record must carry (schema_version is stamped by
# make_config / build_evolution_results).
REQUIRED_CONFIG_FIELDS = (
    F_CONFIG_SCHEMA_VERSION, F_CONFIG_AGENT_TYPE, F_CONFIG_SEED,
    F_CONFIG_POPULATION_SIZE,
)

# Agent families supported by the v2/v3 quantitative interface.
# Canonical values: "agent-type1" (legacy "v2") and "agent-type2" (legacy "v3").
AGENT_TYPES = ("agent-type1", "agent-type2")
# Legacy aliases accepted when reading historical records.
AGENT_TYPES_LEGACY = ("v2", "v3")

# ---------------------------------------------------------------------------
# Canonical directory / file layout
# ---------------------------------------------------------------------------
def run_dir(output_root, label: str, seed: int) -> Path:
    """Canonical per-seed run directory: ``<output_root>/<label>_seed<N>``."""
    return Path(output_root) / f"{label}_seed{seed}"


def evolution_json_path(output_root, label: str, seed: int) -> Path:
    """Canonical result file for one seed run."""
    return run_dir(output_root, label, seed) / RESULTS_FILENAME


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------
def population_entry(
    agent_id: int,
    code: str,
    fitness: float,
    cooperation_rate: float,
    self_reputation: float,
    *,
    lineage_id: Optional[int] = None,
    parent_id: Optional[int] = None,
    parent_lineage_id: Optional[int] = None,
    origin: Optional[str] = None,
    birth_gen: Optional[int] = None,
) -> Dict[str, Any]:
    """Build one per-agent population record (trajectory or final)."""
    return {
        F_AGENT_ID: agent_id,
        F_CODE: code,
        F_FITNESS: fitness,
        F_COOPERATION_RATE: cooperation_rate,
        F_SELF_REPUTATION: self_reputation,
        F_LINEAGE_ID: lineage_id,
        F_PARENT_ID: parent_id,
        F_PARENT_LINEAGE_ID: parent_lineage_id,
        F_ORIGIN: origin,
        F_BIRTH_GEN: birth_gen,
    }


def trajectory_entry(
    generation: int,
    cooperation_rate_mean: float,
    n_interactions: int,
    fitness_mean: float,
    fitness_max: float,
    population: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build one per-generation trajectory record."""
    return {
        F_GENERATION: generation,
        F_COOPERATION_RATE_MEAN: cooperation_rate_mean,
        F_N_INTERACTIONS: n_interactions,
        F_FITNESS_MEAN: fitness_mean,
        F_FITNESS_MAX: fitness_max,
        F_POPULATION: population,
    }


def lineage_event(
    lineage_id: int,
    parent_lineage_id: Optional[int] = None,
    parent_id: Optional[int] = None,
    origin: Optional[str] = None,
    birth_gen: Optional[int] = None,
) -> Dict[str, Any]:
    """Build one birth-event record for the lineage_events section."""
    return {
        F_LINEAGE_ID: lineage_id,
        F_PARENT_LINEAGE_ID: parent_lineage_id,
        F_PARENT_ID: parent_id,
        F_ORIGIN: origin,
        F_BIRTH_GEN: birth_gen,
    }


def make_config(**fields: Any) -> Dict[str, Any]:
    """Assemble a config record.

    ``schema_version`` is always stamped from ``SCHEMA_VERSION`` (single
    source of truth); any other keyword is passed through as-is, so extra
    experiment-specific knobs keep working without a schema change.
    """
    cfg: Dict[str, Any] = dict(fields)
    cfg[F_CONFIG_SCHEMA_VERSION] = SCHEMA_VERSION
    return cfg


# ---------------------------------------------------------------------------
# Top-level assembly + validation
# ---------------------------------------------------------------------------
def build_evolution_results(
    *,
    trajectory: List[Dict[str, Any]],
    final_population: List[Dict[str, Any]],
    lineage_events: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the top-level record, stamp the schema version, validate."""
    data = {
        K_TRAJECTORY: trajectory,
        K_FINAL_POPULATION: final_population,
        K_LINEAGE_EVENTS: lineage_events,
        K_CONFIG: make_config(**config),
    }
    validate_evolution_results(data)
    return data


def validate_evolution_results(
    data: Dict[str, Any],
    *,
    require_final_population: bool = True,
) -> None:
    """Check a record against the schema contract; raise EvolutionLogError.

    Version-aware: the stable core (present in every schema version) is
    always enforced — ``config`` (with ``schema_version``), ``trajectory``
    (records with ``generation`` / ``cooperation_rate_mean`` /
    ``population``), and optionally ``final_population``. Records declaring
    the CURRENT ``SCHEMA_VERSION`` additionally must carry the v4-only
    sections (``lineage_events``, per-agent lineage fields, valid
    ``origin`` values). Older records (e.g. schema v3, which predates
    ``lineage_events``) pass the core checks so existing runs stay readable;
    version-sensitive consumers (``lineage.build`` / ``plot_lineage``) apply
    their own ``schema_version >= 4`` guard on top.

    Readers use ``.get`` for optional fields, so only structural requirements
    are enforced here; renames/removals must bump ``SCHEMA_VERSION``.
    """
    errors: List[str] = []

    def _require(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    for key in (K_CONFIG, K_TRAJECTORY):
        _require(key in data, f"missing top-level key {key!r}")
    if require_final_population:
        _require(K_FINAL_POPULATION in data,
                 f"missing top-level key {K_FINAL_POPULATION!r}")

    if errors:
        raise EvolutionLogError("; ".join(errors))

    config = data[K_CONFIG]
    if not isinstance(config, dict):
        raise EvolutionLogError(f"{K_CONFIG!r} must be a dict, got {type(config).__name__}")
    for f in REQUIRED_CONFIG_FIELDS:
        _require(f in config, f"config missing required field {f!r}")
    version = config.get(F_CONFIG_SCHEMA_VERSION)
    if version is None:
        errors.append(f"config missing {F_CONFIG_SCHEMA_VERSION!r}")
    is_current = version == SCHEMA_VERSION
    if version is not None and not isinstance(version, int):
        errors.append(f"config schema_version={version!r} is not an int")

    trajectory = data[K_TRAJECTORY]
    if not isinstance(trajectory, list):
        errors.append(f"{K_TRAJECTORY!r} must be a list")
    else:
        for i, gen in enumerate(trajectory):
            if not isinstance(gen, dict):
                errors.append(f"trajectory[{i}] is not a dict")
                continue
            for f in (F_GENERATION, F_COOPERATION_RATE_MEAN, F_POPULATION):
                _require(f in gen, f"trajectory[{i}] missing field {f!r}")
            for j, a in enumerate(gen.get(F_POPULATION, [])):
                if not isinstance(a, dict):
                    errors.append(f"trajectory[{i}].population[{j}] is not a dict")
                    continue
                for f in (F_AGENT_ID, F_CODE):
                    _require(f in a, f"trajectory[{i}].population[{j}] missing {f!r}")

    if is_current:
        # v4-only sections are mandatory for current-schema records.
        _require(K_LINEAGE_EVENTS in data,
                 f"missing top-level key {K_LINEAGE_EVENTS!r} (schema v4)")
        agent_type = config.get(F_CONFIG_AGENT_TYPE)
        if agent_type is not None and (
            agent_type not in AGENT_TYPES and agent_type not in AGENT_TYPES_LEGACY
        ):
            errors.append(
                f"config agent_type={agent_type!r} not in "
                f"{AGENT_TYPES} (legacy {AGENT_TYPES_LEGACY})"
            )
        for i, ev in enumerate(data.get(K_LINEAGE_EVENTS, [])):
            if not isinstance(ev, dict):
                errors.append(f"lineage_events[{i}] is not a dict")
                continue
            _require(F_LINEAGE_ID in ev,
                     f"lineage_events[{i}] missing {F_LINEAGE_ID!r}")
            origin = ev.get(F_ORIGIN)
            if origin is not None and origin not in ORIGINS:
                errors.append(
                    f"lineage_events[{i}] origin={origin!r} not in {ORIGINS}"
                )

    if errors:
        raise EvolutionLogError("; ".join(errors))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def write_evolution_json(
    path, data: Dict[str, Any], *, validate: bool = True
) -> Path:
    """Atomically persist a top-level record (tmp file + rename).

    Creates parent directories as needed. Returns the written path.
    """
    if validate:
        validate_evolution_results(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return path


def load_evolution_json(path, *, validate: bool = True) -> Dict[str, Any]:
    """Load a top-level record; optionally validate it against the contract."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if validate:
        validate_evolution_results(data)
    return data
