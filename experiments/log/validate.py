"""Schema validation, split by version tier.

``validate_evolution_results`` runs two tiers:

* **core** -- the sections present in *every* schema version (``config``,
  ``trajectory``, optionally ``final_population``). Enforced always, so
  pre-v4 runs stay readable.
* **v4** -- the sections schema v4 introduced (``lineage_events``, the
  per-agent lineage fields, a valid ``origin``). Enforced only when the
  record declares the *current* ``SCHEMA_VERSION``.

Readers use ``.get`` for optional fields, so only structural requirements are
enforced; renames/removals must bump ``SCHEMA_VERSION``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .schema import (
    AGENT_TYPES,
    F_AGENT_ID,
    F_CODE,
    F_CONFIG_AGENT_TYPE,
    F_CONFIG_SCHEMA_VERSION,
    F_COOPERATION_RATE_MEAN,
    F_GENERATION,
    F_LINEAGE_ID,
    F_ORIGIN,
    F_POPULATION,
    K_CONFIG,
    K_FINAL_POPULATION,
    K_LINEAGE_EVENTS,
    K_TRAJECTORY,
    ORIGINS,
    REQUIRED_CONFIG_FIELDS,
    SCHEMA_VERSION,
    EvolutionLogError,
    schema_version,
)


def _check_top_level(data: Dict[str, Any], require_final_population: bool) -> List[str]:
    """Required sections. These gate everything else, so they raise early."""
    errors = [
        f"missing top-level key {key!r}"
        for key in (K_CONFIG, K_TRAJECTORY)
        if key not in data
    ]
    if require_final_population and K_FINAL_POPULATION not in data:
        errors.append(f"missing top-level key {K_FINAL_POPULATION!r}")
    return errors


def _check_config_fields(config: Dict[str, Any], errors: List[str]) -> None:
    for f in REQUIRED_CONFIG_FIELDS:
        if f not in config:
            errors.append(f"config missing required field {f!r}")
    version = config.get(F_CONFIG_SCHEMA_VERSION)
    if version is None:
        errors.append(f"config missing {F_CONFIG_SCHEMA_VERSION!r}")
    elif not isinstance(version, int):
        errors.append(f"config schema_version={version!r} is not an int")


def _check_trajectory(data: Dict[str, Any], errors: List[str]) -> None:
    trajectory = data[K_TRAJECTORY]
    if not isinstance(trajectory, list):
        errors.append(f"{K_TRAJECTORY!r} must be a list")
        return
    for i, gen in enumerate(trajectory):
        if not isinstance(gen, dict):
            errors.append(f"trajectory[{i}] is not a dict")
            continue
        for f in (F_GENERATION, F_COOPERATION_RATE_MEAN, F_POPULATION):
            if f not in gen:
                errors.append(f"trajectory[{i}] missing field {f!r}")
        for j, agent in enumerate(gen.get(F_POPULATION, [])):
            if not isinstance(agent, dict):
                errors.append(f"trajectory[{i}].population[{j}] is not a dict")
                continue
            for f in (F_AGENT_ID, F_CODE):
                if f not in agent:
                    errors.append(
                        f"trajectory[{i}].population[{j}] missing {f!r}"
                    )


def _check_v4(data: Dict[str, Any], config: Dict[str, Any], errors: List[str]) -> None:
    """Sections and fields that only schema v4 requires."""
    if K_LINEAGE_EVENTS not in data:
        errors.append(
            f"missing top-level key {K_LINEAGE_EVENTS!r} (schema v4)"
        )
    agent_type = config.get(F_CONFIG_AGENT_TYPE)
    if agent_type is not None and agent_type not in AGENT_TYPES:
        errors.append(f"config agent_type={agent_type!r} not in {AGENT_TYPES}")
    for i, event in enumerate(data.get(K_LINEAGE_EVENTS, [])):
        if not isinstance(event, dict):
            errors.append(f"lineage_events[{i}] is not a dict")
            continue
        if F_LINEAGE_ID not in event:
            errors.append(f"lineage_events[{i}] missing {F_LINEAGE_ID!r}")
        origin = event.get(F_ORIGIN)
        if origin is not None and origin not in ORIGINS:
            errors.append(f"lineage_events[{i}] origin={origin!r} not in {ORIGINS}")


def validate_evolution_results(
    data: Dict[str, Any],
    *,
    require_final_population: bool = True,
) -> None:
    """Check a record against the schema contract; raise ``EvolutionLogError``.

    Version-aware: the stable core is always enforced, while records
    declaring the current ``SCHEMA_VERSION`` must additionally carry the
    v4-only sections. Older records (e.g. schema v3, which predates
    ``lineage_events``) pass the core checks so existing runs stay readable;
    version-sensitive consumers call ``require_schema_v4`` on top.
    """
    errors = _check_top_level(data, require_final_population)
    if errors:
        raise EvolutionLogError("; ".join(errors))

    config = data[K_CONFIG]
    if not isinstance(config, dict):
        raise EvolutionLogError(
            f"{K_CONFIG!r} must be a dict, got {type(config).__name__}"
        )

    _check_config_fields(config, errors)
    _check_trajectory(data, errors)
    if schema_version(data) == SCHEMA_VERSION:
        _check_v4(data, config, errors)

    if errors:
        raise EvolutionLogError("; ".join(errors))
