"""Record builders: the only sanctioned way to construct parts of a record.

Each builder owns one shape and guarantees every key is present, so writers
never hand-assemble dicts with string literals. ``build_evolution_results``
composes them into a complete, validated top-level record.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .schema import (
    F_AGENT_ID,
    F_BIRTH_GEN,
    F_CODE,
    F_CONFIG_SCHEMA_VERSION,
    F_COOPERATION_RATE,
    F_COOPERATION_RATE_MEAN,
    F_FITNESS,
    F_FITNESS_MAX,
    F_FITNESS_MEAN,
    F_GENERATION,
    F_LINEAGE_ID,
    F_N_INTERACTIONS,
    F_ORIGIN,
    F_PARENT_ID,
    F_PARENT_LINEAGE_ID,
    F_POPULATION,
    F_SELF_REPUTATION,
    K_CONFIG,
    K_FINAL_POPULATION,
    K_LINEAGE_EVENTS,
    K_TRAJECTORY,
    SCHEMA_VERSION,
)
from .validate import validate_evolution_results


def population_entry(
    agent_id: int,
    code: str,
    fitness: float,
    cooperation_rate: float,
    self_reputation: Optional[float],
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
