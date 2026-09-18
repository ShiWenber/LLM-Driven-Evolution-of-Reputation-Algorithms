"""Evolution-log schema: version, section keys, field names, enum values.

The lowest layer of the ``experiments.log`` contract package. It defines
*names only* -- no building, validating or I/O -- so every other module in
this package (and every consumer) can share one vocabulary without creating
import cycles.

Naming convention (one rule per layer of the record):

``K_*``           top-level section keys
``F_*``           per-generation and per-agent record fields
``F_CONFIG_*``    fields of the ``config`` record
``ORIGIN_*``      allowed values of ``origin``

Adding a field never breaks readers (they use ``.get``). Removing or
*renaming* one MUST bump ``SCHEMA_VERSION`` and update the migration-aware
readers.
"""
from __future__ import annotations

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
ORIGIN_INITIAL = "initial"                    # gen-0 initialization (root)
ORIGIN_IMITATE = "imitate"                    # Fermi 1-mu: mutated role model
ORIGIN_INDEPENDENT_INIT = "independent_init"  # Fermi mu: fresh LLM init
ORIGIN_MUTATE = "mutate"                      # legacy tournament path
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
F_CONFIG_FITNESS_WINDOW_FRACTION = "fitness_window_fraction"
F_CONFIG_OBSERVATION_SCHEDULE = "observation_schedule"
F_CONFIG_LLM_THINKING = "llm_thinking"
F_CONFIG_LLM_MAX_TOKENS = "llm_max_tokens"
F_CONFIG_LEARNING_METHOD = "learning_method"
F_CONFIG_FERMI_BETA = "fermi_beta"
F_CONFIG_MUTATION_RATE_ON_ADOPTION = "mutation_rate_on_adoption"
F_CONFIG_IMITATION_LEARNING_MODE = "imitation_learning_mode"
F_CONFIG_UPDATES_PER_GEN = "updates_per_gen"
F_CONFIG_LLM_CONCURRENCY = "llm_concurrency"
F_CONFIG_INITIAL_REPUTATION = "initial_reputation"
F_CONFIG_FALLBACK_INIT_COUNT = "fallback_init_count"
F_CONFIG_FALLBACK_MUTATION_COUNT = "fallback_mutation_count"
# Noise. These two names are load-bearing: the invasion / fixation /
# consensus analysis modules (experiments/analysis/**) already write them
# into their own config blocks, and the paper compares evolution results
# against those measurements. Reusing the exact keys is what keeps the two
# families of runs directly comparable.
#   action_error_probability      -- execution error ("trembling hand"):
#                                    the intended action may be mis-executed.
#   observation_error_probability -- perception/assessment error: an observer
#                                    may misperceive an action when judging it.
F_CONFIG_ACTION_ERROR = "action_error_probability"
F_CONFIG_OBSERVATION_ERROR = "observation_error_probability"

# Minimal keys every config record must carry (schema_version is stamped by
# make_config / build_evolution_results).
REQUIRED_CONFIG_FIELDS = (
    F_CONFIG_SCHEMA_VERSION, F_CONFIG_AGENT_TYPE, F_CONFIG_SEED,
    F_CONFIG_POPULATION_SIZE,
)

# Agent families supported by the v2/v3 quantitative interface.
# The signal variant adds optional structured fields; legacy records stay scalar.
AGENT_TYPES = ("agent-type1", "agent-type2", "agent-type2-signal")


# ---------------------------------------------------------------------------
# Version queries
#
# Every "is this record new enough?" decision routes through here, so the
# rule lives in one place instead of being re-derived by each consumer.
# ---------------------------------------------------------------------------
def schema_version(data: dict) -> int:
    """Declared ``schema_version`` of a record, or 0 if absent/unreadable."""
    version = data.get(K_CONFIG, {}).get(F_CONFIG_SCHEMA_VERSION, 0)
    return version if isinstance(version, int) else 0


def is_current_schema(data: dict) -> bool:
    """Whether the record declares exactly the current ``SCHEMA_VERSION``."""
    return schema_version(data) == SCHEMA_VERSION


def require_schema_v4(data: dict, *, source: object = "") -> None:
    """Raise ``EvolutionLogError`` unless the record declares schema >= 4.

    Schema v4 is the version that introduced ``lineage_events`` and the
    per-agent lineage fields; consumers that need a phylogeny call this
    instead of re-checking the version themselves.
    """
    version = schema_version(data)
    if version < 4:
        location = f"{source}: " if source else ""
        raise EvolutionLogError(
            f"{location}schema {version} lacks lineage fields; "
            f"requires schema >= 4. Re-run the evolution with the updated framework."
        )


__all__ = [
    "SCHEMA_VERSION",
    "RESULTS_FILENAME",
    "EvolutionLogError",
    "schema_version",
    "is_current_schema",
    "require_schema_v4",
    # top-level section keys
    "K_TRAJECTORY",
    "K_FINAL_POPULATION",
    "K_LINEAGE_EVENTS",
    "K_CONFIG",
    "TOP_LEVEL_KEYS",
    # trajectory fields
    "F_GENERATION",
    "F_COOPERATION_RATE_MEAN",
    "F_N_INTERACTIONS",
    "F_FITNESS_MEAN",
    "F_FITNESS_MAX",
    "F_POPULATION",
    # per-agent fields
    "F_AGENT_ID",
    "F_CODE",
    "F_FITNESS",
    "F_COOPERATION_RATE",
    "F_SELF_REPUTATION",
    "F_LINEAGE_ID",
    "F_PARENT_ID",
    "F_PARENT_LINEAGE_ID",
    "F_ORIGIN",
    "F_BIRTH_GEN",
    "LINEAGE_EVENT_FIELDS",
    # origin values
    "ORIGIN_INITIAL",
    "ORIGIN_IMITATE",
    "ORIGIN_INDEPENDENT_INIT",
    "ORIGIN_MUTATE",
    "ORIGINS",
    # config fields
    "F_CONFIG_SCHEMA_VERSION",
    "F_CONFIG_AGENT_TYPE",
    "F_CONFIG_POPULATION_SIZE",
    "F_CONFIG_NUM_ROUNDS_PER_GEN",
    "F_CONFIG_BENEFIT",
    "F_CONFIG_COST",
    "F_CONFIG_OBSERVABILITY",
    "F_CONFIG_OBSERVABILITY_P",
    "F_CONFIG_ELITE_COUNT",
    "F_CONFIG_NUM_ELIMINATE",
    "F_CONFIG_TOURNAMENT_SIZE",
    "F_CONFIG_LLM_MODEL",
    "F_CONFIG_SEED",
    "F_CONFIG_USE_BASELINE",
    "F_CONFIG_NUM_GENERATIONS",
    "F_CONFIG_TARGET_INTERACTIONS_PER_GEN",
    "F_CONFIG_FITNESS_WINDOW_FRACTION",
    "F_CONFIG_OBSERVATION_SCHEDULE",
    "F_CONFIG_LLM_THINKING",
    "F_CONFIG_LLM_MAX_TOKENS",
    "F_CONFIG_LEARNING_METHOD",
    "F_CONFIG_FERMI_BETA",
    "F_CONFIG_MUTATION_RATE_ON_ADOPTION",
    "F_CONFIG_IMITATION_LEARNING_MODE",
    "F_CONFIG_UPDATES_PER_GEN",
    "F_CONFIG_LLM_CONCURRENCY",
    "F_CONFIG_INITIAL_REPUTATION",
    "F_CONFIG_FALLBACK_INIT_COUNT",
    "F_CONFIG_FALLBACK_MUTATION_COUNT",
    "F_CONFIG_ACTION_ERROR",
    "F_CONFIG_OBSERVATION_ERROR",
    "REQUIRED_CONFIG_FIELDS",
    "AGENT_TYPES",
]
