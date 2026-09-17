"""The invasion payoff matrix follows the evolution log of the strategy loaded.

Historically the invasion engine paid every interaction from a hardcoded
benefit=2/cost=1 table. That silently tested a strategy evolved at b=3 in a b=2
game. These tests pin the replacement rule:

* a source loaded from an ``evolutionary.json`` carries the matrix recorded in
  its config, and that is what the engine plays;
* canonical norms and hand-written ``.py`` sources carry none, so the archived
  constant is the fallback (keeping archived results reproducible);
* two log-derived sources that disagree raise rather than silently picking one;
* an explicit override wins, which is how the archive is reproduced for
  log-derived sources.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.analysis.invasion.core import (
    BENEFIT,
    COST,
    EvolvedSource,
    norm_source,
    resolve_payoff_matrix,
)
from experiments.analysis.invasion.run_invasion import load_representative_from_path
from experiments.evolution_log import (
    F_CONFIG_POPULATION_SIZE,
    build_evolution_results,
    lineage_event,
    population_entry,
    trajectory_entry,
    write_evolution_json,
)


def _log_source(benefit: float | None, cost: float | None, name: str = "run.json") -> EvolvedSource:
    """A source standing in for one loaded from an evolution log."""
    code = "def observe(*a):\n    return 0.0\n\n\ndef decide(*a):\n    return True\n"
    return EvolvedSource(
        agent_type="agent-type1", path=Path(name), agent_id=-1, lineage_id=-1,
        root_lineage_id=-1, root_family_size=0, fitness=0.0, code=code,
        code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        benefit=benefit, cost=cost,
    )


def _bare_source(name: str = "hand_written.py") -> EvolvedSource:
    return _log_source(None, None, name)


# ---------------------------------------------------------------------------
# Resolution rules
# ---------------------------------------------------------------------------
def test_log_matrix_is_used_when_there_is_one():
    source = _log_source(3.0, 1.0)
    assert resolve_payoff_matrix(source, norm_source("L1")) == (3.0, 1.0)


def test_norm_and_custom_sources_fall_back_to_the_archived_constant():
    assert resolve_payoff_matrix(norm_source("L1"), _bare_source()) == (BENEFIT, COST)


def test_a_log_source_covers_norm_residents():
    """The norm carries no matrix, so the candidate's log decides."""
    assert resolve_payoff_matrix(_log_source(5.0, 2.0), norm_source("ALLD")) == (5.0, 2.0)


def test_the_order_of_the_sources_does_not_matter():
    b3, norm = _log_source(3.0, 1.0), norm_source("L1")
    assert resolve_payoff_matrix(b3, norm) == resolve_payoff_matrix(norm, b3)


@pytest.mark.parametrize("benefit, cost", [(None, 1.0), (3.0, None)])
def test_a_field_the_log_omits_falls_back_independently(benefit, cost):
    resolved = resolve_payoff_matrix(_log_source(benefit, cost))
    assert resolved == (benefit if benefit is not None else BENEFIT,
                        cost if cost is not None else COST)


def test_two_logs_that_agree_resolve():
    assert resolve_payoff_matrix(_log_source(3.0, 1.0), _log_source(3.0, 1.0)) == (3.0, 1.0)


def test_two_logs_that_disagree_raise():
    """One population is paid from one table; there is no third option."""
    with pytest.raises(ValueError, match="different payoff matrices"):
        resolve_payoff_matrix(_log_source(3.0, 1.0), _log_source(2.0, 1.0))


def test_an_explicit_override_beats_the_log():
    source = _log_source(3.0, 1.0)
    assert resolve_payoff_matrix(source, benefit=2.0) == (2.0, 1.0)


def test_an_explicit_override_also_gets_past_a_log_conflict():
    """Forcing a matrix is how a deliberate cross-benefit run is expressed."""
    assert resolve_payoff_matrix(
        _log_source(3.0, 1.0), _log_source(2.0, 1.0), benefit=2.0, cost=1.0
    ) == (2.0, 1.0)


def test_a_conflicting_field_the_override_does_not_cover_still_raises():
    with pytest.raises(ValueError, match="cost"):
        resolve_payoff_matrix(
            _log_source(3.0, 1.0), _log_source(3.0, 2.0), benefit=3.0
        )


# ---------------------------------------------------------------------------
# Loading a real log
# ---------------------------------------------------------------------------
_CODE = "def observe(*a):\n    return 0.0\n\n\ndef decide(*a):\n    return True\n"


def _write_log(
    path: Path, *, benefit: float | None, cost: float | None, schema_version: int = 4,
) -> Path:
    """A minimal schema-v4 log built with the project's own constructors."""
    population = [
        population_entry(
            agent_id=0, code=_CODE, fitness=1.0, cooperation_rate=1.0,
            self_reputation=0.0, lineage_id=0, parent_id=None,
            parent_lineage_id=None, origin="initial", birth_gen=0,
        )
    ]
    config = {
        "agent_type": "agent-type1",
        "seed": 0,
        "label": "test",
        F_CONFIG_POPULATION_SIZE: len(population),
    }
    if benefit is not None:
        config["benefit"] = benefit
    if cost is not None:
        config["cost"] = cost
    data = build_evolution_results(
        trajectory=[trajectory_entry(0, 1.0, 10, 1.0, 1.0, population)],
        final_population=population,
        lineage_events=[lineage_event(0, origin="initial", birth_gen=0)],
        config=config,
    )
    data["config"]["schema_version"] = schema_version
    write_evolution_json(path, data)
    return path


def test_loader_reads_the_matrix_out_of_the_log(tmp_path):
    log = _write_log(tmp_path / "b3.json", benefit=3.0, cost=1.0)
    source = load_representative_from_path("b3", "agent-type1", log)
    assert (source.benefit, source.cost) == (3.0, 1.0)


def test_loader_leaves_the_matrix_unset_when_the_log_omits_it(tmp_path):
    log = _write_log(tmp_path / "legacy.json", benefit=None, cost=None)
    source = load_representative_from_path("legacy", "agent-type1", log)
    assert (source.benefit, source.cost) == (None, None)
    # ...and the unresolved case still resolves to the archived constant.
    assert resolve_payoff_matrix(source) == (BENEFIT, COST)


def test_loader_reads_the_legacy_no_lineage_path_too(tmp_path):
    """Schema-v3 runs take the other representative branch; the matrix still applies."""
    log = _write_log(tmp_path / "v3.json", benefit=2.0, cost=1.0, schema_version=3)
    data = json.loads(log.read_text(encoding="utf-8"))
    del data["lineage_events"]
    log.write_text(json.dumps(data), encoding="utf-8")
    source = load_representative_from_path("v3", "agent-type1", log)
    assert (source.benefit, source.cost) == (2.0, 1.0)
