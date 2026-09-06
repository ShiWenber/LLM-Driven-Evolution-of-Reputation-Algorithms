from pathlib import Path

import pytest

from experiments.analysis.invasion.run_n100_invasion_count_sweep import (
    noisy_output,
    parse_sources,
)


def test_noisy_output_encodes_both_error_probabilities():
    path = noisy_output(0.01, 0.05)
    assert path.name == "n100_noisy_invasion_count_sweep_ae0p01_oe0p05"


def test_explicit_source_requires_label_type_and_path():
    with pytest.raises(ValueError, match="LABEL=AGENT_TYPE=PATH"):
        parse_sources(["seed0=missing-path-part"])


def test_explicit_source_labels_are_unique(monkeypatch):
    monkeypatch.setattr(
        "experiments.analysis.invasion.run_n100_invasion_count_sweep."
        "load_representative_from_path",
        lambda label, agent_type, path: (label, agent_type, Path(path)),
    )
    with pytest.raises(ValueError, match="unique"):
        parse_sources([
            "seed0=agent-type1=first.json",
            "seed0=agent-type1=second.json",
        ])
