"""Focused pytest suite for the interchangeable strategy-clustering pipeline.

Migrated from ``experiments/analysis/clustering/test_pipeline.py``
(unittest) and converted to pytest style.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from experiments.analysis.clustering import pipeline
from experiments.evolution_log import F_AGENT_ID, F_CODE, F_GENERATION, F_POPULATION


class _FakeCodeModel:
    def __init__(self):
        self.calls = []
        self.tokenizer = _FakeTokenizer()

    def encode(self, codes, **kwargs):
        self.calls.append((list(codes), kwargs))
        rows = []
        for code in codes:
            if "cooperate" in code:
                rows.append([1.0, 0.0, 0.1])
            elif "defect" in code:
                rows.append([0.0, 1.0, 0.1])
            else:
                rows.append([0.1, 0.0, 1.0])
        values = np.asarray(rows, dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode("utf-8", errors="replace")


@pytest.fixture
def codes():
    return [
        "def decide(self): return 'cooperate'",
        "def decide(self): return 'cooperate'  # forgiving",
        "def decide(self): return 'defect'",
        "def decide(self): return 'defect'  # punitive",
    ]


@pytest.fixture
def patched_pipeline(monkeypatch):
    """Patch LLM-naming + embedding-loading entry points with fakes.

    Returns a helper ``patched(load_model_return, summarize_return)``
    yielding ``(load_model_mock, summarize_mock)``.
    """

    def _patch(load_model_return=None, summarize_return=None):
        load_model = Mock()
        load_model.return_value = load_model_return
        summarize = Mock()
        summarize.return_value = summarize_return
        monkeypatch.setattr(pipeline, "_load_code_embedding_model", load_model)
        monkeypatch.setattr(pipeline, "summarize_cluster_names", summarize)
        return load_model, summarize

    return _patch


def test_code_embedding_is_default_and_uses_shared_llm_naming(codes, patched_pipeline):
    load_model, summarize = patched_pipeline(
        load_model_return=_FakeCodeModel(),
        summarize_return={0: "Cooperative", 1: "Defective"},
    )
    X, labels, km, unique, names = pipeline.cluster_codes(
        codes, k=2, seed=7, embedding_device="cpu", embedding_cache=False
    )
    assert X.shape[0] == len(set(codes))
    assert len(labels) == len(unique)
    assert km.n_clusters == 2
    assert names == summarize.return_value
    summarize.assert_called_once()


def test_code_embedding_reuses_sentence_transformers(codes, patched_pipeline):
    fake_model = _FakeCodeModel()
    load_model, summarize = patched_pipeline(
        load_model_return=fake_model,
        summarize_return={0: "Cooperative", 1: "Defective"},
    )
    X, labels, km, unique, names = pipeline.cluster_codes(
        codes + [codes[0]],
        k=2,
        seed=7,
        embedding_device="cpu",
        embedding_cache=False,
    )
    assert X.shape == (4, 3)
    assert len(labels) == len(unique)
    assert names == summarize.return_value
    assert len(fake_model.calls) == 1
    embedded_codes, kwargs = fake_model.calls[0]
    assert set(embedded_codes) == set(codes)
    assert kwargs["normalize_embeddings"]


def test_global_pipeline_embeds_unique_codes_once(codes, patched_pipeline):
    fake_model = _FakeCodeModel()
    load_model, summarize = patched_pipeline(
        load_model_return=fake_model,
        summarize_return={0: "Cooperative", 1: "Defective"},
    )
    generations = [
        {F_POPULATION: [{F_CODE: codes[0]}, {F_CODE: codes[2]}]},
        {F_POPULATION: [{F_CODE: codes[0]}, {F_CODE: codes[3]}]},
    ]

    state = pipeline.cluster_strategies(
        generations,
        k=2,
        seed=7,
        embedding_device="cpu",
        embedding_cache=False,
    )

    assert state["X"].shape[0] == 4
    assert len(fake_model.calls[0][0]) == 3
    assert state["embedding_method"] == "code"
    assert state["embedding_label"] == "Code embedding"
    assert state["projection_label"] == "PCA"
    # Z is the unique-code PCA projection expanded back to population
    # rows: identical codes share one coordinate, so rows are not
    # mean-centered in the expanded view. Verify row alignment and
    # that repeated codes map to identical coordinates.
    assert state["Z"].shape[0] == 4
    np.testing.assert_allclose(state["Z"][0], state["Z"][2], atol=1e-9)
    summarize.assert_called_once()


def test_projection_is_centered_pca():
    X = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
    Z, _, label = pipeline.project_embeddings(X, seed=7)
    assert Z.shape == (4, 2)
    assert label == "PCA"
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-7)


def test_llm_json_parser_accepts_fences():
    parsed = pipeline._extract_json_object(
        '```json\n{"clusters":{"0":"Reciprocal cooperation"}}\n```'
    )
    assert parsed["clusters"]["0"] == "Reciprocal cooperation"


def test_sqlite_embedding_cache_skips_model_on_exact_hit(codes, patched_pipeline, tmp_path):
    fake_model = _FakeCodeModel()
    load_model, _ = patched_pipeline(load_model_return=fake_model)
    cache_path = tmp_path / "analysis.sqlite3"
    first, first_model = pipeline.embed_codes(
        codes,
        embedding_device="cpu",
        embedding_cache_path=cache_path,
    )
    assert first_model is fake_model
    load_model.reset_mock()
    second, second_model = pipeline.embed_codes(
        codes,
        embedding_device="cpu",
        embedding_cache_path=cache_path,
    )

    np.testing.assert_array_equal(first, second)
    assert second_model is None
    load_model.assert_not_called()
    assert len(fake_model.calls) == 1


def test_cache_key_preserves_complete_exact_code(patched_pipeline, tmp_path):
    fake_model = _FakeCodeModel()
    patched_pipeline(load_model_return=fake_model)
    cache_path = tmp_path / "analysis.sqlite3"
    pipeline.embed_codes(
        ["x = 1"], embedding_device="cpu",
        embedding_cache_path=cache_path,
    )
    pipeline.embed_codes(
        ["x = 1\n"], embedding_device="cpu",
        embedding_cache_path=cache_path,
    )
    assert len(fake_model.calls) == 2


def test_complete_code_overflow_uses_all_tokens():
    model = _FakeCodeModel()
    code = "x" * 9000
    _, owners, weights, _, token_counts, chunk_counts = (
        pipeline._prepare_complete_code(model, [code])
    )
    assert owners == [0, 0]
    assert token_counts == [9000]
    assert chunk_counts == [2]
    assert sum(weights) == 9000


def test_occurrences_deduplicate_across_reindex(tmp_path):
    from experiments.analysis.clustering.cache import AnalysisCache

    cache = AnalysisCache(tmp_path / "analysis.sqlite3")
    record = {
        "code": "x = 1", "source_path": tmp_path / "run.json",
        "generation": 2, "agent_id": 7, "lineage_id": 3,
    }
    cache.put_occurrences([record, record])
    with cache.connection() as connection:
        code_count = connection.execute("SELECT count(*) FROM codes").fetchone()[0]
        occurrence_count = connection.execute(
            "SELECT count(*) FROM code_occurrences"
        ).fetchone()[0]
    assert code_count == 1
    assert occurrence_count == 1


def test_clustering_run_persists_labels_generation_stats_and_artifact(
    codes, patched_pipeline, tmp_path
):
    from experiments.analysis.clustering.cache import AnalysisCache

    _, summarize = patched_pipeline(
        load_model_return=_FakeCodeModel(),
        summarize_return={0: "Cooperative", 1: "Defective"},
    )
    root = tmp_path
    source = root / "experiment_a" / "evolutionary.json"
    cache_path = root / "analysis.sqlite3"
    generations = [
        {F_GENERATION: 0, F_POPULATION: [
            {F_AGENT_ID: 1, F_CODE: codes[0]},
            {F_AGENT_ID: 2, F_CODE: codes[2]},
        ]},
        {F_GENERATION: 1, F_POPULATION: [
            {F_AGENT_ID: 1, F_CODE: codes[0]},
            {F_AGENT_ID: 2, F_CODE: codes[3]},
        ]},
    ]
    state = pipeline.cluster_strategies(
        generations, k=2, seed=7, analysis_source_path=source,
        embedding_cache_path=cache_path, embedding_device="cpu",
    )
    artifact = root / "composition.png"
    artifact.write_bytes(b"png")
    cache = AnalysisCache(cache_path)
    cache.put_artifact(
        run_id=state["analysis_run_id"], artifact_type="composition",
        path=artifact,
    )
    with cache.connection() as connection:
        assignments = connection.execute(
            "SELECT count(*) FROM cluster_assignments"
        ).fetchone()[0]
        stats = connection.execute(
            "SELECT generation, sum(agent_count), sum(fraction) "
            "FROM generation_cluster_stats GROUP BY generation"
        ).fetchall()
        artifacts = connection.execute(
            "SELECT artifact_type, path FROM analysis_artifacts"
        ).fetchall()
    assert assignments == 4
    assert stats == [(0, 2, 1.0), (1, 2, 1.0)]
    assert artifacts == [("composition", str(artifact.resolve()))]
