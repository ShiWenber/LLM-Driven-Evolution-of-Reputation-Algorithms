"""Focused tests for the interchangeable strategy-clustering pipeline."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

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


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.codes = [
            "def decide(self): return 'cooperate'",
            "def decide(self): return 'cooperate'  # forgiving",
            "def decide(self): return 'defect'",
            "def decide(self): return 'defect'  # punitive",
        ]

    @patch.object(pipeline, "summarize_cluster_names")
    @patch.object(pipeline, "_load_code_embedding_model")
    def test_code_embedding_is_default_and_uses_shared_llm_naming(
        self, load_model, summarize
    ):
        load_model.return_value = _FakeCodeModel()
        summarize.return_value = {0: "Cooperative", 1: "Defective"}
        X, labels, km, unique, names = pipeline.cluster_codes(
            self.codes, k=2, seed=7, embedding_device="cpu", embedding_cache=False
        )
        self.assertEqual(X.shape[0], len(set(self.codes)))
        self.assertEqual(len(labels), len(unique))
        self.assertEqual(km.n_clusters, 2)
        self.assertEqual(names, summarize.return_value)
        summarize.assert_called_once()

    @patch.object(pipeline, "summarize_cluster_names")
    @patch.object(pipeline, "_load_code_embedding_model")
    def test_code_embedding_reuses_sentence_transformers(self, load_model, summarize):
        fake_model = _FakeCodeModel()
        load_model.return_value = fake_model
        summarize.return_value = {0: "Cooperative", 1: "Defective"}

        X, labels, km, unique, names = pipeline.cluster_codes(
            self.codes + [self.codes[0]],
            k=2,
            seed=7,
            embedding_device="cpu",
            embedding_cache=False,
        )

        self.assertEqual(X.shape, (4, 3))
        self.assertEqual(len(labels), len(unique))
        self.assertEqual(names, summarize.return_value)
        self.assertEqual(len(fake_model.calls), 1)
        embedded_codes, kwargs = fake_model.calls[0]
        self.assertEqual(set(embedded_codes), set(self.codes))
        self.assertTrue(kwargs["normalize_embeddings"])

    @patch.object(pipeline, "summarize_cluster_names")
    @patch.object(pipeline, "_load_code_embedding_model")
    def test_global_pipeline_embeds_unique_codes_once(self, load_model, summarize):
        fake_model = _FakeCodeModel()
        load_model.return_value = fake_model
        summarize.return_value = {0: "Cooperative", 1: "Defective"}
        generations = [
            {F_POPULATION: [{F_CODE: self.codes[0]}, {F_CODE: self.codes[2]}]},
            {F_POPULATION: [{F_CODE: self.codes[0]}, {F_CODE: self.codes[3]}]},
        ]

        state = pipeline.cluster_strategies(
            generations,
            k=2,
            seed=7,
            embedding_device="cpu",
            embedding_cache=False,
        )

        self.assertEqual(state["X"].shape[0], 4)
        self.assertEqual(len(fake_model.calls[0][0]), 3)
        self.assertEqual(state["embedding_method"], "code")
        self.assertEqual(state["embedding_label"], "Code embedding")
        self.assertEqual(state["projection_label"], "PCA")
        # Z is the unique-code PCA projection expanded back to population
        # rows: identical codes share one coordinate, so rows are not
        # mean-centered in the expanded view. Verify row alignment and
        # that repeated codes map to identical coordinates.
        self.assertEqual(state["Z"].shape[0], 4)
        np.testing.assert_allclose(state["Z"][0], state["Z"][2], atol=1e-9)
        summarize.assert_called_once()

    def test_projection_is_centered_pca(self):
        X = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
        Z, _, label = pipeline.project_embeddings(X, seed=7)
        self.assertEqual(Z.shape, (4, 2))
        self.assertEqual(label, "PCA")
        np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-7)

    def test_llm_json_parser_accepts_fences(self):
        parsed = pipeline._extract_json_object(
            '```json\n{"clusters":{"0":"Reciprocal cooperation"}}\n```'
        )
        self.assertEqual(parsed["clusters"]["0"], "Reciprocal cooperation")

    @patch.object(pipeline, "_load_code_embedding_model")
    def test_sqlite_embedding_cache_skips_model_on_exact_hit(self, load_model):
        fake_model = _FakeCodeModel()
        load_model.return_value = fake_model
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "analysis.sqlite3"
            first, first_model = pipeline.embed_codes(
                self.codes,
                embedding_device="cpu",
                embedding_cache_path=cache_path,
            )
            self.assertIs(first_model, fake_model)
            load_model.reset_mock()
            second, second_model = pipeline.embed_codes(
                self.codes,
                embedding_device="cpu",
                embedding_cache_path=cache_path,
            )

            np.testing.assert_array_equal(first, second)
            self.assertIsNone(second_model)
            load_model.assert_not_called()
            self.assertEqual(len(fake_model.calls), 1)

    @patch.object(pipeline, "_load_code_embedding_model")
    def test_cache_key_preserves_complete_exact_code(self, load_model):
        fake_model = _FakeCodeModel()
        load_model.return_value = fake_model
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "analysis.sqlite3"
            pipeline.embed_codes(
                ["x = 1"], embedding_device="cpu",
                embedding_cache_path=cache_path,
            )
            pipeline.embed_codes(
                ["x = 1\n"], embedding_device="cpu",
                embedding_cache_path=cache_path,
            )
        self.assertEqual(len(fake_model.calls), 2)

    def test_complete_code_overflow_uses_all_tokens(self):
        model = _FakeCodeModel()
        code = "x" * 9000
        _, owners, weights, _, token_counts, chunk_counts = (
            pipeline._prepare_complete_code(model, [code])
        )
        self.assertEqual(owners, [0, 0])
        self.assertEqual(token_counts, [9000])
        self.assertEqual(chunk_counts, [2])
        self.assertEqual(sum(weights), 9000)

    def test_occurrences_deduplicate_across_reindex(self):
        from experiments.analysis.clustering.cache import AnalysisCache

        with TemporaryDirectory() as directory:
            cache = AnalysisCache(Path(directory) / "analysis.sqlite3")
            record = {
                "code": "x = 1", "source_path": Path(directory) / "run.json",
                "generation": 2, "agent_id": 7, "lineage_id": 3,
            }
            cache.put_occurrences([record, record])
            with cache.connection() as connection:
                code_count = connection.execute("SELECT count(*) FROM codes").fetchone()[0]
                occurrence_count = connection.execute(
                    "SELECT count(*) FROM code_occurrences"
                ).fetchone()[0]
            self.assertEqual(code_count, 1)
            self.assertEqual(occurrence_count, 1)

    @patch.object(pipeline, "summarize_cluster_names")
    @patch.object(pipeline, "_load_code_embedding_model")
    def test_clustering_run_persists_labels_generation_stats_and_artifact(
        self, load_model, summarize
    ):
        from experiments.analysis.clustering.cache import AnalysisCache

        summarize.return_value = {0: "Cooperative", 1: "Defective"}
        load_model.return_value = _FakeCodeModel()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "experiment_a" / "evolutionary.json"
            cache_path = root / "analysis.sqlite3"
            generations = [
                {F_GENERATION: 0, F_POPULATION: [
                    {F_AGENT_ID: 1, F_CODE: self.codes[0]},
                    {F_AGENT_ID: 2, F_CODE: self.codes[2]},
                ]},
                {F_GENERATION: 1, F_POPULATION: [
                    {F_AGENT_ID: 1, F_CODE: self.codes[0]},
                    {F_AGENT_ID: 2, F_CODE: self.codes[3]},
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
            self.assertEqual(assignments, 4)
            self.assertEqual(stats, [(0, 2, 1.0), (1, 2, 1.0)])
            self.assertEqual(artifacts, [("composition", str(artifact.resolve()))])


if __name__ == "__main__":
    unittest.main()
