"""SQLite archive for code occurrences and derived analysis results."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parents[3]
    / "results"
    / ".analysis_cache"
    / "strategy_analysis.sqlite3"
)


def stable_hash(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


class AnalysisCache:
    """Content-addressed archive shared by all local experiments."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_CACHE_PATH

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS codes (
                code_hash TEXT PRIMARY KEY,
                code_text TEXT NOT NULL,
                byte_length INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS code_occurrences (
                occurrence_key TEXT PRIMARY KEY,
                code_hash TEXT NOT NULL REFERENCES codes(code_hash),
                experiment_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                generation INTEGER,
                agent_id TEXT,
                lineage_id TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_occurrences_code_hash
                ON code_occurrences(code_hash);
            CREATE INDEX IF NOT EXISTS idx_occurrences_experiment
                ON code_occurrences(experiment_id);

            CREATE TABLE IF NOT EXISTS code_embeddings_v2 (
                cache_key TEXT PRIMARY KEY,
                code_hash TEXT NOT NULL REFERENCES codes(code_hash),
                method TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                backend TEXT NOT NULL,
                device_type TEXT NOT NULL,
                compute_dtype TEXT NOT NULL,
                input_config_json TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                token_count INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_embeddings_v2_code_hash
                ON code_embeddings_v2(code_hash);

            CREATE TABLE IF NOT EXISTS cluster_names (
                cache_key TEXT PRIMARY KEY,
                llm_model TEXT NOT NULL,
                request_json TEXT NOT NULL,
                names_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS leaf_names (
                code_hash TEXT PRIMARY KEY,
                leaf_name TEXT NOT NULL,
                llm_model TEXT NOT NULL,
                prompt_version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS clustering_runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                embedding_method TEXT NOT NULL,
                model_name TEXT,
                cluster_count INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                parameters_json TEXT NOT NULL,
                cluster_names_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_clustering_runs_experiment
                ON clustering_runs(experiment_id, created_at);

            CREATE TABLE IF NOT EXISTS cluster_assignments (
                run_id TEXT NOT NULL REFERENCES clustering_runs(run_id) ON DELETE CASCADE,
                assignment_index INTEGER NOT NULL,
                code_hash TEXT NOT NULL REFERENCES codes(code_hash),
                generation INTEGER,
                agent_id TEXT,
                lineage_id TEXT,
                experiment_id TEXT,
                source_path TEXT,
                cluster_id INTEGER NOT NULL,
                PRIMARY KEY (run_id, assignment_index)
            );
            CREATE INDEX IF NOT EXISTS idx_cluster_assignments_generation
                ON cluster_assignments(run_id, generation, cluster_id);

            CREATE TABLE IF NOT EXISTS generation_cluster_stats (
                run_id TEXT NOT NULL REFERENCES clustering_runs(run_id) ON DELETE CASCADE,
                generation INTEGER NOT NULL,
                cluster_id INTEGER NOT NULL,
                agent_count INTEGER NOT NULL,
                fraction REAL NOT NULL,
                PRIMARY KEY (run_id, generation, cluster_id)
            );

            CREATE TABLE IF NOT EXISTS experiment_generation_cluster_stats (
                run_id TEXT NOT NULL REFERENCES clustering_runs(run_id) ON DELETE CASCADE,
                experiment_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                cluster_id INTEGER NOT NULL,
                agent_count INTEGER NOT NULL,
                fraction REAL NOT NULL,
                PRIMARY KEY (run_id, experiment_id, generation, cluster_id)
            );

            CREATE TABLE IF NOT EXISTS analysis_artifacts (
                artifact_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES clustering_runs(run_id) ON DELETE CASCADE,
                artifact_type TEXT NOT NULL,
                path TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (run_id, artifact_type, path)
            );
            """
        )
        assignment_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(cluster_assignments)")
        }
        if "experiment_id" not in assignment_columns:
            connection.execute("ALTER TABLE cluster_assignments ADD COLUMN experiment_id TEXT")
        if "source_path" not in assignment_columns:
            connection.execute("ALTER TABLE cluster_assignments ADD COLUMN source_path TEXT")
        return connection

    @contextmanager
    def connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def put_codes(self, codes: list[str]) -> list[str]:
        hashes = [code_hash(code) for code in codes]
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (digest, code, len(code.encode("utf-8")), now)
            for digest, code in zip(hashes, codes)
        ]
        with self.connection() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO codes VALUES (?, ?, ?, ?)", rows
            )
        return hashes

    def put_occurrences(self, records: list[dict]) -> None:
        if not records:
            return
        codes = [str(record["code"]) for record in records]
        hashes = self.put_codes(codes)
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for digest, record in zip(hashes, records):
            source = str(Path(record["source_path"]).resolve())
            identity = {
                "source_path": source,
                "generation": record.get("generation"),
                "agent_id": record.get("agent_id"),
                "lineage_id": record.get("lineage_id"),
                "code_hash": digest,
            }
            rows.append(
                (
                    stable_hash(identity),
                    digest,
                    str(record.get("experiment_id") or Path(source).parent.name),
                    source,
                    record.get("generation"),
                    None if record.get("agent_id") is None else str(record["agent_id"]),
                    None if record.get("lineage_id") is None else str(record["lineage_id"]),
                    json.dumps(record.get("metadata", {}), ensure_ascii=False, sort_keys=True),
                    now,
                )
            )
        with self.connection() as connection:
            connection.executemany(
                """
                INSERT INTO code_occurrences VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(occurrence_key) DO UPDATE SET
                    metadata_json=excluded.metadata_json
                """,
                rows,
            )

    def get_embeddings(self, keys: list[str]) -> dict[str, np.ndarray]:
        if not keys or not self.path.exists():
            return {}
        found = {}
        with self.connection() as connection:
            for start in range(0, len(keys), 500):
                batch = keys[start : start + 500]
                marks = ",".join("?" for _ in batch)
                rows = connection.execute(
                    "SELECT cache_key, dimension, vector FROM code_embeddings_v2 "
                    f"WHERE cache_key IN ({marks})",
                    batch,
                ).fetchall()
                found.update(
                    {
                        key: np.frombuffer(blob, dtype=np.float32, count=dim).copy()
                        for key, dim, blob in rows
                    }
                )
        return found

    def put_embeddings(self, rows: list[dict]) -> None:
        if not rows:
            return
        now = datetime.now(timezone.utc).isoformat()
        values = []
        for row in rows:
            vector = np.asarray(row["vector"], dtype=np.float32).ravel()
            config = row["config"]
            values.append(
                (
                    row["key"], row["code_hash"], "code", config["model"],
                    config["revision"], config["backend"], config["device"],
                    config["compute_dtype"],
                    json.dumps(config, ensure_ascii=False, sort_keys=True),
                    vector.size, vector.tobytes(), row["token_count"],
                    row["chunk_count"], now,
                )
            )
        with self.connection() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO code_embeddings_v2
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )

    def get_cluster_names(self, key: str) -> dict[int, str] | None:
        if not self.path.exists():
            return None
        with self.connection() as connection:
            row = connection.execute(
                "SELECT names_json FROM cluster_names WHERE cache_key = ?", (key,)
            ).fetchone()
        return None if row is None else {int(k): str(v) for k, v in json.loads(row[0]).items()}

    def put_cluster_names(
        self, *, key: str, llm_model: str, request: dict, names: dict[int, str]
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO cluster_names VALUES (?, ?, ?, ?, ?)
                """,
                (
                    key, llm_model,
                    json.dumps(request, ensure_ascii=False, sort_keys=True),
                    json.dumps(names, ensure_ascii=False, sort_keys=True), now,
                ),
            )

    def get_leaf_names(
        self, code_hashes: list[str]
    ) -> dict[str, str]:
        """Return cached LLM leaf names keyed by code hash."""
        if not code_hashes or not self.path.exists():
            return {}
        found = {}
        with self.connection() as connection:
            for start in range(0, len(code_hashes), 500):
                batch = code_hashes[start : start + 500]
                marks = ",".join("?" for _ in batch)
                rows = connection.execute(
                    "SELECT code_hash, leaf_name FROM leaf_names "
                    f"WHERE code_hash IN ({marks})",
                    batch,
                ).fetchall()
                found.update({row[0]: row[1] for row in rows})
        return found

    def put_leaf_names(
        self, *, llm_model: str, prompt_version: int,
        names: dict[str, str],
    ) -> None:
        """Cache LLM names for strategy leaves (keyed by code hash)."""
        if not names:
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (digest, name, llm_model, prompt_version, now)
            for digest, name in names.items()
        ]
        with self.connection() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO leaf_names VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def put_clustering_run(
        self, *, source_path: str | Path, embedding_method: str,
        model_name: str | None, cluster_count: int, seed: int,
        parameters: dict, cluster_names: dict[int, str], assignments: list[dict],
    ) -> str:
        """Store one immutable clustering result and its per-generation composition."""
        source = str(Path(source_path).resolve())
        experiment_id = Path(source).parent.name
        run_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        codes = [str(row["code"]) for row in assignments]
        hashes = self.put_codes(codes)
        assignment_rows = []
        counts: dict[tuple[int, int], int] = {}
        totals: dict[int, int] = {}
        experiment_counts: dict[tuple[str, int, int], int] = {}
        experiment_totals: dict[tuple[str, int], int] = {}
        for index, (digest, row) in enumerate(zip(hashes, assignments)):
            generation = row.get("generation")
            cluster_id = int(row["cluster_id"])
            assignment_rows.append((
                run_id, index, digest, generation,
                None if row.get("agent_id") is None else str(row["agent_id"]),
                None if row.get("lineage_id") is None else str(row["lineage_id"]),
                str(row.get("experiment_id") or experiment_id),
                str(Path(row.get("source_path") or source).resolve()), cluster_id,
            ))
            if generation is not None:
                generation = int(generation)
                counts[(generation, cluster_id)] = counts.get((generation, cluster_id), 0) + 1
                totals[generation] = totals.get(generation, 0) + 1
                row_experiment = str(row.get("experiment_id") or experiment_id)
                experiment_counts[(row_experiment, generation, cluster_id)] = (
                    experiment_counts.get((row_experiment, generation, cluster_id), 0) + 1
                )
                experiment_totals[(row_experiment, generation)] = (
                    experiment_totals.get((row_experiment, generation), 0) + 1
                )
        stat_rows = [
            (run_id, generation, cluster_id, count, count / totals[generation])
            for (generation, cluster_id), count in sorted(counts.items())
        ]
        experiment_stat_rows = [
            (run_id, row_experiment, generation, cluster_id, count,
             count / experiment_totals[(row_experiment, generation)])
            for (row_experiment, generation, cluster_id), count
            in sorted(experiment_counts.items())
        ]
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO clustering_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, experiment_id, source, embedding_method, model_name,
                 cluster_count, seed,
                 json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                 json.dumps(cluster_names, ensure_ascii=False, sort_keys=True), now),
            )
            connection.executemany(
                """INSERT INTO cluster_assignments
                   (run_id, assignment_index, code_hash, generation, agent_id,
                    lineage_id, experiment_id, source_path, cluster_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                assignment_rows,
            )
            connection.executemany(
                "INSERT INTO generation_cluster_stats VALUES (?, ?, ?, ?, ?)",
                stat_rows,
            )
            connection.executemany(
                "INSERT INTO experiment_generation_cluster_stats VALUES (?, ?, ?, ?, ?, ?)",
                experiment_stat_rows,
            )
        return run_id

    def put_artifact(
        self, *, run_id: str | None, artifact_type: str, path: str | Path,
        metadata: dict | None = None,
    ) -> None:
        if not run_id:
            return
        resolved = Path(path).resolve()
        details = dict(metadata or {})
        if resolved.exists():
            details.update({"size_bytes": resolved.stat().st_size, "suffix": resolved.suffix})
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO analysis_artifacts
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (stable_hash({"run_id": run_id, "type": artifact_type,
                              "path": str(resolved)}),
                 run_id, artifact_type, str(resolved),
                 json.dumps(details, ensure_ascii=False, sort_keys=True), now),
            )

    def get_clustering_run_labels(self, run_id: str) -> tuple[dict[str, int], dict[int, str]]:
        """Return exact code-text labels and cluster names stored for a run."""
        if not self.path.exists():
            raise FileNotFoundError(f"Analysis cache not found: {self.path}")
        with self.connection() as connection:
            run = connection.execute(
                "SELECT cluster_names_json FROM clustering_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown clustering run_id: {run_id}")
            rows = connection.execute(
                """SELECT c.code_text, a.cluster_id
                   FROM cluster_assignments AS a
                   JOIN codes AS c ON c.code_hash = a.code_hash
                   WHERE a.run_id = ?
                   GROUP BY c.code_text, a.cluster_id""",
                (run_id,),
            ).fetchall()
        labels: dict[str, int] = {}
        for code, cluster_id in rows:
            cluster_id = int(cluster_id)
            previous = labels.setdefault(str(code), cluster_id)
            if previous != cluster_id:
                raise ValueError(f"Code has conflicting labels in run {run_id}")
        names = {int(key): str(value) for key, value in json.loads(run[0]).items()}
        return labels, names
