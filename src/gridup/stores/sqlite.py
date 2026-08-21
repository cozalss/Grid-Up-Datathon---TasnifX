"""Transactional SQLite experiment store with WAL concurrency."""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..experiment import ExperimentRecord

__all__ = ["SQLiteExperimentStore"]


def _is_digest(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _provenance_errors(record: ExperimentRecord) -> list[str]:
    provenance = record.provenance
    if provenance is None:
        return ["provenance"]
    errors: list[str] = []
    if not provenance.data_artifacts:
        errors.append("provenance.data_artifacts")
    if not _is_digest(provenance.recipe_fingerprint):
        errors.append("provenance.recipe_fingerprint")
    if not _is_digest(provenance.fold_fingerprint):
        errors.append("provenance.fold_fingerprint")
    if tuple(record.features) != provenance.feature_names:
        errors.append("provenance.feature_names")
    invalid_artifact = any(
        artifact.size_bytes < 0 or not _is_digest(artifact.sha256)
        for artifact in provenance.data_artifacts
    )
    if invalid_artifact:
        errors.append("provenance.data_artifacts")
    if provenance.git_sha is None or provenance.git_dirty is None:
        errors.append("provenance.git_state")
    # Sentinel ("HESAPLANAMADI:...") KABUL EDILIR, None EDILMEZ. Ikisi ayni
    # sey degil: sentinel "denendi, olmadi, sebebi su" der ve kayitta gorunur;
    # None ise hic denenmedigini ya da sessizce kayboldugunu gosterir.
    # Olculdu (2026-08-21): 10 sn'lik git zaman asimi yuzunden tamamlanmis bir
    # day_one kosusu, submission yazildiktan SONRA atiliyordu.
    if provenance.git_dirty and provenance.git_diff_fingerprint is None:
        errors.append("provenance.git_diff_fingerprint")
    return errors


class SQLiteExperimentStore:
    """Durable experiment storage for concurrent team processes."""

    def __init__(self, path: str | Path, *, strict_provenance: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.strict_provenance = strict_provenance
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextlib.contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        """Islemi commit/rollback EDER ve baglantiyi KAPATIR.

        ``with sqlite3.connect(...) as connection`` yalnizca commit/rollback
        yapar; ``close()`` CAGIRMAZ (CPython belgelerinde acikca yazili). CPython
        refcount'u sayesinde baglanti cogu zaman fonksiyon donusunde finalize
        olur, ama bu dilin garantisi degil: cagiran istisnayi yakalayip saklarsa
        traceback cerceveyi -- dolayisiyla baglantiyi -- canli tutar. O durumda
        WAL modunda acik kalan islem checkpoint'i engeller ve WAL dosyasi buyur.
        "Es zamanli ekip sureci" icin tasarlanmis bir depoda bu varsayim kirilgan.
        """
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _enable_wal(connection: sqlite3.Connection) -> None:
        """Enable persistent WAL mode, retrying SQLite's first-open race."""
        for attempt in range(20):
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                return
            except sqlite3.OperationalError as error:
                if "locked" not in str(error).lower() or attempt == 19:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def _initialize(self) -> None:
        with self._session() as connection:
            self._enable_wal(connection)
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS submissions (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    lb_score REAL NOT NULL,
                    submitted_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_submissions_run
                    ON submissions(run_id, event_id);
                """
            )

    def _validate(self, record: ExperimentRecord) -> None:
        if not self.strict_provenance:
            return
        missing = _provenance_errors(record)
        if not record.params:
            missing.append("params")
        if record.n_features and len(record.features) != record.n_features:
            missing.append("features")
        if not record.fold_scores:
            missing.append("fold_scores")
        if missing:
            raise ValueError("Deney yeniden uretim metadatasi eksik: " + ", ".join(missing))

    def add(self, record: ExperimentRecord) -> ExperimentRecord:
        self._validate(record)
        payload = json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
        with self._session() as connection:
            connection.execute(
                "INSERT INTO runs(run_id, name, created_at, payload_json) VALUES (?, ?, ?, ?)",
                (record.run_id, record.name, record.timestamp, payload),
            )
        return record

    def record_lb(
        self,
        run_id: str,
        lb_score: float,
        *,
        submitted_at: str | None = None,
    ) -> None:
        timestamp = submitted_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._session() as connection:
            exists = connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if exists is None:
                raise KeyError(f"Deney run_id bulunamadi: {run_id}")
            connection.execute(
                "INSERT INTO submissions(run_id, lb_score, submitted_at) VALUES (?, ?, ?)",
                (run_id, float(lb_score), timestamp),
            )

    def load(self) -> list[dict[str, Any]]:
        query = """
            SELECT r.payload_json, s.lb_score, s.submitted_at
            FROM runs AS r
            LEFT JOIN submissions AS s
              ON s.event_id = (
                  SELECT MAX(s2.event_id) FROM submissions AS s2 WHERE s2.run_id = r.run_id
              )
            ORDER BY r.created_at, r.run_id
        """
        with self._session() as connection:
            rows = connection.execute(query).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            record = json.loads(row["payload_json"])
            if row["lb_score"] is not None:
                record["lb_score"] = float(row["lb_score"])
                record["submitted_at"] = str(row["submitted_at"])
            records.append(record)
        return records
