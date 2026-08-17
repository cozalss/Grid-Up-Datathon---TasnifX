from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gridup.experiment import DataArtifact, ExperimentProvenance, ExperimentRecord
from gridup.stores.sqlite import SQLiteExperimentStore

ROOT = Path(__file__).resolve().parents[1]


def _record(tmp_path, name: str) -> ExperimentRecord:
    data = tmp_path / "train.csv"
    if not data.exists():
        data.write_bytes(b"x,y\n1,2\n")
    provenance = ExperimentProvenance.capture(
        recipe_fingerprint="a" * 64,
        data_artifacts=[DataArtifact.from_path(data)],
        feature_names=["x"],
        fold_fingerprint="b" * 64,
    )
    return ExperimentRecord(
        name=name,
        cv_score=1.0,
        metric="mae",
        model_kind="lightgbm",
        n_features=1,
        fold_scores=[1.0, 1.1],
        params={"objective": "mae"},
        features=["x"],
        provenance=provenance,
    )


def test_store_rejects_incomplete_reproduction_metadata(tmp_path):
    store = SQLiteExperimentStore(tmp_path / "experiments.db")
    incomplete = ExperimentRecord(
        name="bad", cv_score=1.0, metric="mae", model_kind="lightgbm", n_features=1
    )

    with pytest.raises(ValueError, match="provenance|params|features"):
        store.add(incomplete)


def test_store_rejects_malformed_or_inconsistent_provenance(tmp_path):
    record = _record(tmp_path, "invalid")
    object.__setattr__(record.provenance, "recipe_fingerprint", "not-a-sha256")

    with pytest.raises(ValueError, match="recipe_fingerprint"):
        SQLiteExperimentStore(tmp_path / "experiments.db").add(record)

    record = _record(tmp_path, "feature-mismatch")
    object.__setattr__(record.provenance, "feature_names", ("different",))
    with pytest.raises(ValueError, match="feature_names"):
        SQLiteExperimentStore(tmp_path / "experiments.db").add(record)


def test_concurrent_writes_are_transactional_and_lossless(tmp_path):
    path = tmp_path / "experiments.db"
    prototype = _record(tmp_path, "prototype")

    def add_one(index: int) -> str:
        record = ExperimentRecord(
            name=f"run-{index}",
            cv_score=prototype.cv_score,
            metric=prototype.metric,
            model_kind=prototype.model_kind,
            n_features=prototype.n_features,
            fold_scores=list(prototype.fold_scores),
            params=dict(prototype.params),
            features=list(prototype.features),
            provenance=prototype.provenance,
        )
        return SQLiteExperimentStore(path).add(record).run_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        run_ids = list(pool.map(add_one, range(40)))

    records = SQLiteExperimentStore(path).load()
    assert len(records) == 40
    assert len(set(run_ids)) == 40


def test_lb_score_is_a_separate_transactional_event(tmp_path):
    store = SQLiteExperimentStore(tmp_path / "experiments.db")
    record = store.add(_record(tmp_path, "candidate"))

    store.record_lb(record.run_id, 0.75, submitted_at="2026-08-22T10:00:00+00:00")
    loaded = store.load()

    assert loaded[0]["lb_score"] == pytest.approx(0.75)
    assert loaded[0]["submitted_at"] == "2026-08-22T10:00:00+00:00"


@pytest.mark.parametrize("script", ["day_one.py", "smoke_test.py"])
def test_primary_entrypoints_use_transactional_provenance_store(script):
    source = (ROOT / "scripts" / script).read_text(encoding="utf-8")

    assert "SQLiteExperimentStore" in source
    assert "ExperimentProvenance" in source
    assert "FoldPlan.from_folds" in source
    assert "ExperimentLog" not in source
